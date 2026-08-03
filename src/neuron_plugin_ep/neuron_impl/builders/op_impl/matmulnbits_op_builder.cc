// Copyright (c) MediaTek Inc. All rights reserved.
// Licensed under the MIT License.

// MatMulNBits op builder for the Neuron EP (standalone plugin EP version).
//
// Strategy: dequantize INT4 weight to FP32 at model-build time, then use
// NEURON_FULLY_CONNECTED with TENSOR_FLOAT32 I/O.  The MDLA FP16 relaxation
// (relaxComputationFloat32toFloat16 = true) converts everything to FP16
// internally — exactly the same path that is used by the MediaTek TFLite
// delegate for the "asym4W" FC layers.
//
// MDLA FULLY_CONNECTED constraints:
//   • Input MUST be exactly 2D [batch, K].  3D inputs (e.g. [1, seq, K])
//     cause incorrect MDLA results even though the driver accepts them.
//     We add explicit RESHAPE ops before/after FC.
//   • NPU requires fully-static shapes.
//   • Batch size in {1,2,4,8}; other values are auto-converted to CONV_2D.
//
// Supported inputs:
//   0  A      – TENSOR_FLOAT32  [..., K]    (dynamic leading dims → CPU)
//   1  B      – TENSOR_UINT8   [N, n_blocks, blob_size]  constant initializer
//   2  scales – TENSOR_FLOAT32 [N * n_blocks]             constant initializer
//   3  zp     – TENSOR_UINT8   (nibble-packed, optional)  constant initializer
//   4  g_idx  – (deprecated, ignored)
//   5  bias   – TENSOR_FLOAT32 [N]                        constant initializer

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

#include "base_op_builder.h"

namespace onnxruntime {
namespace neuron {

using namespace onnxruntime::neuron::op_builder_helpers;
using namespace onnxruntime::neuron::wrapper;

class MatMulNBitsOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder,
                             const onnx::NodeProto& node,
                             const NeuronGraphContext& ctx) const override;

 private:
  NeuronStatus AddToModelBuilderImpl(ModelBuilder& model_builder,
                                     const onnx::NodeProto& node,
                                     const NeuronGraphContext& ctx) const override;


  bool HasSupportedInputsImpl(const onnx::NodeProto& node,
                                    const NeuronGraphContext& ctx,
                                    const OpSupportCheckParams& params) const override;

  bool IsComputeHeavy(const onnx::NodeProto& /*node*/) const override { return true; }

  bool IsNodeUnitTypeSupported(const onnx::NodeProto& /*node*/) const override { return true; }
};

// ---------------------------------------------------------------------------
// AddInitializersToSkip
// ---------------------------------------------------------------------------

void MatMulNBitsOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder,
                                                  const onnx::NodeProto& node,
                                                  const NeuronGraphContext& /*ctx*/) const {
  if (node.input_size() > 1) model_builder.AddInitializerToSkip(node.input(1));  // B (packed uint8)
  if (node.input_size() > 2) model_builder.AddInitializerToSkip(node.input(2));  // scales
  if (node.input_size() > 3 && !node.input(3).empty())
    model_builder.AddInitializerToSkip(node.input(3));  // zero_points
  if (node.input_size() > 4 && !node.input(4).empty())
    model_builder.AddInitializerToSkip(node.input(4));  // g_idx (deprecated)
  if (node.input_size() > 5 && !node.input(5).empty())
    model_builder.AddInitializerToSkip(node.input(5));  // bias
}

// ---------------------------------------------------------------------------
// HasSupportedInputsImpl
// ---------------------------------------------------------------------------

bool MatMulNBitsOpBuilder::HasSupportedInputsImpl(
    const onnx::NodeProto& node,
    const NeuronGraphContext& ctx,
    const OpSupportCheckParams& /*params*/) const {
  // Input A must be float32.
  if (node.input_size() < 1) return false;
  int32_t elem_type = ctx.GetElementType(node.input(0));
  if (elem_type != onnx::TensorProto::FLOAT) {
    LOGS_NEURON(VERBOSE) << "MatMulNBits: input A must be FLOAT, got " << elem_type;
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// AddToModelBuilderImpl  —  FP32 FULLY_CONNECTED path
// ---------------------------------------------------------------------------

NeuronStatus MatMulNBitsOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                          const onnx::NodeProto& node,
                                                          const NeuronGraphContext& ctx) const {
  const int64_t K          = NeuronGraphContext::GetAttrInt(node, "K", 0);
  const int64_t N          = NeuronGraphContext::GetAttrInt(node, "N", 0);
  const int64_t block_size = NeuronGraphContext::GetAttrInt(node, "block_size", 32);

  NEURON_RETURN_IF_NOT(K > 0 && N > 0, "MatMulNBits: K and N must be > 0");

  const int64_t n_blocks  = (K + block_size - 1) / block_size;
  const int64_t blob_size = block_size / 2;

  auto& shaper            = model_builder.GetShaper();
  const auto& operand_indices = model_builder.GetOperandIndices();

  // -----------------------------------------------------------------------
  // 1. Read packed B and scales from constant initializers
  // -----------------------------------------------------------------------
  const auto* b_proto     = ctx.GetInitializer(node.input(1));
  const auto* scales_proto = ctx.GetInitializer(node.input(2));
  NEURON_RETURN_IF_NOT(b_proto && scales_proto,
                       "MatMulNBits: B and scales must be constant initializers");

  // Read raw bytes of B (UINT8)
  std::vector<uint8_t> b_data;
  {
    if (!b_proto->raw_data().empty()) {
      const auto& raw = b_proto->raw_data();
      b_data.assign(reinterpret_cast<const uint8_t*>(raw.data()),
                    reinterpret_cast<const uint8_t*>(raw.data()) + raw.size());
    } else {
      b_data.reserve(b_proto->int32_data_size());
      for (auto v : b_proto->int32_data())
        b_data.push_back(static_cast<uint8_t>(v));
    }
  }

  // Read float scales
  std::vector<float> scales_data;
  NEURON_RETURN_IF_ERROR(NeuronGraphContext::GetFloatData(*scales_proto, scales_data));

  // -----------------------------------------------------------------------
  // 2. Read optional nibble-packed zero points
  // -----------------------------------------------------------------------
  std::vector<uint8_t> zp_bytes;
  const bool has_zp = node.input_size() > 3 && !node.input(3).empty();
  if (has_zp) {
    const auto* zp_proto = ctx.GetInitializer(node.input(3));
    if (zp_proto) {
      if (!zp_proto->raw_data().empty()) {
        const auto& raw = zp_proto->raw_data();
        zp_bytes.assign(reinterpret_cast<const uint8_t*>(raw.data()),
                        reinterpret_cast<const uint8_t*>(raw.data()) + raw.size());
      } else {
        zp_bytes.reserve(zp_proto->int32_data_size());
        for (auto v : zp_proto->int32_data())
          zp_bytes.push_back(static_cast<uint8_t>(v));
      }
    }
  }

  // -----------------------------------------------------------------------
  // 3. Dequantize INT4 → FP32  [N, K]
  // -----------------------------------------------------------------------
  std::vector<float> weight_fp32(static_cast<size_t>(N * K), 0.0f);

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t blk = 0; blk < n_blocks; ++blk) {
      const float scale = scales_data[static_cast<size_t>(n * n_blocks + blk)];

      uint8_t zp = 8u;
      if (has_zp && !zp_bytes.empty()) {
        const int64_t zp_bytes_per_row = (n_blocks + 1) / 2;
        const uint8_t zp_byte = zp_bytes[static_cast<size_t>(n * zp_bytes_per_row + blk / 2)];
        zp = (blk % 2 == 0) ? (zp_byte & 0x0Fu) : ((zp_byte >> 4) & 0x0Fu);
      }

      const int64_t b_block_base = (n * n_blocks + blk) * blob_size;

      for (int64_t i = 0; i < block_size; ++i) {
        const int64_t k = blk * block_size + i;
        if (k >= K) break;

        const uint8_t byte_val = b_data[static_cast<size_t>(b_block_base + i / 2)];
        const uint8_t val      = (i % 2 == 0) ? (byte_val & 0x0Fu) : ((byte_val >> 4) & 0x0Fu);

        weight_fp32[static_cast<size_t>(n * K + k)] =
            (static_cast<float>(val) - static_cast<float>(zp)) * scale;
      }
    }
  }

  // -----------------------------------------------------------------------
  // 4. Register FP32 weight operand [N, K]
  // -----------------------------------------------------------------------
  const std::string weight_name =
      model_builder.GetUniqueName(node.name() + "_matmulnbits_w_fp32");
  const Shape weight_shape = {static_cast<uint32_t>(N), static_cast<uint32_t>(K)};
  shaper.AddShape(weight_name, weight_shape);

  const OperandType weight_operand_type(Type::TENSOR_FLOAT32, weight_shape);
  NEURON_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(
      weight_name, weight_fp32.data(), weight_operand_type));

  { std::vector<float>().swap(weight_fp32); }

  // -----------------------------------------------------------------------
  // 5. Get input shape and compute 2D batch dimension.
  // -----------------------------------------------------------------------
  const std::string& input_name  = node.input(0);
  const std::string& output_name = node.output(0);
  const Shape        input_shape = shaper[input_name];

  NEURON_RETURN_IF_NOT(!input_shape.empty(), "MatMulNBits: input shape is empty");
  NEURON_RETURN_IF_NOT(input_shape.back() == static_cast<uint32_t>(K),
                       "MatMulNBits: input last dim != K");

  uint32_t batch_seq = 1u;
  for (size_t i = 0; i + 1 < input_shape.size(); ++i)
    batch_seq *= input_shape[i];

  const uint32_t input_size  = static_cast<uint32_t>(K);
  const uint32_t output_size = static_cast<uint32_t>(N);
  const Shape    input_2d_shape  = {batch_seq, input_size};
  const Shape    output_2d_shape = {batch_seq, output_size};
  const bool     need_reshape    = (input_shape.size() != 2u);

  LOGS_NEURON(VERBOSE) << "MatMulNBits K=" << K << " N=" << N
                        << " batch_seq=" << batch_seq
                        << " need_reshape=" << need_reshape;

  // -----------------------------------------------------------------------
  // 6. Reshape FP32 input to 2D [batch_seq, K] if not already 2D.
  // -----------------------------------------------------------------------
  std::string flat_input_name;
  if (need_reshape) {
    flat_input_name = model_builder.GetUniqueName(node.name() + "_matmulnbits_in_2d");
    NEURON_RETURN_IF_ERROR(model_builder.AddReshape(input_name, input_2d_shape, flat_input_name));
  } else {
    flat_input_name = input_name;
  }

  // -----------------------------------------------------------------------
  // 7. FP32 bias operand [N].
  // -----------------------------------------------------------------------
  const std::string bias_name =
      model_builder.GetUniqueName(node.name() + "_matmulnbits_bias_fp32");
  const Shape bias_shape = {output_size};
  shaper.AddShape(bias_name, bias_shape);

  const bool has_bias = node.input_size() > 5 && !node.input(5).empty();
  {
    std::vector<float> bias_data(static_cast<size_t>(N), 0.0f);
    if (has_bias) {
      const auto* bias_proto = ctx.GetInitializer(node.input(5));
      if (bias_proto) {
        NEURON_RETURN_IF_ERROR(NeuronGraphContext::GetFloatData(*bias_proto, bias_data));
      }
    }
    const OperandType bias_operand_type(Type::TENSOR_FLOAT32, bias_shape);
    NEURON_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(
        bias_name, bias_data.data(), bias_operand_type));
  }

  // -----------------------------------------------------------------------
  // 8. Emit NEURON_FULLY_CONNECTED.
  // -----------------------------------------------------------------------
  const std::string fc_out_name = need_reshape
      ? model_builder.GetUniqueName(node.name() + "_matmulnbits_fc_2d")
      : output_name;

  if (need_reshape) {
    shaper.AddShape(fc_out_name, output_2d_shape);
  }

  const OperandType fc_out_type(Type::TENSOR_FLOAT32,
                                need_reshape ? output_2d_shape : shaper[output_name]);

  {
    std::vector<uint32_t> fc_indices;
    fc_indices.push_back(operand_indices.at(flat_input_name));
    fc_indices.push_back(operand_indices.at(weight_name));
    fc_indices.push_back(operand_indices.at(bias_name));

    const int32_t fuse_code = model_builder.FindActivation(node, ctx);
    ADD_SCALAR_OPERAND(model_builder, fc_indices, fuse_code);

    NEURON_RETURN_IF_ERROR(model_builder.AddOperation(
        NEURON_FULLY_CONNECTED, fc_indices, {fc_out_name}, {fc_out_type}));
  }

  // -----------------------------------------------------------------------
  // 9. Reshape FP32 output back to original rank.
  // -----------------------------------------------------------------------
  if (need_reshape) {
    Shape output_shape_orig = input_shape;
    output_shape_orig.back() = output_size;
    NEURON_RETURN_IF_ERROR(
        model_builder.AddReshape(fc_out_name, output_shape_orig, output_name));
  }

  return NeuronStatus::OK();
}

// ---------------------------------------------------------------------------
// Factory entry point
// ---------------------------------------------------------------------------

void CreateMatMulNBitsOpBuilder(const std::string& op_type,
                                OpBuilderRegistrations& op_registrations) {
  CreateSharedOpBuilderImpl<MatMulNBitsOpBuilder>(
      op_type, op_registrations, {"MatMulNBits"});
}

}  // namespace neuron
}  // namespace onnxruntime
