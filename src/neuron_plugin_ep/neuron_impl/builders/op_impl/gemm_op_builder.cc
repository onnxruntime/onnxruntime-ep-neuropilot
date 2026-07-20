// gemm_op_builder.cc
// Plugin EP version – uses onnx::NodeProto + NeuronGraphContext.
//
// Ported from in-tree gemm_op_builder.cc.
//
// Translation notes:
//   node_unit.Inputs()[i].node_arg.Name()       → node.input(i)
//   node_unit.Outputs()[i].node_arg.Name()       → node.output(i)
//   node_unit.OpType()                           → node.op_type()
//   node_unit.Name()                             → node.name()
//   helper.Get("transB", 0)                      → NeuronGraphContext::GetAttrInt(node, "transB", 0)
//   ORT_RETURN_IF_ERROR / ORT_RETURN_IF_NOT      → NEURON_RETURN_IF_ERROR / NEURON_RETURN_IF_NOT
//   ORT_MAKE_STATUS(ONNXRUNTIME, ...)            → NeuronStatus{false, MakeNeuronString(...)}
//   Status::OK()                                 → NeuronStatus::OK()
//
// Quantized path:
//   In the plugin EP, QDQ nodes arrive as individual QuantizeLinear /
//   DequantizeLinear ONNX nodes.  For QDQGemm, the A/B quant params are NOT
//   attached to the NodeUnit (there is no NodeUnit); scale and zero-point are
//   resolved via GetConvMatMulOpQuantizationScaleAndZeroPoint(ctx, node, ...).
//
//   AddInitializersToSkip for the quantized path:
//     QDQGemm inputs layout (per ORT convention):
//       0: A, 1: B, 2: C(bias, optional), 3: a_scale, 4: a_zp, 5: b_scale,
//       6: b_zp, 7: y_scale, 8: y_zp
//     We skip: a_scale(3), a_zp(4), b_scale(5), b_zp(6), y_scale(7), y_zp(8)
//     and optionally B(1) if not transposed, bias(2), bias_scale, bias_zp.

#include "base_op_builder.h"

#include <onnx/onnx_pb.h>

#include "../op_builder_helpers.h"

namespace onnxruntime {
namespace neuron {
using namespace onnxruntime::neuron::op_builder_helpers;
using namespace onnxruntime::neuron::wrapper;

namespace {
// Get the bias size (C) of Gemm op.
// NEURON_FULLY_CONNECTED only supports 1-D bias.
// Tests that C can be squeezed to 1-D and returns the size.
bool GetGemmBiasSize(const Shape& c_shape, const NeuronRuntimeVersion& version, uint32_t& size) {
  (void)version;
  size_t c_dim = c_shape.size();
  if (c_dim == 0) {
    LOGS_NEURON(VERBOSE) << "C of Gemm cannot be a scalar";
    return false;
  }
  if (c_dim != 1) {
    // If C is a (2+)d tensor, it must have the form {1, 1, ..., 1, n}
    for (size_t i = 0; i < c_dim - 1; ++i) {
      if (c_shape[i] != 1) {
        LOGS_NEURON(VERBOSE) << "C of Gemm must be a vector or a tensor with only "
                             << "last dimension != 1, c_shape: " << Shape2String(c_shape);
        return false;
      }
    }
  }
  size = c_shape[c_dim - 1];
  return true;
}
}  // namespace

class GemmOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder,
                              const onnx::NodeProto& node,
                              const NeuronGraphContext& ctx) const override;

 private:
  NeuronStatus AddToModelBuilderImpl(ModelBuilder& model_builder,
                                      const onnx::NodeProto& node,
                                      const NeuronGraphContext& ctx) const override;


  bool HasSupportedInputOutputsImpl(const onnx::NodeProto& node,
                                     const NeuronGraphContext& ctx,
                                     const OpSupportCheckParams& params) const override;

  int GetMinSupportedOpSet(const onnx::NodeProto& /*node*/) const override {
    // Gemm opset 6- has broadcast attributes we do not support.
    return 7;
  }

  bool IsNodeUnitTypeSupported(const onnx::NodeProto& /*node*/) const override { return true; }

  bool IsQuantizedOp(const onnx::NodeProto& node,
                     const NeuronGraphContext& ctx) const override {
    return IsQuantizedGemm(GetQuantizedOpType(node, ctx));
  }

};


// ---------------------------------------------------------------------------
// AddInitializersToSkip
// ---------------------------------------------------------------------------
void GemmOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder,
                                            const onnx::NodeProto& node,
                                            const NeuronGraphContext& ctx) const {
  const bool is_quant_gemm = IsQuantizedOp(node, ctx);
  const int64_t transB = NeuronGraphContext::GetAttrInt(node, "transB", 0);

  if (is_quant_gemm) {
    // QDQGemm layout (matching ORT QDQ fusion convention):
    //   input(0)=A, input(1)=B, input(2)=C(optional bias)
    //   input(3)=a_scale, input(4)=a_zp, input(5)=b_scale, input(6)=b_zp
    //   input(7)=y_scale, input(8)=y_zp
    if (node.input_size() > 3) AddQuantizationScaleAndZeroPointToSkip(model_builder, node.input(3),
        node.input_size() > 4 ? node.input(4) : std::string{});
    if (node.input_size() > 5) AddQuantizationScaleAndZeroPointToSkip(model_builder, node.input(5),
        node.input_size() > 6 ? node.input(6) : std::string{});
    if (node.input_size() > 7) AddQuantizationScaleAndZeroPointToSkip(model_builder, node.input(7),
        node.input_size() > 8 ? node.input(8) : std::string{});

    // For transB == 0, B will be transposed and added separately; skip it.
    // For QDQ Gemm, node.input(1) is the DQ output — skip the underlying quantized weight.
    if (transB == 0 && node.input_size() > 1) {
      const std::string& b_input = node.input(1);
      const auto* b_dq = ctx.GetProducer(b_input);
      if (b_dq && b_dq->op_type() == "DequantizeLinear" && b_dq->input_size() > 0 &&
          ctx.GetInitializer(b_dq->input(0))) {
        model_builder.AddInitializerToSkip(b_dq->input(0));
      } else {
        model_builder.AddInitializerToSkip(b_input);
      }
    }

    // Skip bias (C) and its quant params if present.
    if (node.input_size() > 2 && !node.input(2).empty()) {
      // For QDQGemm, bias is int32 – skip it (we'll add it inline).
      model_builder.AddInitializerToSkip(node.input(2));
    }
  } else {
    // For non-quantized Gemm: if B is not transposed it must be an initializer
    // that we transpose ourselves.
    if (transB == 0 && node.input_size() > 1)
      model_builder.AddInitializerToSkip(node.input(1));
  }
}

// ---------------------------------------------------------------------------
// AddToModelBuilderImpl
// ---------------------------------------------------------------------------
NeuronStatus GemmOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                    const onnx::NodeProto& node,
                                                    const NeuronGraphContext& ctx) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  const bool is_quant_gemm = IsQuantizedOp(node, ctx);
  const int64_t transB = NeuronGraphContext::GetAttrInt(node, "transB", 0);

  const auto& input1 = node.input(0);   // A
  const auto& input2 = node.input(1);   // B
  const auto& output = node.output(0);  // Y

  float a_scale = 0.0f, b_scale = 0.0f, y_scale = 0.0f;
  int32_t a_zero_point = 0, b_zero_point = 0, y_zero_point = 0;

  // For plugin-EP QDQ, scales live in the DQ/Q sibling nodes captured in
  // ctx.qdq_groups — NOT in the Gemm node's own inputs (which is the
  // QLinearConv convention and does not apply here).
  if (is_quant_gemm) {
    const auto* qdq = ctx.GetQDQGroup(output);
    NEURON_RETURN_IF_NOT(qdq != nullptr,
                         "QDQGemm: no QDQ group found for output '", output, "'");
    if (qdq->input_quant_params.size() > 0 && qdq->input_quant_params[0].has_value()) {
      a_scale      = qdq->input_quant_params[0]->scale;
      a_zero_point = qdq->input_quant_params[0]->zero_point;
    }
    if (qdq->input_quant_params.size() > 1 && qdq->input_quant_params[1].has_value()) {
      b_scale      = qdq->input_quant_params[1]->scale;
      b_zero_point = qdq->input_quant_params[1]->zero_point;
    }
    if (qdq->output_quant_params.has_value()) {
      y_scale      = qdq->output_quant_params->scale;
      y_zero_point = qdq->output_quant_params->zero_point;
    }
  }

  // For QDQ Gemm, input2 (node.input(1)) is the DQ output — resolve to the actual quantized weight.
  std::string actual_b_name = input2;
  {
    const auto* b_dq = ctx.GetProducer(input2);
    if (b_dq && b_dq->op_type() == "DequantizeLinear" && b_dq->input_size() > 0 &&
        ctx.GetInitializer(b_dq->input(0))) {
      actual_b_name = b_dq->input(0);
    }
  }

  uint32_t input_2_idx;
  if (transB == 0) {
    // Determine the Neuron type for B from the actual QDQ quant_type, not a
    // hardcoded QUANT8_ASYMM, so signed/unsigned and per-channel are handled.
    Type onnx_mat_b_type = Type::TENSOR_FLOAT32;
    if (is_quant_gemm) {
      const auto* qdq = ctx.GetQDQGroup(output);
      if (qdq && qdq->input_quant_params.size() > 1 && qdq->input_quant_params[1].has_value()) {
        const auto& bqp = *qdq->input_quant_params[1];
        onnx_mat_b_type = bqp.IsSigned() ? Type::TENSOR_QUANT8_ASYMM_SIGNED
                                          : Type::TENSOR_QUANT8_ASYMM;
      } else {
        onnx_mat_b_type = Type::TENSOR_QUANT8_ASYMM;
      }
    }

    const auto* mat_b_tensor_ptr = ctx.GetInitializer(actual_b_name);
    NEURON_RETURN_IF_NOT(mat_b_tensor_ptr != nullptr,
                          "Gemm: B (", input2, ") must be a constant initializer when transB=0");
    const auto& mat_b_tensor = *mat_b_tensor_ptr;

    Shape onnx_mat_b_shape;
    for (auto dim : mat_b_tensor.dims())
      onnx_mat_b_shape.push_back(static_cast<uint32_t>(dim));

    const OperandType onnx_mat_b_operand_type(onnx_mat_b_type, onnx_mat_b_shape, b_scale, b_zero_point);
    NEURON_RETURN_IF_ERROR(AddInitializerTransposed(model_builder, onnx_mat_b_operand_type, actual_b_name, /*is_per_tensor_u8s8=*/false));
  }

  input_2_idx = operand_indices.at(actual_b_name);

  if (is_quant_gemm) {
    NEURON_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input1, a_scale, a_zero_point));
    NEURON_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, actual_b_name, b_scale, b_zero_point));
  }

  uint32_t bias_idx;
  const bool has_bias = node.input_size() > 2 && !node.input(2).empty();

  if (has_bias) {
    const auto& bias = node.input(2);
    if (!is_quant_gemm) {
      // Squeeze bias to 1-D if necessary.
      if (shaper[bias].size() > 1) {
        std::string bias_squeezed = model_builder.GetUniqueName(node.name() + node.op_type() + "_bias_squeezed");
        NEURON_RETURN_IF_ERROR(AddSqueezeOp(model_builder, node.name(),
                                             bias, bias_squeezed, {} /* axes */));
        bias_idx = operand_indices.at(bias_squeezed);
        LOGS_NEURON(VERBOSE) << "GemmOpBuilder - Operand [" << bias << "] squeezed from "
                              << Shape2String(shaper[bias])
                              << " to "
                              << Shape2String(shaper[bias_squeezed]);
      } else {
        bias_idx = operand_indices.at(bias);
      }
    } else {
      // QDQGemm: bias must be int32.
      const auto* bias_tensor_ptr = ctx.GetInitializer(bias);
      NEURON_RETURN_IF_NOT(bias_tensor_ptr != nullptr,
                            "QDQGemm: bias (", bias, ") must be a constant initializer");
      NEURON_RETURN_IF_NOT(bias_tensor_ptr->data_type() == onnx::TensorProto::INT32,
                            "bias of QDQGemm should be int32, actual type: ",
                            bias_tensor_ptr->data_type());

      Shape bias_dimen;
      for (auto dim : bias_tensor_ptr->dims())
        bias_dimen.push_back(static_cast<uint32_t>(dim));

      const void* bias_data = op_builder_helpers::GetRawInitializerData(*bias_tensor_ptr);
      NEURON_RETURN_IF_NOT(bias_data != nullptr, "QDQGemm: bias has no data");

      OperandType bias_operand_type(Type::TENSOR_INT32, bias_dimen, a_scale * b_scale);
      NEURON_RETURN_IF_ERROR(
          model_builder.AddOperandFromPersistMemoryBuffer(bias, bias_data, bias_operand_type));
      bias_idx = operand_indices.at(bias);
    }
  } else {
    LOGS_NEURON(VERBOSE) << "Use FullyConnected with zero bias for node: " << node.name();
    std::string bias = model_builder.GetUniqueName(node.name() + node.op_type() + "_bias");
    const auto& bias_type = operand_types.at(actual_b_name).type;
    const Shape& bias_dimen = {shaper[actual_b_name][0]};
    if (bias_type == Type::TENSOR_FLOAT32) {
      std::vector<float> buffer(bias_dimen[0], 0.f);
      OperandType bias_operand_type(Type::TENSOR_FLOAT32, bias_dimen);
      NEURON_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(bias, buffer.data(), bias_operand_type));
    } else if (bias_type == Type::TENSOR_QUANT8_ASYMM) {
      std::vector<int32_t> buffer(bias_dimen[0], 0);
      OperandType bias_operand_type(Type::TENSOR_INT32, bias_dimen, a_scale * b_scale, 0);
      NEURON_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(bias, buffer.data(), bias_operand_type));
    } else {
      return NeuronStatus{false, MakeNeuronString("GemmOpBuilder: unknown weight type ", TypeToStr(bias_type))};
    }
    bias_idx = operand_indices.at(bias);
  }

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input1));  // A
  input_indices.push_back(input_2_idx);                 // B
  input_indices.push_back(bias_idx);                    // C
  int32_t fuse_code = model_builder.FindActivation(node, ctx);
  ADD_SCALAR_OPERAND(model_builder, input_indices, fuse_code);

  // For QDQ, derive output type from the Q node's quant params, not the input type.
  Type output_type = operand_types.at(input1).type;
  if (is_quant_gemm) {
    const auto* qdq = ctx.GetQDQGroup(output);
    if (qdq && qdq->output_quant_params.has_value()) {
      output_type = qdq->output_quant_params->IsSigned()
                        ? Type::TENSOR_QUANT8_ASYMM_SIGNED
                        : Type::TENSOR_QUANT8_ASYMM;
    }
  }
  const OperandType output_operand_type(output_type, shaper[output], y_scale, y_zero_point);
  NEURON_RETURN_IF_ERROR(model_builder.AddOperation(NEURON_FULLY_CONNECTED, input_indices,
                                                     {output}, {output_operand_type}));
  return NeuronStatus::OK();
}

// ---------------------------------------------------------------------------
// HasSupportedInputOutputsImpl
// ---------------------------------------------------------------------------
bool GemmOpBuilder::HasSupportedInputOutputsImpl(const onnx::NodeProto& node,
                                                   const NeuronGraphContext& ctx,
                                                   const OpSupportCheckParams& params) const {
  if (!IsQuantizedOp(node, ctx)) {
    return BaseOpBuilder::HasSupportedInputOutputsImpl(node, ctx, params);
  }

  // QDQGemm: check quantized IO types.
  if (!IsQuantizedIOSupported(ctx, node, {0, 1}, params, /*is_input*/ true))
    return false;
  if (!IsQuantizedIOSupported(ctx, node, {0}, params, /*is_input*/ false))
    return false;

  return true;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
void CreateGemmOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  CreateSharedOpBuilderImpl<GemmOpBuilder>(
      op_type, op_registrations,
      {
          "Gemm",
      });
}

}  // namespace neuron
}  // namespace onnxruntime
