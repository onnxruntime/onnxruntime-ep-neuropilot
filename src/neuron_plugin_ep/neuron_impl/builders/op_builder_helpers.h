#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <optional>
#include <vector>
#include <gsl/span>
#include <onnx/onnx_pb.h>
#include "../neuron_types_internal.h"
#include "../neuron_graph_context.h"
#include "../neuron_lib/neuron_wrapper_operand.h"
#include "neuron_model_builder.h"
#include "helper.h"

// ---------------------------------------------------------------------------
// AutoPadType — mirrors the ONNX auto_pad attribute values
// ---------------------------------------------------------------------------
enum class AutoPadType {
  NOTSET = 0,
  VALID,
  SAME_UPPER,
  SAME_LOWER,
};

inline AutoPadType StringToAutoPadType(const std::string& s) {
  if (s == "VALID")      return AutoPadType::VALID;
  if (s == "SAME_UPPER") return AutoPadType::SAME_UPPER;
  if (s == "SAME_LOWER") return AutoPadType::SAME_LOWER;
  return AutoPadType::NOTSET;
}

namespace onnxruntime {
namespace neuron {
namespace op_builder_helpers {

using onnxruntime::neuron::wrapper::OperandType, onnxruntime::neuron::wrapper::Type;

// Scalar operand helper
template <typename T>
NeuronStatus AddScalarOperand(ModelBuilder& model_builder,
                               std::vector<uint32_t>& input_indices,
                               T scalar_value);

#define ADD_SCALAR_OPERAND(model_builder, input_indices, scalar_value) \
  NEURON_RETURN_IF_ERROR(onnxruntime::neuron::op_builder_helpers::AddScalarOperand(model_builder, input_indices, scalar_value))

// Primitive Neuron op helpers
NeuronStatus AddNeuronTranspose(ModelBuilder& model_builder, const std::string& data_input, const std::string& perm_input, gsl::span<const int32_t> perm, const std::string& output);
NeuronStatus AddNeuronReshape(ModelBuilder& model_builder, const std::string& data_input, const std::string& shape_input, const std::vector<int32_t>& shape_value, const std::string& output);
NeuronStatus AddNeuronSplit(ModelBuilder& model_builder, const std::string& input, int32_t axis, const std::vector<std::string>& outputs);
NeuronStatus AddNeuronBatchNormalization(ModelBuilder& model_builder, const std::string& input1, const std::string& input2, const std::string& input3, const std::string& output1, const std::string& output2, int32_t fuse_code, float output_scale = 0.0f, int32_t output_zero_point = 0);
NeuronStatus AddBinaryOperator(int32_t op_type, ModelBuilder& model_builder, const std::string& input1, const std::string& input2, bool add_activation, int32_t fuse_code, const std::string& output, float output_scale = 0.0f, int32_t output_zero_point = 0);
NeuronStatus AddIdentityOp(ModelBuilder& model_builder, const onnx::NodeProto& node, const NeuronGraphContext& ctx);
NeuronStatus AddSqueezeOp(ModelBuilder& model_builder, const std::string& node_name, const std::string& input, const std::string& output, std::vector<int32_t> axes);
NeuronStatus AddMinMaxOperator(ModelBuilder& model_builder, const onnx::NodeProto& node, const NeuronGraphContext& ctx, const std::string& input1, const std::string& input2);
NeuronStatus AddReshapeOperator(ModelBuilder& model_builder, const onnx::NodeProto& node, const NeuronGraphContext& ctx, const std::string& input, const std::vector<int32_t>& shape);
NeuronStatus GetAxesForSqueezeAndUnSqueeze(ModelBuilder& model_builder, const onnx::NodeProto& node, const NeuronGraphContext& ctx, std::vector<int32_t>& axes);

// Layout check
inline bool IsNodeLayoutNHWC(const onnx::NodeProto& node) {
  for (const auto& attr : node.attribute()) {
    if (attr.name() == "domain" && attr.s().find("NHWCDomain") != std::string::npos) return true;
  }
  return false;
}

// Quantization helpers
// GetQuantizationScaleAndZeroPoint is declared in helper.h (onnxruntime::neuron ns)
NeuronStatus GetBinaryOpQuantizationScaleAndZeroPoint(const NeuronGraphContext& ctx, const onnx::NodeProto& node, float& a_scale, float& b_scale, float& y_scale, int32_t& a_zero_point, int32_t& b_zero_point, int32_t& y_zero_point);
NeuronStatus GetConvMatMulOpQuantizationScaleAndZeroPoint(const ModelBuilder& model_builder, const NeuronGraphContext& ctx, const onnx::NodeProto& node, float& a_scale, float& w_scale, float& y_scale, int32_t& a_zero_point, int32_t& w_zero_point, int32_t& y_zero_point, std::optional<std::vector<float>>& w_scales, bool& is_per_tensor_u8s8);
NeuronStatus IsValidInputQuantizedType(const ModelBuilder& model_builder, const std::string& input_name, float scale, int32_t zero_point);
NeuronStatus IsValidConvWeightQuantizedType(const ModelBuilder& model_builder, const std::string& input_name, float scale, int32_t zero_point, const std::optional<std::vector<float>>& scales);
NeuronStatus IsOpInRequiredLayout(bool use_nchw, const onnx::NodeProto& node, const NeuronGraphContext& ctx);
void AddQuantizationScaleAndZeroPointToSkip(ModelBuilder& model_builder, const std::string& scale_name, const std::string& zp_name);
void AddInputToSkip(ModelBuilder& model_builder, const std::string& input_name, const std::string& scale_name, const std::string& zp_name);
bool IsQuantizedIOSupported(const NeuronGraphContext& ctx, const onnx::NodeProto& node,
                             const std::vector<size_t>& indices,
                             const OpSupportCheckParams& params, bool is_input);
bool HasRequiredScaleAndZeroPoint(const NeuronGraphContext& ctx, const std::string& op_desc,
                                   const std::string& scale_name, const std::string& zp_name,
                                   float required_scale, int32_t required_zp);

// Layout/initializer helpers
bool IsSupportedBatchMatMul(const onnx::NodeProto& node, const NeuronGraphContext& ctx, const NeuronRuntimeVersion& version);
NeuronStatus BuildBatchMatMul(ModelBuilder& model_builder, const onnx::NodeProto& node, const NeuronGraphContext& ctx);

// Data manipulation
NeuronStatus AddInitializerInNewLayout(ModelBuilder& model_builder, const std::string& name, const OperandType& source_operand_type, ModelBuilder::DataLayout new_layout, bool is_per_tensor_u8s8);
NeuronStatus AddInitializerTransposed(ModelBuilder& model_builder, const OperandType& source_operand_type, const std::string& name, bool is_per_tensor_u8s8);

// Pad computation (signatures match op_builder_helpers.cc implementations)
NeuronStatus ComputePad(int32_t input_size, int32_t stride, uint32_t weight_size, int32_t dilation,
                         AutoPadType auto_pad, int64_t& pad_head, int64_t& pad_tail);
NeuronStatus ComputeConvPads(const Shape& input_dimen, uint32_t weight_size_y, uint32_t weight_size_x,
                              const std::vector<int32_t>& onnx_pads,
                              const std::vector<int32_t>& onnx_strides,
                              const std::vector<int32_t>& onnx_dilations,
                              AutoPadType auto_pad_type, bool nchw,
                              std::vector<int32_t>& pads_out);
NeuronStatus HandleAutoPad(const Shape& input_shape, uint32_t weight_size_y, uint32_t weight_size_x,
                            const std::vector<int32_t>& onnx_strides,
                            const std::vector<int32_t>& onnx_dilations,
                            AutoPadType auto_pad_type, bool use_nchw,
                            std::vector<int32_t>& onnx_pads,
                            int32_t& neuron_padding_code, bool& use_auto_pad);

// Quantized support checks
bool IsQuantizationScaleSupported(const NeuronGraphContext& ctx, const std::string& scale_name, const OpSupportCheckParams& params, const std::string& op_type, bool is_quant_matmul, bool is_conv_matmul_u8s8_weight);
bool IsQuantizationZeroPointSupported(const NeuronGraphContext& ctx, const std::string& zp_name, const std::string& op_type, bool is_quant_matmul, bool is_conv_matmul_u8s8_weight);
bool IsQuantizedIOSupported(const NeuronGraphContext& ctx, const onnx::NodeProto& node, const std::vector<size_t>& indices, const OpSupportCheckParams& params, bool is_input);
bool HasRequiredScaleAndZeroPoint(const NeuronGraphContext& ctx, const std::string& op_desc, const std::string& scale_name, const std::string& zp_name, float required_scale, int32_t required_zp);

NeuronStatus PerformBroadcasting(const Shape& shape1, const Shape& shape2, Shape& output_shape);

// Template implementations (in header since they must be instantiated here)
template <typename T>
NeuronStatus AddScalarOperand(ModelBuilder& model_builder,
                               std::vector<uint32_t>& input_indices,
                               T scalar_value) {
  uint32_t index = 0;
  NEURON_RETURN_IF_ERROR(model_builder.AddOperandFromScalar(std::move(scalar_value), index));
  input_indices.push_back(index);
  return NeuronStatus::OK();
}

// ---------------------------------------------------------------------------
// GetRawInitializerData — return pointer to raw byte data in a TensorProto
// ---------------------------------------------------------------------------
inline const void* GetRawInitializerData(const onnx::TensorProto& tensor) {
  if (!tensor.raw_data().empty())
    return tensor.raw_data().data();
  switch (tensor.data_type()) {
    case onnx::TensorProto::FLOAT:   return tensor.float_data().data();
    case onnx::TensorProto::INT32:
    case onnx::TensorProto::INT16:
    case onnx::TensorProto::INT8:
    case onnx::TensorProto::UINT8:
    case onnx::TensorProto::BOOL:    return tensor.int32_data().data();
    case onnx::TensorProto::INT64:   return tensor.int64_data().data();
    case onnx::TensorProto::DOUBLE:  return tensor.double_data().data();
    default: return nullptr;
  }
}

// ---------------------------------------------------------------------------
// TransposeToNewBuffer<T> — transpose tensor data according to `perm`
// ---------------------------------------------------------------------------
template <typename T>
std::vector<T> TransposeToNewBuffer(const void* raw_buffer,
                                     size_t /*buffer_byte_size*/,
                                     const Shape& dims,
                                     const Shape& perm,
                                     Shape& new_dims) {
  const size_t rank = dims.size();
  new_dims.resize(rank);
  for (size_t i = 0; i < rank; ++i)
    new_dims[i] = dims[perm[i]];

  size_t total = 1;
  for (auto d : dims) total *= d;

  // src strides (row-major)
  std::vector<size_t> src_strides(rank, 1);
  for (int i = (int)rank - 2; i >= 0; --i)
    src_strides[i] = src_strides[i + 1] * dims[i + 1];

  // dst strides
  std::vector<size_t> dst_strides(rank, 1);
  for (int i = (int)rank - 2; i >= 0; --i)
    dst_strides[i] = dst_strides[i + 1] * new_dims[i + 1];

  const T* src = reinterpret_cast<const T*>(raw_buffer);
  std::vector<T> result(total);

  std::vector<size_t> idx(rank, 0);
  for (size_t elem = 0; elem < total; ++elem) {
    size_t src_idx = 0, dst_idx = 0;
    for (size_t d = 0; d < rank; ++d) {
      src_idx += idx[d] * src_strides[d];
      dst_idx += idx[perm[d]] * dst_strides[d];
    }
    result[dst_idx] = src[src_idx];
    // increment multi-dim index
    for (int d = (int)rank - 1; d >= 0; --d) {
      if (++idx[d] < dims[d]) break;
      idx[d] = 0;
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// DequantizeInitializerToBuffer<T> — dequantize quantized tensor to float32
// ---------------------------------------------------------------------------
template <typename T>
NeuronStatus DequantizeInitializerToBuffer(
    const onnx::TensorProto& tensor,
    const std::vector<float>& scales,
    const std::vector<int32_t>& zero_points,
    std::optional<int64_t> /*axis*/,
    std::vector<float>& float_buffer) {
  const void* raw = GetRawInitializerData(tensor);
  if (!raw)
    return NeuronStatus{false, "DequantizeInitializerToBuffer: tensor has no data"};

  size_t total = 1;
  for (int d = 0; d < tensor.dims_size(); ++d)
    total *= static_cast<size_t>(tensor.dims(d));

  const T* src = reinterpret_cast<const T*>(raw);
  float_buffer.resize(total);
  // Per-tensor dequantize (per-channel axis not needed for current ops)
  float scale = scales.empty() ? 1.0f : scales[0];
  float zp    = zero_points.empty() ? 0.f : static_cast<float>(zero_points[0]);
  for (size_t i = 0; i < total; ++i)
    float_buffer[i] = (static_cast<float>(src[i]) - zp) * scale;
  return NeuronStatus::OK();
}

} // namespace op_builder_helpers
} // namespace neuron
} // namespace onnxruntime
