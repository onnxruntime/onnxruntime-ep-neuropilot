// pool_op_builder.cc
// Plugin EP version – uses onnx::NodeProto + NeuronGraphContext.
//
// CHANGES FROM ORIGINAL:
//   node_unit.Inputs()[i].node_arg.Name() → node.input(i)
//   node_unit.Outputs()[i].node_arg.Name() → node.output(i)
//   node_unit.OpType() → node.op_type()
//   helper.Get("kernel_shape", ...) → NeuronGraphContext::GetAttrInts(node, "kernel_shape")
//   IsQuantizedOp(node_unit) → dropped (plugin EP does not use QLinearAveragePool quant params)
//   GetQuantizationScaleAndZeroPoint → removed (not needed in non-QDQ plugin EP)
//   ORT_RETURN_IF_ERROR → NEURON_RETURN_IF_ERROR
//   Status → NeuronStatus
//
// Shared by: GlobalAveragePool, GlobalMaxPool, AveragePool, MaxPool
// (QLinearAveragePool support is deferred - plugin EP handles it via preceding DQ nodes)

#include "base_op_builder.h"

namespace onnxruntime {
namespace neuron {
using namespace onnxruntime::neuron::op_builder_helpers;
using namespace onnxruntime::neuron::wrapper;

class PoolOpBuilder : public BaseOpBuilder {
 private:
  NeuronStatus AddToModelBuilderImpl(ModelBuilder& model_builder,
                                      const onnx::NodeProto& node,
                                      const NeuronGraphContext& ctx) const override;


  bool HasSupportedInputsImpl(const onnx::NodeProto& node,
                                     const NeuronGraphContext& ctx,
                                     const OpSupportCheckParams& params) const override;
};

NeuronStatus PoolOpBuilder::AddToModelBuilderImpl(
    ModelBuilder& model_builder,
    const onnx::NodeProto& node,
    const NeuronGraphContext& ctx) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  const auto& input  = node.input(0);
  const auto& output = node.output(0);
  const auto& op_type = node.op_type();

  bool use_nchw = model_builder.UseNCHW();

  int32_t op_code;
  bool is_average_pool = (op_type == "AveragePool");
  if (is_average_pool || op_type == "GlobalAveragePool")
    op_code = NEURON_AVERAGE_POOL_2D;
  else
    op_code = NEURON_MAX_POOL_2D;

  std::vector<int32_t> onnx_pads, onnx_strides, kernel_shape;
  bool use_auto_pad = false;
  int32_t nnapi_padding_code = NEURON_PADDING_VALID;
  const auto input_shape = shaper[input];

  if (is_average_pool || op_type == "MaxPool") {
    const auto auto_pad_type = StringToAutoPadType(
        NeuronGraphContext::GetAttrString(node, "auto_pad", "NOTSET"));

    auto kernel_raw = NeuronGraphContext::GetAttrInts(node, "kernel_shape");
    if (kernel_raw.size() >= 2) {
      kernel_shape = {static_cast<int32_t>(kernel_raw[0]), static_cast<int32_t>(kernel_raw[1])};
    } else {
      kernel_shape = {0, 0};
    }

    auto strides_raw = NeuronGraphContext::GetAttrInts(node, "strides");
    if (strides_raw.size() >= 2) {
      onnx_strides = {static_cast<int32_t>(strides_raw[0]), static_cast<int32_t>(strides_raw[1])};
    } else {
      onnx_strides = {1, 1};
    }

    auto pads_raw = NeuronGraphContext::GetAttrInts(node, "pads");
    if (pads_raw.size() >= 4) {
      onnx_pads = {static_cast<int32_t>(pads_raw[0]), static_cast<int32_t>(pads_raw[1]),
                   static_cast<int32_t>(pads_raw[2]), static_cast<int32_t>(pads_raw[3])};
    } else {
      onnx_pads = {0, 0, 0, 0};
    }

    const auto weight_size_y = static_cast<uint32_t>(kernel_shape[0]);
    const auto weight_size_x = static_cast<uint32_t>(kernel_shape[1]);
    const auto padding_top    = static_cast<uint32_t>(onnx_pads[0]);
    const auto padding_bottom = static_cast<uint32_t>(onnx_pads[2]);
    const auto padding_left   = static_cast<uint32_t>(onnx_pads[1]);
    const auto padding_right  = static_cast<uint32_t>(onnx_pads[3]);

    // When the kernel exactly covers the spatial dimension with no padding,
    // stride must be 1 to produce a single-element output.
    if (weight_size_y == input_shape[1] && (padding_top + padding_bottom) == 0)
      onnx_strides[0] = 1;
    if (weight_size_x == input_shape[2] && (padding_left + padding_right) == 0)
      onnx_strides[1] = 1;

    NEURON_RETURN_IF_ERROR(
        HandleAutoPad(input_shape, weight_size_y, weight_size_x,
                      onnx_strides, {1, 1} /* onnx_dilations */,
                      auto_pad_type, use_nchw,
                      onnx_pads, nnapi_padding_code, use_auto_pad));
  } else {
    // GlobalAveragePool / GlobalMaxPool
    use_auto_pad = true;
    nnapi_padding_code = NEURON_PADDING_VALID;
    onnx_strides = {1, 1};
    onnx_pads    = {0, 0, 0, 0};
    if (use_nchw) {
      kernel_shape = {static_cast<int32_t>(input_shape[2]),
                      static_cast<int32_t>(input_shape[3])};
    } else {
      kernel_shape = {static_cast<int32_t>(input_shape[1]),
                      static_cast<int32_t>(input_shape[2])};
    }
  }

  int32_t fuse_code = model_builder.FindActivation(node, ctx);

  // Output scale/zp: for non-quantized ops inherit from input
  const OperandType& input_operand_type = operand_types.at(input);
  float y_scale       = input_operand_type.operandType.scale;
  int32_t y_zero_point = input_operand_type.operandType.zeroPoint;

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));

  if (use_auto_pad) {
    ADD_SCALAR_OPERAND(model_builder, input_indices, nnapi_padding_code);
  } else {
    ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_pads[1]);
    ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_pads[3]);
    ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_pads[0]);
    ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_pads[2]);
  }

  ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_strides[1]);
  ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_strides[0]);
  ADD_SCALAR_OPERAND(model_builder, input_indices, kernel_shape[1]);
  ADD_SCALAR_OPERAND(model_builder, input_indices, kernel_shape[0]);
  ADD_SCALAR_OPERAND(model_builder, input_indices, fuse_code);
  ADD_SCALAR_OPERAND(model_builder, input_indices, use_nchw);

  const OperandType output_operand_type(operand_types.at(input).type, shaper[output],
                                        y_scale, y_zero_point);
  NEURON_RETURN_IF_ERROR(model_builder.AddOperation(op_code, input_indices,
                                                    {output}, {output_operand_type}));
  return NeuronStatus::OK();
}

// Operator support related
bool PoolOpBuilder::HasSupportedInputsImpl(
    const onnx::NodeProto& node,
    const NeuronGraphContext& ctx,
    const OpSupportCheckParams& params) const {
  const auto& op_type = node.op_type();
  bool is_max_pool = (op_type == "MaxPool");

  if (!is_max_pool) {
    return BaseOpBuilder::HasSupportedInputsImpl(node, ctx, params);
  }

  // MaxPool supports float and uint8 input
  return IsTypeSupported(node, ctx, 0, /*is_input*/ true,
                         {onnx::TensorProto::FLOAT,
                          onnx::TensorProto::UINT8});
}

void CreatePoolOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  CreateSharedOpBuilderImpl<PoolOpBuilder>(
      op_type, op_registrations,
      {
          "GlobalAveragePool",
          "GlobalMaxPool",
          "AveragePool",
          "MaxPool",
      });
}

}  // namespace neuron
}  // namespace onnxruntime
