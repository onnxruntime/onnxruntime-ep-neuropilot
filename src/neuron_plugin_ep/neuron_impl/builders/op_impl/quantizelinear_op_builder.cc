// quantizelinear_op_builder.cc
// Plugin EP version – uses onnx::NodeProto + NeuronGraphContext.
//
// Ported from in-tree quantizelinear_op_builder.cc.
//
// ONNX QuantizeLinear (opset 10/13/19):
//   Inputs:
//     [0] x       – tensor to quantize (float32 / float16)
//     [1] y_scale – scalar or 1-D float initializer
//     [2] y_zero_point – scalar initializer (optional; defaults to 0)
//   Output:
//     [0] y       – quantized tensor (uint8 / int8)
//
// Translation notes:
//   In the in-tree EP, scale/ZP came from node_unit.Outputs()[0].quant_param.
//   In the plugin EP those quant params are regular node inputs at indices 1 and 2.
//   We read them from the constant initializers and pass them to
//   ModelBuilder::AddQuantizeLinear via an OnnxDataTypeInfo struct.
//
//   AddInitializersToSkip: skip y_scale (input 1) and y_zero_point (input 2).

#include "base_op_builder.h"

#include <onnx/onnx_pb.h>

#include "../op_builder_helpers.h"

namespace onnxruntime {
namespace neuron {
using namespace onnxruntime::neuron::op_builder_helpers;
using namespace onnxruntime::neuron::wrapper;

class QuantizeLinearOpBuilder : public BaseOpBuilder {
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
};

// ---------------------------------------------------------------------------
// AddInitializersToSkip
// ---------------------------------------------------------------------------
void QuantizeLinearOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder,
                                                      const onnx::NodeProto& node,
                                                      const NeuronGraphContext& /*ctx*/) const {
  // Skip y_scale (input 1) and y_zero_point (input 2).
  if (node.input_size() > 1 && !node.input(1).empty())
    AddQuantizationScaleAndZeroPointToSkip(
        model_builder,
        node.input(1),
        node.input_size() > 2 ? node.input(2) : std::string{});
}

// ---------------------------------------------------------------------------
// HasSupportedInputsImpl
// ---------------------------------------------------------------------------
bool QuantizeLinearOpBuilder::HasSupportedInputsImpl(
    const onnx::NodeProto& node,
    const NeuronGraphContext& ctx,
    const OpSupportCheckParams& /*params*/) const {
  // x (input 0): float32 or float16
  if (!IsTypeSupported(node, ctx, 0, /*is_input*/ true,
                       {onnx::TensorProto::FLOAT, onnx::TensorProto::FLOAT16})) {
    return false;
  }
  // y (output 0): int8 or uint8
  if (!IsTypeSupported(node, ctx, 0, /*is_input*/ false,
                       {onnx::TensorProto::INT8, onnx::TensorProto::UINT8})) {
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// AddToModelBuilderImpl
// ---------------------------------------------------------------------------
NeuronStatus QuantizeLinearOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                              const onnx::NodeProto& node,
                                                              const NeuronGraphContext& ctx) const {
  NEURON_RETURN_IF_NOT(node.input_size() >= 2,
                        "QuantizeLinear expects at least 2 inputs (x, y_scale).");

  const auto& input  = node.input(0);   // x
  const auto& output = node.output(0);  // y

  // Read y_scale from constant initializer (input 1).
  const std::string& scale_name = node.input(1);
  float y_scale = 0.0f;
  NEURON_RETURN_IF_ERROR(ctx.GetScalarFloat(scale_name, y_scale));

  // Read y_zero_point from constant initializer (input 2), default 0.
  int32_t y_zero_point = 0;
  if (node.input_size() > 2 && !node.input(2).empty()) {
    NEURON_RETURN_IF_ERROR(ctx.GetScalarInt32(node.input(2), y_zero_point));
  }

  // Determine the output ONNX element type from the output value info.
  int32_t onnx_data_type = ctx.GetElementType(output);
  if (onnx_data_type == 0) {
    // Fall back to uint8 if type is unresolved.
    onnx_data_type = onnx::TensorProto::UINT8;
  }

  // Build OnnxDataTypeInfo for the quantized output.
  wrapper::OnnxDataTypeInfo output_type_info;
  output_type_info.onnx_data_type = onnx_data_type;
  output_type_info.scales         = {y_scale};
  output_type_info.zero_points    = {y_zero_point};

  NEURON_RETURN_IF_ERROR(model_builder.AddQuantizeLinear(input, output_type_info, output));
  return NeuronStatus::OK();
}

void CreateQuantizeLinearOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<QuantizeLinearOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace neuron
}  // namespace onnxruntime
