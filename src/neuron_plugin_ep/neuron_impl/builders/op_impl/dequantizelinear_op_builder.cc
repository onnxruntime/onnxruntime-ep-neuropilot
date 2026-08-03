// dequantizelinear_op_builder.cc
// Plugin EP version – uses onnx::NodeProto + NeuronGraphContext.
//
// Ported from in-tree dequantizelinear_op_builder.cc.
//
// ONNX DequantizeLinear (opset 10/13/19):
//   Inputs:
//     [0] x       – tensor to dequantize (uint8 / int8 / int32)
//     [1] x_scale – scalar or 1-D float initializer
//     [2] x_zero_point – scalar initializer (optional; defaults to 0)
//   Output:
//     [0] y       – dequantized tensor (float32)
//
// Translation notes:
//   In the in-tree EP, scale/ZP came from node_unit.Inputs()[0].quant_param.
//   In the plugin EP those quant params are regular node inputs at indices 1 and 2.
//   We read them from the constant initializers and pass them to
//   ModelBuilder::AddDequantizeLinear via an OnnxDataTypeInfo struct.
//
//   AddInitializersToSkip: skip x_scale (input 1) and x_zero_point (input 2).

#include "base_op_builder.h"

#include <onnx/onnx_pb.h>

#include "../op_builder_helpers.h"

namespace onnxruntime {
namespace neuron {
using namespace onnxruntime::neuron::op_builder_helpers;
using namespace onnxruntime::neuron::wrapper;

class DequantizeLinearOpBuilder : public BaseOpBuilder {
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
void DequantizeLinearOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder,
                                                        const onnx::NodeProto& node,
                                                        const NeuronGraphContext& /*ctx*/) const {
  // Skip x_scale (input 1) and x_zero_point (input 2).
  if (node.input_size() > 1 && !node.input(1).empty()) {
    const std::string zp = (node.input_size() > 2) ? node.input(2) : std::string{};
    AddQuantizationScaleAndZeroPointToSkip(model_builder, node.input(1), zp);
  }
}

// ---------------------------------------------------------------------------
// HasSupportedInputsImpl
// ---------------------------------------------------------------------------
bool DequantizeLinearOpBuilder::HasSupportedInputsImpl(
    const onnx::NodeProto& node,
    const NeuronGraphContext& ctx,
    const OpSupportCheckParams& /*params*/) const {
  // x (input 0): int8, uint8, or int32
  if (!IsTypeSupported(node, ctx, 0, /*is_input*/ true,
                       {onnx::TensorProto::INT8,
                        onnx::TensorProto::UINT8,
                        onnx::TensorProto::INT32})) {
    return false;
  }
  // y (output 0): float32
  if (!IsTypeSupported(node, ctx, 0, /*is_input*/ false,
                       {onnx::TensorProto::FLOAT})) {
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// AddToModelBuilderImpl
// ---------------------------------------------------------------------------
NeuronStatus DequantizeLinearOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                                const onnx::NodeProto& node,
                                                                const NeuronGraphContext& ctx) const {
  NEURON_RETURN_IF_NOT(node.input_size() >= 2,
                        "DequantizeLinear expects at least 2 inputs (x, x_scale).");

  const auto& input  = node.input(0);   // x
  const auto& output = node.output(0);  // y

  // Read x_scale from constant initializer (input 1).
  const std::string& scale_name = node.input(1);
  float x_scale = 0.0f;
  NEURON_RETURN_IF_ERROR(ctx.GetScalarFloat(scale_name, x_scale));

  // Read x_zero_point from constant initializer (input 2), default 0.
  int32_t x_zero_point = 0;
  if (node.input_size() > 2 && !node.input(2).empty()) {
    NEURON_RETURN_IF_ERROR(ctx.GetScalarInt32(node.input(2), x_zero_point));
  }

  // Determine the input ONNX element type.
  int32_t onnx_data_type = ctx.GetElementType(input);
  if (onnx_data_type == 0) {
    // Fall back to uint8 if type is unresolved.
    onnx_data_type = onnx::TensorProto::UINT8;
  }

  // Build OnnxDataTypeInfo for the quantized input.
  wrapper::OnnxDataTypeInfo input_type_info;
  input_type_info.onnx_data_type = onnx_data_type;
  input_type_info.scales         = {x_scale};
  input_type_info.zero_points    = {x_zero_point};

  NEURON_RETURN_IF_ERROR(model_builder.AddDequantizeLinear(input, input_type_info, output));
  return NeuronStatus::OK();
}

void CreateDequantizeLinearOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<DequantizeLinearOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace neuron
}  // namespace onnxruntime
