// softmax_op_builder.cc
// Plugin EP version – uses onnx::NodeProto + NeuronGraphContext.
//
// CHANGES FROM ORIGINAL:
//   node_unit.Inputs()[i].node_arg.Name() → node.input(i)
//   node_unit.Outputs()[i].node_arg.Name() → node.output(i)
//   helper.Get("axis", ...) → NeuronGraphContext::GetAttrInt(node, "axis", ...)
//   node_unit.SinceVersion() → check for num_outputs / axis attribute presence
//   IsQuantizedOp(node_unit) → check op_type == "QLinearSoftmax" (plugin EP doesn't
//     support QDQ SoftMax; only non-quantized and QLinearSoftmax are handled)
//   ORT_RETURN_IF_ERROR → NEURON_RETURN_IF_ERROR
//   Status → NeuronStatus
//
// Note: In the plugin EP there is no SinceVersion() from NodeUnit.  We use opset 13+
// semantics (axis refers to the last dim) for all nodes.  The pre-13 reshape-to-2D
// codepath is preserved for nodes that set axis != last_dim but that is handled via
// the reshape helper if needed.

#include <algorithm>
#include <numeric>

#include "base_op_builder.h"

namespace onnxruntime {
namespace neuron {
using namespace onnxruntime::neuron::op_builder_helpers;
using namespace onnxruntime::neuron::wrapper;

class SoftMaxOpBuilder : public BaseOpBuilder {
 private:
  NeuronStatus AddToModelBuilderImpl(ModelBuilder& model_builder,
                                      const onnx::NodeProto& node,
                                      const NeuronGraphContext& ctx) const override;


};

NeuronStatus SoftMaxOpBuilder::AddToModelBuilderImpl(
    ModelBuilder& model_builder,
    const onnx::NodeProto& node,
    const NeuronGraphContext& ctx) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  const auto& input  = node.input(0);
  const auto& output = node.output(0);

  int32_t axis = static_cast<int32_t>(NeuronGraphContext::GetAttrInt(node, "axis", 1));

  auto input_shape = shaper[input];
  if (axis < 0) {
    axis = static_cast<int32_t>(HandleNegativeAxis(axis, static_cast<int64_t>(input_shape.size())));
  }

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  const float beta = 1.0f;
  ADD_SCALAR_OPERAND(model_builder, input_indices, beta);
  ADD_SCALAR_OPERAND(model_builder, input_indices, axis);

  // For a QDQ softmax (DQ→Softmax→Q) the output operand must carry the Q node's
  // quantization params.  Leaving scale=0 (the default) makes the MDLA compiler
  // reject the op with "Act1/Act2 shift exceeds range".  Mirrors the shared
  // provider Neuron EP, which reads y_scale/y_zp from the output NodeUnit.
  float y_scale = 0.0f;
  int32_t y_zero_point = 0;
  if (const auto* qdq = ctx.GetQDQGroup(output);
      qdq && qdq->output_quant_params.has_value()) {
    y_scale      = qdq->output_quant_params->scale;
    y_zero_point = qdq->output_quant_params->zero_point;
  }

  const OperandType output_operand_type(operand_types.at(input).type, shaper[output],
                                        y_scale, y_zero_point);
  NEURON_RETURN_IF_ERROR(model_builder.AddOperation(NEURON_SOFTMAX, input_indices,
                                                    {output}, {output_operand_type}));
  return NeuronStatus::OK();
}

void CreateSoftMaxOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<SoftMaxOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace neuron
}  // namespace onnxruntime
