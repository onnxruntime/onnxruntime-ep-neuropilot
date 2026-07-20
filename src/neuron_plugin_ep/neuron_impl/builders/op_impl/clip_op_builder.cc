// clip_op_builder.cc
// Plugin EP version – uses onnx::NodeProto + NeuronGraphContext.
// Supports Clip as Relu6 [0,6] or Relu1 [-1,1] via NEURON_RELU6 / NEURON_RELU1.

#include <limits>

#include "base_op_builder.h"

namespace onnxruntime {
namespace neuron {
using namespace onnxruntime::neuron::op_builder_helpers;
using namespace onnxruntime::neuron::wrapper;

// ---------------------------------------------------------------------------
// Local helper: read Clip min/max from opset-6 attributes or opset-11+
// scalar initializer inputs.
// ---------------------------------------------------------------------------
static bool GetClipMinMax(const NeuronGraphContext& ctx, const onnx::NodeProto& node,
                           float& min_val, float& max_val) {
  min_val = -std::numeric_limits<float>::max();
  max_val =  std::numeric_limits<float>::max();

  // Opset 11+: inputs[1] = min, inputs[2] = max (optional scalar initializers)
  if (node.input_size() >= 2 && !node.input(1).empty()) {
    const auto* t = ctx.GetInitializer(node.input(1));
    if (!t) return false;
    std::vector<float> v;
    if (!NeuronGraphContext::GetFloatData(*t, v).IsOK() || v.empty()) return false;
    min_val = v[0];
  } else {
    min_val = NeuronGraphContext::GetAttrFloat(node, "min", min_val);
  }

  if (node.input_size() >= 3 && !node.input(2).empty()) {
    const auto* t = ctx.GetInitializer(node.input(2));
    if (!t) return false;
    std::vector<float> v;
    if (!NeuronGraphContext::GetFloatData(*t, v).IsOK() || v.empty()) return false;
    max_val = v[0];
  } else {
    max_val = NeuronGraphContext::GetAttrFloat(node, "max", max_val);
  }
  return true;
}

class ClipOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder,
                              const onnx::NodeProto& node,
                              const NeuronGraphContext& ctx) const override;

 private:
  NeuronStatus AddToModelBuilderImpl(ModelBuilder& model_builder,
                                      const onnx::NodeProto& node,
                                      const NeuronGraphContext& ctx) const override;

};

void ClipOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder,
                                           const onnx::NodeProto& node,
                                           const NeuronGraphContext& /*ctx*/) const {
  if (node.input_size() > 1)
    model_builder.AddInitializerToSkip(node.input(1));  // min

  if (node.input_size() > 2)
    model_builder.AddInitializerToSkip(node.input(2));  // max
}

NeuronStatus ClipOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                   const onnx::NodeProto& node,
                                                   const NeuronGraphContext& ctx) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  const auto& input  = node.input(0);
  const auto& output = node.output(0);

  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);

  if (Contains(model_builder.GetFusedActivations(), input)) {
    LOGS_NEURON(VERBOSE) << "Clip Node [" << node.name() << "] fused";
    model_builder.RegisterOperand(output, operand_indices.at(input), output_operand_type);
    return NeuronStatus::OK();
  }

  float min_val, max_val;
  GetClipMinMax(ctx, node, min_val, max_val);

  int32_t op_code;
  if (min_val == 0.0f && max_val == 6.0f)
    op_code = NEURON_RELU6;
  else if (min_val == -1.0f && max_val == 1.0f)
    op_code = NEURON_RELU1;
  else
    return NeuronStatus{false, MakeNeuronString("ClipOpBuilder: unsupported range [",
                                                 min_val, ", ", max_val, "]. "
                                                 "op_capability library should have caught this.")};

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  NEURON_RETURN_IF_ERROR(model_builder.AddOperation(op_code, input_indices,
                                                     {output}, {output_operand_type}));
  return NeuronStatus::OK();
}
void CreateClipOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ClipOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace neuron
}  // namespace onnxruntime
