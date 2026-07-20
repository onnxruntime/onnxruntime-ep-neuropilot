// elu_op_builder.cc
// Plugin EP version – uses onnx::NodeProto + NeuronGraphContext.

#include "base_op_builder.h"

namespace onnxruntime {
namespace neuron {
using namespace onnxruntime::neuron::op_builder_helpers;
using namespace onnxruntime::neuron::wrapper;

class EluOpBuilder : public BaseOpBuilder {
 private:
  NeuronStatus AddToModelBuilderImpl(ModelBuilder& model_builder,
                                      const onnx::NodeProto& node,
                                      const NeuronGraphContext& ctx) const override;

  // Elu opset 5- uses consumed_inputs attribute which is not supported
  int GetMinSupportedOpSet(const onnx::NodeProto& /*node*/) const override { return 6; }
};

NeuronStatus EluOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                   const onnx::NodeProto& node,
                                                   const NeuronGraphContext& /*ctx*/) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  const auto& input  = node.input(0);
  const auto& output = node.output(0);

  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);
  const auto alpha = NeuronGraphContext::GetAttrFloat(node, "alpha", 1.0f);

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  ADD_SCALAR_OPERAND(model_builder, input_indices, alpha);

  NEURON_RETURN_IF_ERROR(model_builder.AddOperation(NEURON_ELU, input_indices,
                                                     {output}, {output_operand_type}));
  return NeuronStatus::OK();
}

void CreateEluOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<EluOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace neuron
}  // namespace onnxruntime
