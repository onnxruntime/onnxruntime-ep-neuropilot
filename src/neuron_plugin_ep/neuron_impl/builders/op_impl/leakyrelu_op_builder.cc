// leakyrelu_op_builder.cc
// Plugin EP version – uses onnx::NodeProto + NeuronGraphContext.
// Implements LeakyRelu via NEURON_PRELU with a broadcast alpha vector.

#include "base_op_builder.h"

namespace onnxruntime {
namespace neuron {
using namespace onnxruntime::neuron::op_builder_helpers;
using namespace onnxruntime::neuron::wrapper;

class LeakyReluOpBuilder : public BaseOpBuilder {
 private:
  NeuronStatus AddToModelBuilderImpl(ModelBuilder& model_builder,
                                      const onnx::NodeProto& node,
                                      const NeuronGraphContext& ctx) const override;


  bool HasSupportedInputOutputsImpl(const onnx::NodeProto& /*node*/,
                                     const NeuronGraphContext& /*ctx*/,
                                     const OpSupportCheckParams& /*params*/) const override {
    return true;
  }

  // LeakyRelu opset 6- has unsupported attributes
  int GetMinSupportedOpSet(const onnx::NodeProto& /*node*/) const override { return 6; }
};

NeuronStatus LeakyReluOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                        const onnx::NodeProto& node,
                                                        const NeuronGraphContext& /*ctx*/) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto& operand_indices(model_builder.GetOperandIndices());

  const auto& input  = node.input(0);
  const auto& output = node.output(0);

  auto input_shape = shaper[input];

  const auto alpha = NeuronGraphContext::GetAttrFloat(node, "alpha", 0.01f);

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));

  std::vector<uint32_t> value_shape{1};
  const std::string alpha_vec_name = model_builder.GetUniqueName(node.name() + input + "_alpha");
  const OperandType alpha_operand_type(Type::TENSOR_FLOAT32, value_shape);
  NEURON_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(alpha_vec_name, &alpha,
                                                                          alpha_operand_type));
  input_indices.push_back(operand_indices.at(alpha_vec_name));

  const OperandType prelu_output_operand_type(operand_types.at(input).type, input_shape);
  NEURON_RETURN_IF_ERROR(model_builder.AddOperation(NEURON_PRELU, input_indices,
                                                     {output}, {prelu_output_operand_type}));
  return NeuronStatus::OK();
}
void CreateLeakyReluOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<LeakyReluOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace neuron
}  // namespace onnxruntime
