// depthtospace_op_builder.cc
// Plugin EP version – uses onnx::NodeProto + NeuronGraphContext.

#include "base_op_builder.h"

namespace onnxruntime {
namespace neuron {
using namespace onnxruntime::neuron::op_builder_helpers;
using namespace onnxruntime::neuron::wrapper;

class DepthToSpaceOpBuilder : public BaseOpBuilder {
 private:
  NeuronStatus AddToModelBuilderImpl(ModelBuilder& model_builder,
                                      const onnx::NodeProto& node,
                                      const NeuronGraphContext& ctx) const override;

};

NeuronStatus DepthToSpaceOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                           const onnx::NodeProto& node,
                                                           const NeuronGraphContext& ctx) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  const auto& input  = node.input(0);
  const auto& output = node.output(0);

  bool use_nchw = model_builder.UseNCHW();
  NEURON_RETURN_IF_ERROR(IsOpInRequiredLayout(use_nchw, node, ctx));

  const int32_t blocksize = static_cast<int32_t>(NeuronGraphContext::GetAttrInt(node, "blocksize", 0));

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  ADD_SCALAR_OPERAND(model_builder, input_indices, blocksize);

  if (use_nchw && GreaterEqualVersion(model_builder.GetNeuronVersion(), {4, 1, 0})) {
    ADD_SCALAR_OPERAND(model_builder, input_indices, use_nchw);
  }

  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);
  NEURON_RETURN_IF_ERROR(model_builder.AddOperation(NEURON_DEPTH_TO_SPACE, input_indices,
                                                     {output}, {output_operand_type}));
  return NeuronStatus::OK();
}
void CreateDepthToSpaceOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<DepthToSpaceOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace neuron
}  // namespace onnxruntime
