// LRN_op_builder.cc
// Plugin EP version – uses onnx::NodeProto + NeuronGraphContext.
// LRN: maps ONNX alpha/beta/bias/size to NEURON_LOCAL_RESPONSE_NORMALIZATION.

#include "base_op_builder.h"

namespace onnxruntime {
namespace neuron {
using namespace onnxruntime::neuron::op_builder_helpers;
using namespace onnxruntime::neuron::wrapper;

class LRNOpBuilder : public BaseOpBuilder {
 private:
  NeuronStatus AddToModelBuilderImpl(ModelBuilder& model_builder,
                                      const onnx::NodeProto& node,
                                      const NeuronGraphContext& ctx) const override;

};

NeuronStatus LRNOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                  const onnx::NodeProto& node,
                                                  const NeuronGraphContext& ctx) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  const auto& input  = node.input(0);
  const auto& output = node.output(0);

  auto use_nchw = model_builder.UseNCHW();
  NEURON_RETURN_IF_ERROR(IsOpInRequiredLayout(use_nchw, node, ctx));

  auto alpha = NeuronGraphContext::GetAttrFloat(node, "alpha", 0.0001f);
  const auto beta  = NeuronGraphContext::GetAttrFloat(node, "beta", 0.75f);
  const auto bias  = NeuronGraphContext::GetAttrFloat(node, "bias", 1.0f);
  const auto size  = static_cast<int32_t>(NeuronGraphContext::GetAttrInt(node, "size", 1));

  const auto radius = (size - 1) / 2;
  alpha /= static_cast<float>(size);  // NNAPI alpha = ONNX alpha / size

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  ADD_SCALAR_OPERAND(model_builder, input_indices, radius);
  ADD_SCALAR_OPERAND(model_builder, input_indices, bias);
  ADD_SCALAR_OPERAND(model_builder, input_indices, alpha);
  ADD_SCALAR_OPERAND(model_builder, input_indices, beta);

  // ONNX LRN is always on the C dimension: axis=1 (NCHW) or axis=3 (NHWC)
  int32_t axis = use_nchw ? 1 : 3;
  ADD_SCALAR_OPERAND(model_builder, input_indices, axis);

  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);
  NEURON_RETURN_IF_ERROR(model_builder.AddOperation(NEURON_LOCAL_RESPONSE_NORMALIZATION,
                                                     input_indices, {output}, {output_operand_type}));
  return NeuronStatus::OK();
}
void CreateLRNOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<LRNOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace neuron
}  // namespace onnxruntime
