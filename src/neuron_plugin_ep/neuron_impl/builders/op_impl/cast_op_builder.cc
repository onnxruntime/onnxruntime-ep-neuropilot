// cast_op_builder.cc
// Plugin EP version – uses onnx::NodeProto + NeuronGraphContext.
// Supports Cast to FLOAT (TENSOR_FLOAT32) or INT32 (TENSOR_INT32).

#include "base_op_builder.h"

namespace onnxruntime {
namespace neuron {
using namespace onnxruntime::neuron::op_builder_helpers;
using namespace onnxruntime::neuron::wrapper;

class CastOpBuilder : public BaseOpBuilder {
 private:
  NeuronStatus AddToModelBuilderImpl(ModelBuilder& model_builder,
                                      const onnx::NodeProto& node,
                                      const NeuronGraphContext& ctx) const override;


  bool HasSupportedInputOutputsImpl(const onnx::NodeProto& node,
                                     const NeuronGraphContext& ctx,
                                     const OpSupportCheckParams& /*params*/) const override {
    return (IsTypeSupported(node, ctx, 0, /*is_input*/ true,
                            {onnx::TensorProto::FLOAT}) &&
            IsTypeSupported(node, ctx, 0, /*is_input*/ false,
                            {onnx::TensorProto::FLOAT})) ||
           (IsTypeSupported(node, ctx, 0, /*is_input*/ true,
                            {onnx::TensorProto::UINT32}) &&
            IsTypeSupported(node, ctx, 0, /*is_input*/ false,
                            {onnx::TensorProto::UINT32}));
  }

  // Cast opset 5- uses string attribute for to type, not supported
  int GetMinSupportedOpSet(const onnx::NodeProto& /*node*/) const override { return 6; }
};

NeuronStatus CastOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                   const onnx::NodeProto& node,
                                                   const NeuronGraphContext& /*ctx*/) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());

  const auto& input  = node.input(0);
  const auto& output = node.output(0);

  const auto to = static_cast<int32_t>(NeuronGraphContext::GetAttrInt(node, "to", 0));

  Type type;
  switch (to) {
    case onnx::TensorProto::FLOAT:
      type = Type::TENSOR_FLOAT32;
      break;
    case onnx::TensorProto::INT32:
      type = Type::TENSOR_INT32;
      break;
    default:
      return NeuronStatus{false, MakeNeuronString("CastOpBuilder: invalid cast to type: ", to)};
  }

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  const OperandType output_operand_type(type, shaper[output]);
  NEURON_RETURN_IF_ERROR(model_builder.AddOperation(NEURON_CAST, input_indices,
                                                     {output}, {output_operand_type}));
  return NeuronStatus::OK();
}
void CreateCastOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<CastOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace neuron
}  // namespace onnxruntime
