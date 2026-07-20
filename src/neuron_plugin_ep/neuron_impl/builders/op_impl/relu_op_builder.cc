// relu_op_builder.cc
// Plugin EP version – uses onnx::NodeProto + NeuronGraphContext.
//
// CHANGES FROM ORIGINAL:
//   OLD: AddToModelBuilderImpl(ModelBuilder&, const NodeUnit& node_unit)
//        - node_unit.Inputs()[0].node_arg.Name()  → node.input(0)
//        - node_unit.Outputs()[0].node_arg.Name() → node.output(0)
//        - Contains(model_builder.GetFusedActivations(), input) → same (no change)
//   NEW: AddToModelBuilderImpl(ModelBuilder&, const onnx::NodeProto& node, const NeuronGraphContext& ctx)

#include "base_op_builder.h"

namespace onnxruntime {
namespace neuron {
using namespace onnxruntime::neuron::op_builder_helpers;
using namespace onnxruntime::neuron::wrapper;

class ReluOpBuilder : public BaseOpBuilder {
 private:
  NeuronStatus AddToModelBuilderImpl(ModelBuilder& model_builder,
                                      const onnx::NodeProto& node,
                                      const NeuronGraphContext& /*ctx*/) const override;

  bool HasSupportedInputOutputsImpl(const onnx::NodeProto& node,
                                     const NeuronGraphContext& ctx,
                                     const OpSupportCheckParams& /*params*/) const override {
    return IsTypeSupported(node, ctx, 0, /*is_input*/ true,
                           {onnx::TensorProto::FLOAT,
                            onnx::TensorProto::FLOAT16,
                            onnx::TensorProto::INT8});
  }
};

NeuronStatus ReluOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                    const onnx::NodeProto& node,
                                                    const NeuronGraphContext& /*ctx*/) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  // In plugin EP: node.input(0) / node.output(0) replace node_unit.Inputs/Outputs.
  const auto& input  = node.input(0);
  const auto& output = node.output(0);

  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);

  // Skip if this Relu is a fused activation for another op.
  if (Contains(model_builder.GetFusedActivations(), input)) {
    LOGS_NEURON(VERBOSE) << "Relu Node [" << node.name() << "] fused";
    model_builder.RegisterOperand(output, operand_indices.at(input), output_operand_type);
  } else {
    std::vector<uint32_t> input_indices;
    input_indices.push_back(operand_indices.at(input));
    NEURON_RETURN_IF_ERROR(model_builder.AddOperation(NEURON_RELU, input_indices,
                                                       {output}, {output_operand_type}));
  }
  return NeuronStatus::OK();
}

void CreateReluOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ReluOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace neuron
}  // namespace onnxruntime
