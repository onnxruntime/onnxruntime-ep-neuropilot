// gelu_op_builder.cc
// Plugin EP version – uses onnx::NodeProto + NeuronGraphContext.

#include "base_op_builder.h"

namespace onnxruntime {
namespace neuron {
using namespace onnxruntime::neuron::op_builder_helpers;
using namespace onnxruntime::neuron::wrapper;

class GeluOpBuilder : public BaseOpBuilder {
 private:
  NeuronStatus AddToModelBuilderImpl(ModelBuilder& model_builder,
                                      const onnx::NodeProto& node,
                                      const NeuronGraphContext& ctx) const override;

  bool HasSupportedInputOutputsImpl(const onnx::NodeProto& node,
                                     const NeuronGraphContext& ctx,
                                     const OpSupportCheckParams& /*params*/) const override {
    return IsTypeSupported(node, ctx, 0, /*is_input*/ true,
                           {onnx::TensorProto::FLOAT,
                            onnx::TensorProto::FLOAT16,
                            onnx::TensorProto::UINT8,
                            onnx::TensorProto::INT8});
  }

};

NeuronStatus GeluOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                    const onnx::NodeProto& node,
                                                    const NeuronGraphContext& ctx) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());

  const auto& input  = node.input(0);
  const auto& output = node.output(0);

  // Build output type from the graph context element type.
  int32_t elem_type = ctx.GetElementType(output);
  if (elem_type == 0) elem_type = onnx::TensorProto::FLOAT;
  wrapper::OnnxDataTypeInfo output_io_info;
  output_io_info.onnx_data_type = elem_type;

  const OperandType output_operand_type(output_io_info, shaper[output]);

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  const int32_t approximate = 1;  // Using tanh approximation
  ADD_SCALAR_OPERAND(model_builder, input_indices, approximate);

  NEURON_RETURN_IF_ERROR(model_builder.AddOperation(NEURON_GELU_V2, input_indices,
                                                     {output}, {output_operand_type}));
  return NeuronStatus::OK();
}

void CreateGeluOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<GeluOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace neuron
}  // namespace onnxruntime
