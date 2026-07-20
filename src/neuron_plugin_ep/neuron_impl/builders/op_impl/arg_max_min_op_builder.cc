// arg_max_min_op_builder.cc
// Plugin EP version – uses onnx::NodeProto + NeuronGraphContext.
//
// CHANGES FROM ORIGINAL:
//   node_unit.Inputs()[i].node_arg.Name() → node.input(i)
//   node_unit.Outputs()[i].node_arg.Name() → node.output(i)
//   node_unit.OpType() → node.op_type()
//   node_unit.Name() → node.name()
//   helper.Get("axis", 0) → NeuronGraphContext::GetAttrInt(node, "axis", 0)
//   helper.Get("keepdims", 1) → NeuronGraphContext::GetAttrInt(node, "keepdims", 1)
//   GetType(node_arg, node_type) → ctx.GetElementType(name)
//   ORT_RETURN_IF_NOT → NEURON_RETURN_IF_NOT
//   ORT_RETURN_IF_ERROR → NEURON_RETURN_IF_ERROR
//   Status → NeuronStatus
//   CreateSharedOpBuilderImpl handles registration for both ArgMax and ArgMin.
// Handles: {"ArgMax", "ArgMin"}

#include "base_op_builder.h"
#include "../op_builder_helpers.h"

namespace onnxruntime {
namespace neuron {
using namespace onnxruntime::neuron::op_builder_helpers;
using namespace onnxruntime::neuron::wrapper;

class ArgMaxMinOpBuilder : public BaseOpBuilder {
 private:
  NeuronStatus AddToModelBuilderImpl(ModelBuilder& model_builder,
                                      const onnx::NodeProto& node,
                                      const NeuronGraphContext& ctx) const override;


  bool HasSupportedInputOutputsImpl(const onnx::NodeProto& node,
                                     const NeuronGraphContext& ctx,
                                     const OpSupportCheckParams& /*params*/) const override {
    // Input must be float or uint8; output must be int32
    // (ONNX spec requires int64 output but Neuron only supports int32, so only int32 is accepted).
    return IsTypeSupported(node, ctx, 0, /*is_input*/ true,
                           {onnx::TensorProto::FLOAT,
                            onnx::TensorProto::UINT8}) &&
           IsTypeSupported(node, ctx, 0, /*is_input*/ false,
                           {onnx::TensorProto::INT32});
  }
};

NeuronStatus ArgMaxMinOpBuilder::AddToModelBuilderImpl(
    ModelBuilder& model_builder,
    const onnx::NodeProto& node,
    const NeuronGraphContext& ctx) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  const auto& input  = node.input(0);
  const auto& output = node.output(0);

  const int32_t op_code = (node.op_type() == "ArgMax") ? NEURON_ARGMAX : NEURON_ARGMIN;
  Shape target_shape = shaper[output];

  // Determine output element type.
  int32_t node_type = ctx.GetElementType(output);
  Type output_type = Type::TENSOR_INT64;
  if (node_type == onnx::TensorProto::INT32) {
    output_type = Type::TENSOR_INT32;
  }
  OperandType output_operand_type(output_type, target_shape);

  const int32_t axis     = static_cast<int32_t>(NeuronGraphContext::GetAttrInt(node, "axis", 0));
  const int32_t keepdims = static_cast<int32_t>(NeuronGraphContext::GetAttrInt(node, "keepdims", 1));

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  ADD_SCALAR_OPERAND(model_builder, input_indices, axis);

  if (keepdims == 0 || operand_types.at(input).dimensions.size() == 1) {
    NEURON_RETURN_IF_ERROR(model_builder.AddOperation(op_code, input_indices,
                                                       {output}, {output_operand_type}));
  } else {
    // keepdims == 1 and rank > 1: Neuron ArgMax/ArgMin reduces the axis dimension
    // (no keepdims support), so emit the op then reshape to restore the 1-dim axis.
    auto middle_output_name = model_builder.GetUniqueName(output + "_argmax");
    int64_t positive_axis = HandleNegativeAxis(axis, static_cast<int64_t>(target_shape.size()));
    auto reduced_shape = target_shape;
    reduced_shape.erase(reduced_shape.begin() + positive_axis);
    shaper.AddShape(middle_output_name, reduced_shape);

    auto reduced_operand_type = output_operand_type;
    reduced_operand_type.SetDimensions(reduced_shape);
    NEURON_RETURN_IF_ERROR(model_builder.AddOperation(op_code, input_indices,
                                                       {middle_output_name}, {reduced_operand_type}));

    std::vector<int32_t> new_shape_value;
    new_shape_value.reserve(output_operand_type.dimensions.size());
    for (uint32_t d : output_operand_type.dimensions)
      new_shape_value.push_back(static_cast<int32_t>(d));
    NEURON_RETURN_IF_ERROR(AddReshapeOperator(model_builder, node, ctx, middle_output_name, new_shape_value));
  }

  return NeuronStatus::OK();
}
void CreateArgMaxOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  CreateSharedOpBuilderImpl<ArgMaxMinOpBuilder>(
      op_type, op_registrations,
      {
          "ArgMin",
          "ArgMax",
      });
}

}  // namespace neuron
}  // namespace onnxruntime
