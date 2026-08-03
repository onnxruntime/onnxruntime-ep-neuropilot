// gather_op_builder.cc
// Plugin EP version – uses onnx::NodeProto + NeuronGraphContext.
//
// CHANGES FROM ORIGINAL:
//   node_unit.Inputs()[i].node_arg.Name() → node.input(i)
//   node_unit.Outputs()[i].node_arg.Name() → node.output(i)
//   model_builder.GetInitializerTensors() → ctx.GetInitializer(name)
//   Contains(initializers, name) → ctx.IsInitializer(name)
//   Initializer unpacked_tensor / DataAsSpan → NeuronGraphContext::GetInt32/Int64Data
//   GetOnnxDataTypeInfoFromNodeUnit → build OnnxDataTypeInfo from ctx.GetElementType(output)
//   GetType(node_arg, type) → ctx.GetElementType(name)
//   helper.Get("axis", 0) → NeuronGraphContext::GetAttrInt(node, "axis", 0)
//   ORT_RETURN_IF_ERROR → NEURON_RETURN_IF_ERROR
//   Status → NeuronStatus

#include "base_op_builder.h"
#include "../op_builder_helpers.h"

namespace onnxruntime {
namespace neuron {
using namespace onnxruntime::neuron::op_builder_helpers;
using namespace onnxruntime::neuron::wrapper;

class GatherOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder,
                              const onnx::NodeProto& node,
                              const NeuronGraphContext& ctx) const override;

 private:
  NeuronStatus AddToModelBuilderImpl(ModelBuilder& model_builder,
                                      const onnx::NodeProto& node,
                                      const NeuronGraphContext& ctx) const override;


  bool HasSupportedInputsImpl(const onnx::NodeProto& node,
                                     const NeuronGraphContext& ctx,
                                     const OpSupportCheckParams& /*params*/) const override {
    return IsTypeSupported(node, ctx, 0, /*is_input*/ true,
                           {onnx::TensorProto::FLOAT,
                            onnx::TensorProto::FLOAT16,
                            onnx::TensorProto::UINT8,
                            onnx::TensorProto::INT8}) &&
           IsTypeSupported(node, ctx, 1, /*is_input*/ true,
                           {onnx::TensorProto::INT64,
                            onnx::TensorProto::INT32});
  }
};

void GatherOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder,
                                              const onnx::NodeProto& node,
                                              const NeuronGraphContext& ctx) const {
  // Skip the indices input if it is an initializer (we consume it inline).
  const auto& indices_name = node.input(1);
  if (ctx.IsInitializer(indices_name)) {
    model_builder.AddInitializerToSkip(indices_name);
  }
}

NeuronStatus GatherOpBuilder::AddToModelBuilderImpl(
    ModelBuilder& model_builder,
    const onnx::NodeProto& node,
    const NeuronGraphContext& ctx) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());

  const auto& input1        = node.input(0);
  const auto& indices_input = node.input(1);
  const auto& output        = node.output(0);

  const int32_t rank = static_cast<int32_t>(shaper[input1].size());
  const int32_t axis = static_cast<int32_t>(
      HandleNegativeAxis(NeuronGraphContext::GetAttrInt(node, "axis", 0), rank));

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input1));
  ADD_SCALAR_OPERAND(model_builder, input_indices, axis);

  auto output_shape = shaper[output];
  bool need_squeeze = false;

  // Read the indices initializer and convert to int32.
  const auto* indices_tensor = ctx.GetInitializer(indices_input);
  NEURON_RETURN_IF_NOT(indices_tensor != nullptr,
                        "Gather: indices must be a constant initializer for non-int32 types");

  const auto data_type = indices_tensor->data_type();
  Shape indices_dimen;
  indices_dimen.reserve(static_cast<size_t>(indices_tensor->dims_size()));
  for (int i = 0; i < indices_tensor->dims_size(); ++i)
    indices_dimen.push_back(static_cast<uint32_t>(indices_tensor->dims(i)));

  std::vector<int32_t> indices;
  if (data_type == onnx::TensorProto::INT64) {
    std::vector<int64_t> raw;
    NEURON_RETURN_IF_ERROR(NeuronGraphContext::GetInt64Data(*indices_tensor, raw));
    indices.reserve(raw.size());
    for (int64_t v : raw) indices.push_back(static_cast<int32_t>(v));
  } else if (data_type == onnx::TensorProto::INT32) {
    NEURON_RETURN_IF_ERROR(NeuronGraphContext::GetInt32Data(*indices_tensor, indices));
  } else {
    return NeuronStatus{false, MakeNeuronString("Gather: unsupported indices data type: ", data_type)};
  }

  // If indices is a scalar (0-D), insert a fake size-1 dimension to avoid Neuron errors.
  if (indices_dimen.empty()) {
    indices_dimen.push_back(1);
    output_shape.insert(output_shape.begin() + axis, 1);
    need_squeeze = true;
  }

  const OperandType indices_operand_type(Type::TENSOR_INT32, indices_dimen);
  NEURON_RETURN_IF_ERROR(
      model_builder.AddOperandFromPersistMemoryBuffer(indices_input, indices.data(),
                                                      indices_operand_type));
  input_indices.push_back(operand_indices.at(indices_input));

  // Build the output operand type from the element type of the output value.
  const int32_t out_elem_type = ctx.GetElementType(output);
  OnnxDataTypeInfo output_info;
  output_info.onnx_data_type = out_elem_type;
  const OperandType output_operand_type(output_info, output_shape);

  if (!need_squeeze) {
    return model_builder.AddOperation(NEURON_GATHER, input_indices,
                                       {output}, {output_operand_type});
  }

  // Scalar-indices path: gather into intermediate, then squeeze the injected axis.
  const std::string intermediate_output =
      model_builder.GetUniqueName(node.name() + "_need_squeeze");
  shaper.AddShape(intermediate_output, output_shape);

  NEURON_RETURN_IF_ERROR(model_builder.AddOperation(NEURON_GATHER, input_indices,
                                                     {intermediate_output}, {output_operand_type}));

  const std::string squeeze_op_name = model_builder.GetUniqueName(node.name() + "_squeeze");
  return op_builder_helpers::AddSqueezeOp(model_builder, squeeze_op_name,
                                           intermediate_output, output, {axis});
}
void CreateGatherOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<GatherOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace neuron
}  // namespace onnxruntime
