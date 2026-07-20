// instancenorm_op_builder.cc
// Plugin EP version – uses onnx::NodeProto + NeuronGraphContext.
//
// CHANGES FROM ORIGINAL:
//   node_unit.Inputs()[i].node_arg.Name() → node.input(i)
//   node_unit.Outputs()[i].node_arg.Name() → node.output(i)
//   helper.Get("epsilon", ...) → NeuronGraphContext::GetAttrFloat(node, "epsilon", ...)
//   GetOnnxDataTypeInfoFromNodeUnit → ctx.GetElementType(name) / operand_types.at(name)
//   model_builder.AddOperandFromConstantInitializer(node_unit, ArgType::kInput, idx, ...) →
//     model_builder.AddOperandFromConstantInitializer(node, ctx, idx, ...)
//   graph_viewer.IsConstantInitializer → ctx.IsInitializer
//   node_unit.Domain() == kMSInternalNHWCDomain → IsNodeLayoutNHWC(node)
//   ORT_RETURN_IF_ERROR → NEURON_RETURN_IF_ERROR
//   Status → NeuronStatus

#include "base_op_builder.h"

namespace onnxruntime {
namespace neuron {
using namespace onnxruntime::neuron::op_builder_helpers;
using namespace onnxruntime::neuron::wrapper;

class InstanceNormalizationOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder,
                              const onnx::NodeProto& node,
                              const NeuronGraphContext& ctx) const override;

 private:
  NeuronStatus AddToModelBuilderImpl(ModelBuilder& model_builder,
                                      const onnx::NodeProto& node,
                                      const NeuronGraphContext& ctx) const override;

  bool HasSupportedInputOutputsImpl(const onnx::NodeProto& node,
                                     const NeuronGraphContext& ctx,
                                     const OpSupportCheckParams& params) const override;

};

// Add operator related

void InstanceNormalizationOpBuilder::AddInitializersToSkip(
    ModelBuilder& model_builder,
    const onnx::NodeProto& node,
    const NeuronGraphContext& /*ctx*/) const {
  model_builder.AddInitializerToSkip(node.input(2));  // B (bias)
}

bool InstanceNormalizationOpBuilder::HasSupportedInputOutputsImpl(
    const onnx::NodeProto& node,
    const NeuronGraphContext& ctx,
    const OpSupportCheckParams& params) const {
  return IsTypeSupported(node, ctx, 0, /*is_input*/ true,
                         {onnx::TensorProto::FLOAT,
                          onnx::TensorProto::UINT8,
                          onnx::TensorProto::INT8});
}

NeuronStatus InstanceNormalizationOpBuilder::AddToModelBuilderImpl(
    ModelBuilder& model_builder,
    const onnx::NodeProto& node,
    const NeuronGraphContext& ctx) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());

  const auto& input  = node.input(0);
  const auto& scale  = node.input(1);
  const auto& bias   = node.input(2);
  const auto& output = node.output(0);

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));

  // Add scale operand (constant initializer, no layout change)
  NEURON_RETURN_IF_ERROR(
      model_builder.AddOperandFromConstantInitializer(node, ctx, 1,
                                                      /* dequantize_to_float32 */ false,
                                                      ModelBuilder::DataLayout::NO_CHANGE,
                                                      scale));

  // Add bias operand. If INT32, strip per-channel quant params as Neuron
  // manages them internally.
  const int32_t bias_elem_type = ctx.GetElementType(bias);
  if (bias_elem_type == onnx::TensorProto::INT32) {
    const auto* initializer = ctx.GetInitializer(bias);
    NEURON_RETURN_IF_NOT(initializer != nullptr,
                         "InstanceNormalization: bias initializer not found");
    // Build a plain INT32 OperandType (no quant params)
    const OperandType bias_operand_type(Type::TENSOR_INT32, shaper[bias]);
    const void* raw = op_builder_helpers::GetRawInitializerData(*initializer);
    NEURON_RETURN_IF_NOT(raw != nullptr,
                         "InstanceNormalization: bias initializer has no data");
    NEURON_RETURN_IF_ERROR(
        model_builder.AddOperandFromPersistMemoryBuffer(bias, raw, bias_operand_type));
  } else {
    NEURON_RETURN_IF_ERROR(
        model_builder.AddOperandFromConstantInitializer(node, ctx, 2,
                                                        /* dequantize_to_float32 */ false,
                                                        ModelBuilder::DataLayout::NO_CHANGE,
                                                        bias));
  }

  input_indices.push_back(operand_indices.at(scale));
  input_indices.push_back(operand_indices.at(bias));

  // Epsilon
  const auto eps = NeuronGraphContext::GetAttrFloat(node, "epsilon", 1e-5f);
  const Shape eps_shape{1};
  const std::string eps_name = model_builder.GetUniqueName(node.name() + input + "_epsilon");
  const OperandType eps_operand_type(Type::TENSOR_FLOAT32, eps_shape);
  NEURON_RETURN_IF_ERROR(
      model_builder.AddOperandFromPersistMemoryBuffer(eps_name, &eps, eps_operand_type));
  input_indices.push_back(operand_indices.at(eps_name));

  // NCHW indicator: 0 = NHWC, 1 = NCHW.
  // op_capability library guarantees only NHWC nodes reach here, so this is always 0.
  const std::vector<int32_t> nchw_data = {IsNodeLayoutNHWC(node) ? 0 : 1};
  const std::string nchw_name = model_builder.GetUniqueName(node.name() + input + "_nchw");
  const OperandType nchw_operand_type(Type::TENSOR_INT32, {1});
  NEURON_RETURN_IF_ERROR(
      model_builder.AddOperandFromPersistMemoryBuffer(nchw_name, nchw_data.data(), nchw_operand_type));
  input_indices.push_back(operand_indices.at(nchw_name));

  // Output operand type: same shape as input, same element type.
  const auto& operand_types = model_builder.GetOperandTypes();
  const OperandType output_operand_type(operand_types.at(input).type, shaper[input]);
  NEURON_RETURN_IF_ERROR(model_builder.AddOperation(NEURON_INSTANCE_NORMALIZATION, input_indices,
                                                    {output}, {output_operand_type}));
  return NeuronStatus::OK();
}

// Operator support related
void CreateInstanceNormalizationOpBuilder(const std::string& op_type,
                                           OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<InstanceNormalizationOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace neuron
}  // namespace onnxruntime
