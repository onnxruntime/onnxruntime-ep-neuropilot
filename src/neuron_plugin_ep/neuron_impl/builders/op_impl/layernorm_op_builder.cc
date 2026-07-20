// layernorm_op_builder.cc
// Plugin EP version – uses onnx::NodeProto + NeuronGraphContext.
//
// CHANGES FROM ORIGINAL:
//   node_unit.Inputs()[i].node_arg.Name() → node.input(i)
//   node_unit.Outputs()[i].node_arg.Name() → node.output(i)
//   helper.Get("axis", ...) → NeuronGraphContext::GetAttrInt(node, "axis", ...)
//   helper.Get("epsilon", ...) → NeuronGraphContext::GetAttrFloat(node, "epsilon", ...)
//   GetOnnxDataTypeInfoFromNodeUnit → ctx.GetElementType / operand_types.at(name)
//   model_builder.AddOperandFromConstantInitializer(node_unit, ArgType::kInput, idx, ...)
//     → model_builder.AddOperandFromConstantInitializer(node, ctx, idx, ...)
//   graph_viewer.IsConstantInitializer → ctx.IsInitializer
//   ORT_RETURN_IF_ERROR → NEURON_RETURN_IF_ERROR
//   Status → NeuronStatus
//
// The MTK extension operation (MTKEXT_LAYER_NORMALIZATION) path is preserved.
// Quantized input/output dequantize/quantize wrapping is preserved.

#include "base_op_builder.h"

namespace onnxruntime {
namespace neuron {
using namespace onnxruntime::neuron::op_builder_helpers;
using namespace onnxruntime::neuron::wrapper;

class LayerNormalizationOpBuilder : public BaseOpBuilder {
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

void LayerNormalizationOpBuilder::AddInitializersToSkip(
    ModelBuilder& model_builder,
    const onnx::NodeProto& node,
    const NeuronGraphContext& /*ctx*/) const {
  model_builder.AddInitializerToSkip(node.input(1));  // gamma (scale)
  model_builder.AddInitializerToSkip(node.input(2));  // beta
}

bool LayerNormalizationOpBuilder::HasSupportedInputOutputsImpl(
    const onnx::NodeProto& node,
    const NeuronGraphContext& ctx,
    const OpSupportCheckParams& params) const {
  return IsTypeSupported(node, ctx, 0, /*is_input*/ true,
                         {onnx::TensorProto::FLOAT,
                          onnx::TensorProto::UINT8,
                          onnx::TensorProto::INT8});
}

NeuronStatus LayerNormalizationOpBuilder::AddToModelBuilderImpl(
    ModelBuilder& model_builder,
    const onnx::NodeProto& node,
    const NeuronGraphContext& ctx) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());

  std::string input = node.input(0);
  const auto& output = node.output(0);

  const int32_t input_elem_type = ctx.GetElementType(input);

  // If input is not float32, dequantize to float32 first.
  if (input_elem_type != onnx::TensorProto::FLOAT) {
    const std::string mediate_input = model_builder.GetUniqueName(output + "_quant_mediate");
    shaper.AddShape(mediate_input, shaper[input]);

    // Build OnnxDataTypeInfo for the dequantize call.
    wrapper::OnnxDataTypeInfo input_info;
    input_info.onnx_data_type = input_elem_type;
    // Scale and zero-point come from the preceding DQ node; at this stage the
    // operand_types entry carries the quant params registered by the DQ builder.
    const auto& operand_types = model_builder.GetOperandTypes();
    input_info.scales    = {operand_types.at(input).operandType.scale};
    input_info.zero_points = {operand_types.at(input).operandType.zeroPoint};

    NEURON_RETURN_IF_ERROR(model_builder.AddDequantizeLinear(input, input_info, mediate_input));
    input = mediate_input;
  }

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));

  // Axis operand
  const int32_t axis = static_cast<int32_t>(NeuronGraphContext::GetAttrInt(node, "axis", -1));
  const Shape axis_shape{1};
  const std::string axis_name = model_builder.GetUniqueName(node.name() + input + "_axis");
  const OperandType axis_operand_type(Type::TENSOR_INT32, axis_shape);
  NEURON_RETURN_IF_ERROR(
      model_builder.AddOperandFromPersistMemoryBuffer(axis_name, &axis, axis_operand_type));
  input_indices.push_back(operand_indices.at(axis_name));

  // Scale (gamma) and beta: always converted to float32.
  const std::string scale_name =
      model_builder.GetUniqueName(node.name() + node.input(1) + "/converted_f32");
  const std::string bias_name =
      model_builder.GetUniqueName(node.name() + node.input(2) + "/converted_f32");

  NEURON_RETURN_IF_ERROR(
      model_builder.AddOperandFromConstantInitializer(node, ctx, 1,
                                                      /* dequantize_to_float32 */ true,
                                                      ModelBuilder::DataLayout::NO_CHANGE,
                                                      scale_name));
  NEURON_RETURN_IF_ERROR(
      model_builder.AddOperandFromConstantInitializer(node, ctx, 2,
                                                      /* dequantize_to_float32 */ true,
                                                      ModelBuilder::DataLayout::NO_CHANGE,
                                                      bias_name));

  input_indices.push_back(operand_indices.at(scale_name));
  input_indices.push_back(operand_indices.at(bias_name));

  // Epsilon
  const auto eps = NeuronGraphContext::GetAttrFloat(node, "epsilon", 1e-5f);
  const Shape eps_shape{1};
  const std::string eps_name = model_builder.GetUniqueName(node.name() + input + "_epsilon");
  const OperandType eps_operand_type(Type::TENSOR_FLOAT32, eps_shape);
  NEURON_RETURN_IF_ERROR(
      model_builder.AddOperandFromPersistMemoryBuffer(eps_name, &eps, eps_operand_type));
  input_indices.push_back(operand_indices.at(eps_name));

  // MTK extension operation name operand
  const std::string ext_name = "MTKEXT_LAYER_NORMALIZATION";
  uint32_t mtxext_operand_index;
  const auto mtxext_operand_name = model_builder.GetUniqueName(node.name() + "_mtkext");
  NEURON_RETURN_IF_ERROR(
      model_builder.AddCustomNameStringOperand(ext_name, mtxext_operand_name, mtxext_operand_index));
  input_indices.push_back(operand_indices.at(mtxext_operand_name));

  NeuronOperationType ext_op_type;
  NEURON_RETURN_IF_ERROR(model_builder.GetExtensionGeneralOperationType(&ext_op_type));

  const int32_t output_elem_type = ctx.GetElementType(output);

  if (output_elem_type != onnx::TensorProto::FLOAT) {
    // Output is quantized: write float32 intermediate, then quantize.
    const std::string mediate_output = model_builder.GetUniqueName(output + "_quant_mediate");
    const OperandType mediate_output_operand_type(Type::FLOAT32, shaper[input]);
    NEURON_RETURN_IF_ERROR(model_builder.AddOperation(ext_op_type, input_indices,
                                                      {mediate_output},
                                                      {mediate_output_operand_type}));

    wrapper::OnnxDataTypeInfo output_info;
    output_info.onnx_data_type = output_elem_type;
    const auto& operand_types = model_builder.GetOperandTypes();
    output_info.scales     = {operand_types.at(output).operandType.scale};
    output_info.zero_points = {operand_types.at(output).operandType.zeroPoint};
    NEURON_RETURN_IF_ERROR(model_builder.AddQuantizeLinear(mediate_output, output_info, output));
  } else {
    const OperandType output_operand_type(Type::FLOAT32, shaper[input]);
    NEURON_RETURN_IF_ERROR(model_builder.AddOperation(ext_op_type, input_indices,
                                                      {output},
                                                      {output_operand_type}));
  }
  return NeuronStatus::OK();
}

// Operator support related
void CreateLayerNormalizationOpBuilder(const std::string& op_type,
                                        OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<LayerNormalizationOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace neuron
}  // namespace onnxruntime
