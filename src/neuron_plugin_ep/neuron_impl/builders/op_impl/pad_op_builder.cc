// pad_op_builder.cc
// Plugin EP version – uses onnx::NodeProto + NeuronGraphContext.
//
// CHANGES FROM ORIGINAL:
//   node_unit.Inputs()[i].node_arg.Name() → node.input(i)
//   node_unit.Outputs()[i].node_arg.Name() → node.output(i)
//   model_builder.GetConstantInitializer(name) → ctx.GetInitializer(name)
//   Initializer(...).DataAsSpan<int64_t>() → NeuronGraphContext::GetInt64Data
//   Initializer(...).DataAsSpan<float>() → NeuronGraphContext::GetFloatData
//   helper.Get("mode", "constant") → NeuronGraphContext::GetAttrString(node, "mode", "constant")
//   inputs[i].node_arg.Exists() → !node.input(i).empty()
//   ORT_RETURN_IF_NOT → NEURON_RETURN_IF_NOT
//   ORT_RETURN_IF_ERROR → NEURON_RETURN_IF_ERROR
//   Status → NeuronStatus
//   Minimum supported opset: 11

#include "base_op_builder.h"
#include "../op_builder_helpers.h"

namespace onnxruntime {
namespace neuron {
using namespace onnxruntime::neuron::op_builder_helpers;
using namespace onnxruntime::neuron::wrapper;

class PadOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder,
                              const onnx::NodeProto& node,
                              const NeuronGraphContext& /*ctx*/) const override;

 private:
  // Before Pad-11, pads/constant_value were attributes — only support opset 11+.
  int GetMinSupportedOpSet(const onnx::NodeProto& /*node*/) const override { return 11; }

  NeuronStatus AddToModelBuilderImpl(ModelBuilder& model_builder,
                                      const onnx::NodeProto& node,
                                      const NeuronGraphContext& ctx) const override;

};

void PadOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder,
                                           const onnx::NodeProto& node,
                                           const NeuronGraphContext& /*ctx*/) const {
  model_builder.AddInitializerToSkip(node.input(1));  // pads
  if (node.input_size() > 2 && !node.input(2).empty())
    model_builder.AddInitializerToSkip(node.input(2));  // constant_value
}

NeuronStatus PadOpBuilder::AddToModelBuilderImpl(
    ModelBuilder& model_builder,
    const onnx::NodeProto& node,
    const NeuronGraphContext& ctx) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices = model_builder.GetOperandIndices();
  const auto& operand_types   = model_builder.GetOperandTypes();

  std::vector<uint32_t> input_indices;

  // data input
  const auto& data = node.input(0);
  input_indices.push_back(operand_indices.at(data));

  // pads input: convert from [begin_0, begin_1, ..., end_0, end_1, ...] (ONNX format, int64)
  // to [begin_0, end_0, begin_1, end_1, ...] (Neuron format, int32).
  const auto data_shape = shaper[data];
  const uint32_t data_rank = static_cast<uint32_t>(data_shape.size());

  const auto* pads_initializer = ctx.GetInitializer(node.input(1));
  NEURON_RETURN_IF_NOT(pads_initializer != nullptr, "Pad: pads must be a constant initializer");

  std::vector<int64_t> pads_int64;
  NEURON_RETURN_IF_ERROR(NeuronGraphContext::GetInt64Data(*pads_initializer, pads_int64));

  std::vector<int32_t> converted_pads;
  converted_pads.reserve(2 * data_rank);
  for (uint32_t i = 0; i < data_rank; ++i) {
    converted_pads.push_back(static_cast<int32_t>(pads_int64[i]));
    converted_pads.push_back(static_cast<int32_t>(pads_int64[i + data_rank]));
  }

  const Shape converted_pads_shape{data_rank, 2};
  const OperandType converted_pads_operand_type{Type::TENSOR_INT32, converted_pads_shape};
  NEURON_RETURN_IF_ERROR(
      model_builder.AddOperandFromPersistMemoryBuffer(node.input(1), converted_pads.data(),
                                                      converted_pads_operand_type));
  input_indices.push_back(operand_indices.at(node.input(1)));

  // constant_value input (optional, default 0.0f).
  float pad_value = 0.0f;
  if (node.input_size() > 2 && !node.input(2).empty()) {
    const auto* cv_initializer = ctx.GetInitializer(node.input(2));
    NEURON_RETURN_IF_NOT(cv_initializer != nullptr,
                          "Pad: constant_value must be a constant initializer");
    std::vector<float> cv_data;
    NEURON_RETURN_IF_ERROR(NeuronGraphContext::GetFloatData(*cv_initializer, cv_data));
    if (!cv_data.empty()) pad_value = cv_data[0];
  }

  ADD_SCALAR_OPERAND(model_builder, input_indices, pad_value);

  const auto& output = node.output(0);
  const auto& input_type = operand_types.at(data);
  const OperandType output_operand_type{input_type.type, shaper[output],
                                        input_type.operandType.scale,
                                        input_type.operandType.zeroPoint};

  return model_builder.AddOperation(NEURON_PAD_V2, input_indices, {output}, {output_operand_type});
}
void CreatePadOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<PadOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace neuron
}  // namespace onnxruntime
