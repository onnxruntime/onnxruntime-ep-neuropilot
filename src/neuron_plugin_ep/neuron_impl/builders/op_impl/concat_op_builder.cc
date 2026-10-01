// concat_op_builder.cc
// Plugin EP version – uses onnx::NodeProto + NeuronGraphContext.
//
// CHANGES FROM ORIGINAL:
//   node_unit.Inputs()[i].node_arg.Name() → node.input(i)
//   node_unit.Outputs()[i].node_arg.Name() → node.output(i)
//   helper.Get("axis", ...) → NeuronGraphContext::GetAttrInt(node, "axis", ...)
//   GetQuantizationScaleAndZeroPoint(graph_viewer, ...) → ctx.GetScalarFloat / GetScalarInt32
//   ORT_RETURN_IF_NOT → NEURON_RETURN_IF_NOT
//   ORT_RETURN_IF_ERROR → NEURON_RETURN_IF_ERROR
//   Status → NeuronStatus
//
// Note: In the plugin EP, QDQ concat is not supported (no NodeUnit quant_param).
// Only non-quantized Concat (FLOAT, UINT8) is handled.

#include "base_op_builder.h"

namespace onnxruntime {
namespace neuron {
using namespace onnxruntime::neuron::op_builder_helpers;
using namespace onnxruntime::neuron::wrapper;

class ConcatOpBuilder : public BaseOpBuilder {
 private:
  NeuronStatus AddToModelBuilderImpl(ModelBuilder& model_builder,
                                      const onnx::NodeProto& node,
                                      const NeuronGraphContext& ctx) const override;


  bool HasSupportedInputsImpl(const onnx::NodeProto& node,
                                     const NeuronGraphContext& ctx,
                                     const OpSupportCheckParams& params) const override;
};

NeuronStatus ConcatOpBuilder::AddToModelBuilderImpl(
    ModelBuilder& model_builder,
    const onnx::NodeProto& node,
    const NeuronGraphContext& ctx) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  const int node_input_size = node.input_size();
  const auto& input0 = node.input(0);

  // For uint8 non-quantized concat: verify all inputs share the same scale/zp.
  if (operand_types.at(input0).type == Type::TENSOR_QUANT8_ASYMM) {
    auto scale      = operand_types.at(input0).operandType.scale;
    auto zero_point = operand_types.at(input0).operandType.zeroPoint;

    NEURON_RETURN_IF_NOT(scale > 0, "NNAPI requires scale to be > 0.");

    for (int i = 1; i < node_input_size; i++) {
      const auto& type = operand_types.at(node.input(i));
      NEURON_RETURN_IF_NOT(scale == type.operandType.scale,
                           "Input[", i, "]'s scale: ", type.operandType.scale,
                           " is different than input[0]'s scale: ", scale);
      NEURON_RETURN_IF_NOT(zero_point == type.operandType.zeroPoint,
                           "Input[", i, "]'s zero_point: ", type.operandType.zeroPoint,
                           " is different than input[0]'s zero_point: ", zero_point);
    }
  }

  std::vector<uint32_t> input_indices;
  for (int i = 0; i < node_input_size; i++) {
    input_indices.push_back(operand_indices.at(node.input(i)));
  }

  // For QDQ-absorbed Concat, use the Q node's scale/ZP for the output so
  // Neuron sees a consistent quantized output rather than scale=0.
  const auto* qdq = ctx.GetQDQGroup(node.output(0));
  float y_scale       = 0.0f;
  int32_t y_zero_point = 0;
  if (qdq && qdq->output_quant_params.has_value()) {
    y_scale      = qdq->output_quant_params->scale;
    y_zero_point = qdq->output_quant_params->zero_point;
  } else {
    y_scale      = operand_types.at(input0).operandType.scale;
    y_zero_point = operand_types.at(input0).operandType.zeroPoint;
  }

  // Normalize the axis against the ONNX rank, then shift it down by however
  // many leading unit dims the operand had squeezed away.
  int32_t onnx_rank = static_cast<int32_t>(ctx.GetShape(input0).size());
  int32_t rank = static_cast<int32_t>(shaper[input0].size());
  if (onnx_rank < rank) onnx_rank = rank;
  int32_t axis = static_cast<int32_t>(
      HandleNegativeAxis(NeuronGraphContext::GetAttrInt(node, "axis", 1), onnx_rank));
  axis -= onnx_rank - rank;
  NEURON_RETURN_IF_NOT(axis >= 0, "Concat: axis lies on a squeezed leading unit dim");

  ADD_SCALAR_OPERAND(model_builder, input_indices, axis);

  const auto& output = node.output(0);
  OperandType output_operand_type(operand_types.at(input0).type, shaper[output], y_scale, y_zero_point);
  NEURON_RETURN_IF_ERROR(model_builder.AddOperation(NEURON_CONCATENATION, input_indices,
                                                    {output}, {output_operand_type}));
  return NeuronStatus::OK();
}

// Operator support related
bool ConcatOpBuilder::HasSupportedInputsImpl(
    const onnx::NodeProto& node,
    const NeuronGraphContext& ctx,
    const OpSupportCheckParams& params) const {
  const int input_size = node.input_size();
  for (int i = 0; i < input_size; i++) {
    if (!IsTypeSupported(node, ctx, static_cast<size_t>(i), /*is_input*/ true,
                         {onnx::TensorProto::FLOAT}) &&
        !IsTypeSupported(node, ctx, static_cast<size_t>(i), /*is_input*/ true,
                         {onnx::TensorProto::UINT8,
                          onnx::TensorProto::INT8})) {
      return false;
    }
  }
  if (!IsTypeSupported(node, ctx, 0, /*is_input*/ false,
                       {onnx::TensorProto::FLOAT}) &&
      !IsTypeSupported(node, ctx, 0, /*is_input*/ false,
                       {onnx::TensorProto::UINT8,
                        onnx::TensorProto::INT8})) {
    return false;
  }
  return true;
}

void CreateConcatOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ConcatOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace neuron
}  // namespace onnxruntime
