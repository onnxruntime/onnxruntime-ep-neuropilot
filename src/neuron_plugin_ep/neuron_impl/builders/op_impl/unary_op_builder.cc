// unary_op_builder.cc
// Plugin EP version – uses onnx::NodeProto + NeuronGraphContext.
// Shared by: Abs, Exp, Floor, Log, Sigmoid, Neg, Sin, Sqrt, Tanh, QLinearSigmoid.

#include "base_op_builder.h"

namespace onnxruntime {
namespace neuron {
using namespace onnxruntime::neuron::op_builder_helpers;
using namespace onnxruntime::neuron::wrapper;

class UnaryOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder,
                              const onnx::NodeProto& node,
                              const NeuronGraphContext& ctx) const override;

 private:
  bool IsQuantizedOp(const onnx::NodeProto& node,
                     const NeuronGraphContext& /*ctx*/) const override {
    return node.op_type() == "QLinearSigmoid";
  }

  NeuronStatus AddToModelBuilderImpl(ModelBuilder& model_builder,
                                      const onnx::NodeProto& node,
                                      const NeuronGraphContext& ctx) const override;


  bool HasSupportedInputsImpl(const onnx::NodeProto& node,
                                     const NeuronGraphContext& ctx,
                                     const OpSupportCheckParams& params) const override;

  int GetMinSupportedOpSet(const onnx::NodeProto& node) const override;

  static bool IsQuantizedOpSupported(const NeuronGraphContext& ctx,
                                     const onnx::NodeProto& node,
                                     const OpSupportCheckParams& params);
};

// QLinearSigmoid inputs: [X, X_scale, X_zero_point, Y_scale, Y_zero_point]
void UnaryOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder,
                                            const onnx::NodeProto& node,
                                            const NeuronGraphContext& ctx) const {
  if (!IsQuantizedOp(node, ctx))
    return;

  // Skip X_scale, X_zero_point, Y_scale, Y_zero_point
  if (node.input_size() > 1) AddQuantizationScaleAndZeroPointToSkip(model_builder, node.input(1), node.input(2));
  if (node.input_size() > 3) AddQuantizationScaleAndZeroPointToSkip(model_builder, node.input(3), node.input(4));
}

NeuronStatus UnaryOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                    const onnx::NodeProto& node,
                                                    const NeuronGraphContext& ctx) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto& op_type = node.op_type();

  const auto& input  = node.input(0);
  const auto& output = node.output(0);

  const bool is_qlinear_sigmoid = (op_type == "QLinearSigmoid");

  int32_t op_code;
  if (op_type == "Abs")
    op_code = NEURON_ABS;
  else if (op_type == "Exp")
    op_code = NEURON_EXP;
  else if (op_type == "Floor")
    op_code = NEURON_FLOOR;
  else if (op_type == "Log")
    op_code = NEURON_LOG;
  else if (op_type == "Sigmoid" || is_qlinear_sigmoid)
    op_code = NEURON_LOGISTIC;
  else if (op_type == "Neg")
    op_code = NEURON_NEG;
  else if (op_type == "Sin")
    op_code = NEURON_SIN;
  else if (op_type == "Sqrt")
    op_code = NEURON_SQRT;
  else if (op_type == "Tanh")
    op_code = NEURON_TANH;
  else
    return NeuronStatus{false, MakeNeuronString("UnaryOpBuilder: unknown op: ", op_type)};

  float y_scale = 0.0f;
  int32_t y_zero_point = 0;

  const Type input_operand_type = operand_types.at(input).type;
  const bool output_is_quantized =
      input_operand_type == Type::TENSOR_QUANT8_ASYMM_SIGNED ||
      input_operand_type == Type::TENSOR_QUANT8_ASYMM;

  if (is_qlinear_sigmoid) {
    // QLinearSigmoid: inputs[1]=X_scale, inputs[2]=X_zero_point
    float x_scale = 0.0f;
    int32_t x_zero_point = 0;
    NEURON_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(
        ctx, node.input(1), node.input(2), x_scale, x_zero_point));
    NEURON_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input, x_scale, x_zero_point));
  }

  // NEURON_LOGISTIC requires a FIXED output quantization: scale = 1/256, with
  // zero_point = -128 for ASYMM_SIGNED (or 0 for ASYMM).  This must also be
  // applied to a plain `Sigmoid` absorbed into a DQ→Sigmoid→Q group — not only
  // to QLinearSigmoid.  Otherwise the output operand is registered with scale 0
  // and the MDLA compiler rejects it ("Act1/Act2 shift exceeds range").  A plain
  // float (non-quantized) Sigmoid keeps y_scale = 0 (unused for FLOAT32).
  if (op_code == NEURON_LOGISTIC && output_is_quantized) {
    y_scale = 1.f / 256;
    y_zero_point =
        (input_operand_type == Type::TENSOR_QUANT8_ASYMM_SIGNED) ? -128 : 0;
  }

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output],
                                        y_scale, y_zero_point);
  NEURON_RETURN_IF_ERROR(model_builder.AddOperation(op_code, input_indices,
                                                     {output}, {output_operand_type}));
  return NeuronStatus::OK();
}
bool UnaryOpBuilder::HasSupportedInputsImpl(const onnx::NodeProto& node,
                                                   const NeuronGraphContext& ctx,
                                                   const OpSupportCheckParams& params) const {
  // For QLinearSigmoid, verify quantized I/O types
  if (node.op_type() != "QLinearSigmoid")
    return BaseOpBuilder::HasSupportedInputsImpl(node, ctx, params);

  if (!IsQuantizedIOSupported(ctx, node, {0}, params, /*is_input*/ true))
    return false;

  if (!IsQuantizedIOSupported(ctx, node, {0}, params, /*is_input*/ false))
    return false;

  return true;
}

int UnaryOpBuilder::GetMinSupportedOpSet(const onnx::NodeProto& node) const {
  // QLinearSigmoid is a contrib op, OpSet is always 1
  if (node.op_type() == "QLinearSigmoid")
    return 1;

  // All other ops: opset 6+ (opset 5- used consumed_inputs attribute)
  return 6;
}

/* static */ bool UnaryOpBuilder::IsQuantizedOpSupported(const NeuronGraphContext& ctx,
                                                          const onnx::NodeProto& node,
                                                          const OpSupportCheckParams& /*params*/) {
  // Neuron requires sigmoid output scale = 1/256 and zero point = 0.
  // QLinearSigmoid inputs[3] = Y_scale, inputs[4] = Y_zero_point
  if (node.input_size() < 5) {
    LOGS_NEURON(VERBOSE) << "QLinearSigmoid: insufficient inputs";
    return false;
  }
  const std::string op_desc = MakeNeuronString("Op [QLinearSigmoid] name [", node.name(), "]'s output 0 ");
  return HasRequiredScaleAndZeroPoint(ctx, op_desc,
                                      node.input(3) /* Y_scale */,
                                      node.input(4) /* Y_zero_point */,
                                      1.f / 256 /* required_scale */,
                                      0 /* required_zp */);
}

void CreateUnaryOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  CreateSharedOpBuilderImpl<UnaryOpBuilder>(
      op_type, op_registrations,
      {
          "Abs",
          "Exp",
          "Floor",
          "Log",
          "Sigmoid",
          "Neg",
          "Sin",
          "Sqrt",
          "Tanh",
          "QLinearSigmoid",
      });
}

}  // namespace neuron
}  // namespace onnxruntime
