// binary_op_builder.cc
// Plugin EP version – uses onnx::NodeProto + NeuronGraphContext.
//
// Handles: Add, Sub, Mul, Div, QLinearAdd, QLinearMul, Pow, PRelu
//
// CHANGES FROM ORIGINAL:
//   node_unit.Inputs()[i].node_arg.Name() → node.input(i)
//   node_unit.Outputs()[i].node_arg.Name() → node.output(i)
//   IsQuantizedOp(node_unit) → uses op_type string
//   GetBinaryOpQuantizationScaleAndZeroPoint → ctx.GetScalarFloat / GetScalarInt32
//   ORT_RETURN_IF_ERROR → NEURON_RETURN_IF_ERROR
//   ORT_MAKE_STATUS → NeuronStatus{false, MakeNeuronString(...)}
//   Status → NeuronStatus

#include "base_op_builder.h"

namespace onnxruntime {
namespace neuron {
using namespace onnxruntime::neuron::op_builder_helpers;
using namespace onnxruntime::neuron::wrapper;

class BinaryOpBuilder : public BaseOpBuilder {
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
                                     const neuron::OpSupportCheckParams& params) const override;

  int GetMinSupportedOpSet(const onnx::NodeProto& node) const override;

  bool IsQuantizedOp(const onnx::NodeProto& node,
                     const NeuronGraphContext& /*ctx*/) const override {
    const auto& op = node.op_type();
    return op == "QLinearAdd" || op == "QLinearMul";
  }
};

// Add operator related

void BinaryOpBuilder::AddInitializersToSkip(ModelBuilder& /*model_builder*/,
                                             const onnx::NodeProto& /*node*/,
                                             const NeuronGraphContext& /*ctx*/) const {
  // In the plugin EP, quantized inputs come as DQ nodes in the graph.
  // The Q params (scale/zp tensors) for QLinear ops are graph initializers
  // referenced directly via node.input(1..7); they are consumed by the op builder
  // directly and do not need to be registered as separate operands, so we skip them.
}

NeuronStatus BinaryOpBuilder::AddToModelBuilderImpl(
    ModelBuilder& model_builder,
    const onnx::NodeProto& node,
    const NeuronGraphContext& ctx) const {
  const auto& op_type = node.op_type();

  int32_t op_code = 0;
  bool add_activation = true;
  if (op_type == "Add" || op_type == "QLinearAdd") {
    op_code = NEURON_ADD;
  } else if (op_type == "Sub") {
    op_code = NEURON_SUB;
  } else if (op_type == "Mul" || op_type == "QLinearMul") {
    op_code = NEURON_MUL;
  } else if (op_type == "Div") {
    op_code = NEURON_DIV;
  } else if (op_type == "Pow") {
    add_activation = false;
    op_code = NEURON_POW;
  } else if (op_type == "PRelu") {
    add_activation = false;
    op_code = NEURON_PRELU;
  } else {
    return NeuronStatus{false, MakeNeuronString("BinaryOpBuilder: unknown op ", op_type)};
  }

  const bool is_quantized = IsQuantizedOp(node, ctx);

  // ---- Non-quantized path (also handles QDQ-absorbed Add/Sub) ---------------
  if (!is_quantized) {
    const auto& input1 = node.input(0);
    const auto& input2 = node.input(1);
    const auto& output = node.output(0);

    // For QDQ-absorbed ops (DQ→Add→Q pattern), the inputs are QUANT8 (not
    // FLOAT32).  The output scale/ZP must come from the Q node's quant params
    // stored in the QDQ group; using scale=0 causes Neuron to reject the op.
    float out_scale = 0.0f;
    int32_t out_zp  = 0;
    const auto* qdq = ctx.GetQDQGroup(output);
    if (qdq && qdq->output_quant_params.has_value()) {
      out_scale = qdq->output_quant_params->scale;
      out_zp    = qdq->output_quant_params->zero_point;
    }

    int32_t fuse_code = NEURON_FUSED_NONE;
    if (add_activation) {
      fuse_code = model_builder.FindActivation(node, ctx);
    }

    NEURON_RETURN_IF_ERROR(AddBinaryOperator(op_code, model_builder,
                                             input1, input2,
                                             add_activation, fuse_code,
                                             output, out_scale, out_zp));
    return NeuronStatus::OK();
  }

  // ---- Quantized path (QLinearAdd / QLinearMul) ----------------------------
  // QLinearAdd inputs:  a(0), a_scale(1), a_zp(2), b(3), b_scale(4), b_zp(5),
  //                     C_scale(6), C_zp(7)
  if (node.input_size() < 8) {
    return NeuronStatus{false, "QLinear binary op: expected 8 inputs"};
  }

  const auto& a      = node.input(0);
  const auto& b      = node.input(3);
  const auto& output = node.output(0);

  float a_scale = 0.0f, b_scale = 0.0f, y_scale = 0.0f;
  int32_t a_zp = 0, b_zp = 0, y_zp = 0;
  NEURON_RETURN_IF_ERROR(ctx.GetScalarFloat(node.input(1), a_scale));
  NEURON_RETURN_IF_ERROR(ctx.GetScalarInt32(node.input(2), a_zp));
  NEURON_RETURN_IF_ERROR(ctx.GetScalarFloat(node.input(4), b_scale));
  NEURON_RETURN_IF_ERROR(ctx.GetScalarInt32(node.input(5), b_zp));
  NEURON_RETURN_IF_ERROR(ctx.GetScalarFloat(node.input(6), y_scale));
  NEURON_RETURN_IF_ERROR(ctx.GetScalarInt32(node.input(7), y_zp));

  NEURON_RETURN_IF_ERROR(AddBinaryOperator(op_code, model_builder,
                                           a, b,
                                           /*add_activation*/ false, NEURON_FUSED_NONE,
                                           output, y_scale, y_zp));
  return NeuronStatus::OK();
}

// Operator support related

int BinaryOpBuilder::GetMinSupportedOpSet(const onnx::NodeProto& node) const {
  const auto& op = node.op_type();
  // Add/Sub/Mul/Div/Pow opset 6- has broadcast attributes we do not support
  if (op == "Add" || op == "Sub" || op == "Mul" || op == "Div" || op == "Pow")
    return 7;
  return 1;
}

bool BinaryOpBuilder::HasSupportedInputOutputsImpl(
    const onnx::NodeProto& node,
    const NeuronGraphContext& ctx,
    const OpSupportCheckParams& params) const {
  if (IsQuantizedOp(node, ctx)) {
    // QLinear ops: input 0 is the first tensor
    return IsTypeSupported(node, ctx, 0, /*is_input*/ true,
                           {onnx::TensorProto::UINT8, onnx::TensorProto::INT8});
  }
  // Regular binary ops: float, float16, or int32
  return IsTypeSupported(node, ctx, 0, /*is_input*/ true,
                         {onnx::TensorProto::FLOAT,
                          onnx::TensorProto::FLOAT16,
                          onnx::TensorProto::INT32});
}
void CreateBinaryOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  CreateSharedOpBuilderImpl<BinaryOpBuilder>(
      op_type, op_registrations,
      {
          "Add",
          "Sub",
          "Mul",
          "Div",
          "QLinearAdd",
          "QLinearMul",
          "Pow",
          "PRelu",
      });
}

}  // namespace neuron
}  // namespace onnxruntime
