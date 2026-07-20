// matmul_op_builder.cc
// Plugin EP version – uses onnx::NodeProto + NeuronGraphContext.
//
// Ported from in-tree matmul_op_builder.cc.
//
// Translation notes:
//   node_unit.Inputs()[i].node_arg.Name()       → node.input(i)
//   node_unit.Outputs()[i].node_arg.Name()       → node.output(i)
//   node_unit.OpType()                           → node.op_type()
//   node_unit.Name()                             → node.name()
//   ORT_RETURN_IF_ERROR / ORT_RETURN_IF_NOT      → NEURON_RETURN_IF_ERROR / NEURON_RETURN_IF_NOT
//   ORT_MAKE_STATUS(ONNXRUNTIME, ...)            → NeuronStatus{false, MakeNeuronString(...)}
//   Status::OK()                                 → NeuronStatus::OK()
//
// Quantized path:
//   QLinearMatMul / QDQMatMul: scale and zero-point are resolved via
//   GetConvMatMulOpQuantizationScaleAndZeroPoint(ctx, node, ...).
//
//   AddInitializersToSkip for the quantized path:
//     QLinearMatMul layout:
//       0: A, 1: a_scale, 2: a_zp, 3: B, 4: b_scale, 5: b_zp, 6: y_scale, 7: y_zp
//     QDQMatMul (same quant param layout but split across DQ/Q nodes in ORT).
//   We skip: a_scale(1), a_zp(2), b_scale(4), b_zp(5), y_scale(6), y_zp(7)
//   and possibly B(3) if the batch-matmul path is not used.

#include "base_op_builder.h"

#include <onnx/onnx_pb.h>

#include "../op_builder_helpers.h"

namespace onnxruntime {
namespace neuron {
using namespace onnxruntime::neuron::op_builder_helpers;
using namespace onnxruntime::neuron::wrapper;

class MatMulOpBuilder : public BaseOpBuilder {
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

  bool IsNodeUnitTypeSupported(const onnx::NodeProto& /*node*/) const override { return true; }

  bool IsQuantizedOp(const onnx::NodeProto& node,
                     const NeuronGraphContext& ctx) const override {
    return IsQuantizedMatMul(GetQuantizedOpType(node, ctx));
  }

};


// ---------------------------------------------------------------------------
// AddInitializersToSkip
// ---------------------------------------------------------------------------
void MatMulOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder,
                                              const onnx::NodeProto& node,
                                              const NeuronGraphContext& ctx) const {
  const bool use_neuron_batch_matmul = GreaterEqualVersion(model_builder.GetNeuronVersion(), {5, 1, 2});

  if (IsQuantizedOp(node, ctx)) {
    // QLinearMatMul: 0:A, 1:a_scale, 2:a_zp, 3:B, 4:b_scale, 5:b_zp, 6:y_scale, 7:y_zp
    if (node.input_size() > 1) AddQuantizationScaleAndZeroPointToSkip(model_builder, node.input(1),
        node.input_size() > 2 ? node.input(2) : std::string{});
    if (use_neuron_batch_matmul) {
      if (node.input_size() > 4) AddQuantizationScaleAndZeroPointToSkip(model_builder, node.input(4),
          node.input_size() > 5 ? node.input(5) : std::string{});
    } else {
      // Skip B and its quant params entirely.
      if (node.input_size() > 3 && !node.input(3).empty())
        AddInputToSkip(model_builder, node.input(3),
                       node.input_size() > 4 ? node.input(4) : std::string{},
                       node.input_size() > 5 ? node.input(5) : std::string{});
    }
    if (node.input_size() > 6) AddQuantizationScaleAndZeroPointToSkip(model_builder, node.input(6),
        node.input_size() > 7 ? node.input(7) : std::string{});
  } else if (!use_neuron_batch_matmul) {
    // Non-quantized 2-D MatMul: B must be an initializer that we transpose.
    if (node.input_size() > 1 && !node.input(1).empty())
      model_builder.AddInitializerToSkip(node.input(1));
  }
}

// ---------------------------------------------------------------------------
// HasSupportedInputOutputsImpl
// ---------------------------------------------------------------------------
bool MatMulOpBuilder::HasSupportedInputOutputsImpl(const onnx::NodeProto& node,
                                                     const NeuronGraphContext& ctx,
                                                     const OpSupportCheckParams& params) const {
  // For QLinearMatMul, the first tensor input is at index 0.
  if (!IsTypeSupported(node, ctx, 0, /*is_input*/ true,
                       {onnx::TensorProto::FLOAT,
                        onnx::TensorProto::UINT8,
                        onnx::TensorProto::INT8,
                        onnx::TensorProto::INT16,
                        onnx::TensorProto::UINT16})) {
    return false;
  }

  if (IsQuantizedOp(node, ctx)) {
    // Check quantized IO support.
    if (!IsQuantizedIOSupported(ctx, node, {0, 1}, params, /*is_input*/ true))
      return false;
    if (!IsQuantizedIOSupported(ctx, node, {0}, params, /*is_input*/ false))
      return false;
  }

  return true;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------// ---------------------------------------------------------------------------
// AddToModelBuilderImpl
// ---------------------------------------------------------------------------
NeuronStatus MatMulOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                      const onnx::NodeProto& node,
                                                      const NeuronGraphContext& ctx) const {
  if (op_builder_helpers::IsSupportedBatchMatMul(node, ctx, model_builder.GetNeuronVersion())) {
    LOGS_NEURON(VERBOSE) << "Use NEURON_BATCH_MATMUL to support MatMul";
    return op_builder_helpers::BuildBatchMatMul(model_builder, node, ctx);
  }
  LOGS_NEURON(VERBOSE) << "BatchMatMul not supported. Use FullyConnected to support 2D MatMul";

  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  const bool is_quant_matmul = IsQuantizedOp(node, ctx);

  const auto& input1 = node.input(0);                                          // A
  const auto& input2 = is_quant_matmul ? node.input(3) : node.input(1);        // B
  const auto& output = node.output(0);                                          // Y

  float a_scale = 0.0f, b_scale = 0.0f, y_scale = 0.0f;
  int32_t a_zero_point = 0, b_zero_point = 0, y_zero_point = 0;
  bool is_per_tensor_u8s8 = false;
  Type onnx_mat_b_type = Type::TENSOR_FLOAT32;

  if (is_quant_matmul) {
    std::optional<std::vector<float>> w_scales;
    NEURON_RETURN_IF_ERROR(
        GetConvMatMulOpQuantizationScaleAndZeroPoint(
            model_builder, ctx, node,
            a_scale, b_scale, y_scale,
            a_zero_point, b_zero_point, y_zero_point,
            w_scales, is_per_tensor_u8s8));
    onnx_mat_b_type = Type::TENSOR_QUANT8_ASYMM;
  }

  const auto* mat_b_tensor_ptr = ctx.GetInitializer(input2);
  NEURON_RETURN_IF_NOT(mat_b_tensor_ptr != nullptr,
                        "MatMul: B (", input2, ") must be a constant initializer");
  const auto& mat_b_tensor = *mat_b_tensor_ptr;

  Shape onnx_mat_b_shape;
  for (auto dim : mat_b_tensor.dims())
    onnx_mat_b_shape.push_back(static_cast<uint32_t>(dim));

  const OperandType onnx_mat_b_operand_type(onnx_mat_b_type, onnx_mat_b_shape, b_scale, b_zero_point);
  NEURON_RETURN_IF_ERROR(AddInitializerTransposed(model_builder, onnx_mat_b_operand_type, input2, is_per_tensor_u8s8));

  uint32_t input_2_idx = operand_indices.at(input2);

  if (is_quant_matmul) {
    NEURON_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input1, a_scale, a_zero_point));
    NEURON_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input2, b_scale, b_zero_point));
  }

  LOGS_NEURON(VERBOSE) << "Use FullyConnected with zero bias for MatMul node: " << node.name();
  std::string bias = model_builder.GetUniqueName(node.name() + node.op_type() + "_bias");
  const auto& bias_type = operand_types.at(input2).type;
  const Shape& bias_dimen = {shaper[input2][0]};
  if (bias_type == Type::TENSOR_FLOAT32) {
    std::vector<float> buffer(bias_dimen[0], 0.f);
    OperandType bias_operand_type(Type::TENSOR_FLOAT32, bias_dimen);
    NEURON_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(bias, buffer.data(), bias_operand_type));
  } else if (bias_type == Type::TENSOR_QUANT8_ASYMM) {
    std::vector<int32_t> buffer(bias_dimen[0], 0);
    OperandType bias_operand_type(Type::TENSOR_INT32, bias_dimen, a_scale * b_scale, 0);
    NEURON_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(bias, buffer.data(), bias_operand_type));
  } else {
    return NeuronStatus{false, MakeNeuronString("MatMulOpBuilder: unknown weight type ", TypeToStr(bias_type))};
  }

  uint32_t bias_idx = operand_indices.at(bias);

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input1));  // A
  input_indices.push_back(input_2_idx);                 // B
  input_indices.push_back(bias_idx);                    // zero bias
  int32_t fuse_code = model_builder.FindActivation(node, ctx);
  ADD_SCALAR_OPERAND(model_builder, input_indices, fuse_code);

  const OperandType output_operand_type(operand_types.at(input1).type, shaper[output], y_scale, y_zero_point);
  NEURON_RETURN_IF_ERROR(model_builder.AddOperation(NEURON_FULLY_CONNECTED, input_indices,
                                                     {output}, {output_operand_type}));
  return NeuronStatus::OK();
}

void CreateMatMulOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  CreateSharedOpBuilderImpl<MatMulOpBuilder>(
      op_type, op_registrations,
      {
          "MatMul",
          "QLinearMatMul",
      });
}

}  // namespace neuron
}  // namespace onnxruntime
