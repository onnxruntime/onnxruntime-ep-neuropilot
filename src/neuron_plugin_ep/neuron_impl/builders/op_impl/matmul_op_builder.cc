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

// Resolves a MatMul's B edge to a constant weight initializer name: either a
// direct initializer or one wrapped in an (absorbed) DequantizeLinear.
// Returns an empty string when B is a runtime tensor.
static std::string ResolveConstantMatMulB(const onnx::NodeProto& node,
                                          const NeuronGraphContext& ctx) {
  if (node.op_type() != "MatMul" || node.input_size() < 2) return {};
  const std::string& b_edge = node.input(1);
  if (ctx.GetInitializer(b_edge)) return b_edge;
  if (const auto* dq = ctx.GetProducer(b_edge)) {
    if (dq->op_type() == "DequantizeLinear" && dq->input_size() > 0 &&
        ctx.GetInitializer(dq->input(0))) {
      return dq->input(0);
    }
  }
  return {};
}

// A MatMul with a constant rank-2 B and a rank-2 (or rank-3, batch-1) A lowers
// to NEURON_FULLY_CONNECTED.  This is not an optimisation but a correctness
// requirement: lowering this class as NEURON_BATCH_MATMUL with a rank-3
// expanded constant B produces garbage on MDLA (measured on the Genio 720 EVK:
// cosine ~0.000 against the CPU EP for [S,2048]x[2048,N] at N=512, 2048 and
// 64128, while the same driver computes runtime-B BATCH_MATMUL correctly).
static bool ShouldLowerAsFullyConnected(const onnx::NodeProto& node,
                                        const NeuronGraphContext& ctx) {
  const std::string b_name = ResolveConstantMatMulB(node, ctx);
  if (b_name.empty()) return false;
  const auto* b_init = ctx.GetInitializer(b_name);
  if (b_init == nullptr || b_init->dims_size() != 2) return false;
  const Shape a_shape = ctx.GetShape(node.input(0));
  return a_shape.size() == 2 || (a_shape.size() == 3 && a_shape[0] == 1);
}

class MatMulOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder,
                              const onnx::NodeProto& node,
                              const NeuronGraphContext& ctx) const override;

 private:
  NeuronStatus AddToModelBuilderImpl(ModelBuilder& model_builder,
                                      const onnx::NodeProto& node,
                                      const NeuronGraphContext& ctx) const override;

  NeuronStatus BuildConstBFullyConnected(ModelBuilder& model_builder,
                                          const onnx::NodeProto& node,
                                          const NeuronGraphContext& ctx) const;

  bool HasSupportedInputsImpl(const onnx::NodeProto& node,
                                     const NeuronGraphContext& ctx,
                                     const OpSupportCheckParams& params) const override;

  bool IsNodeUnitTypeSupported(const onnx::NodeProto& /*node*/) const override { return true; }

  bool IsQuantizedOp(const onnx::NodeProto& node,
                     const NeuronGraphContext& ctx) const override {
    return IsQuantizedMatMul(GetQuantizedOpType(node, ctx));
  }

  bool IsComputeHeavy(const onnx::NodeProto& /*node*/) const override { return true; }
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

  // The const-B FULLY_CONNECTED route adds the weight itself (transposed);
  // keep RegisterInitializers from adding an untransposed duplicate.
  if (node.op_type() == "MatMul" && ShouldLowerAsFullyConnected(node, ctx)) {
    model_builder.AddInitializerToSkip(ResolveConstantMatMulB(node, ctx));
  }
}

// ---------------------------------------------------------------------------
// HasSupportedInputsImpl
// ---------------------------------------------------------------------------
bool MatMulOpBuilder::HasSupportedInputsImpl(const onnx::NodeProto& node,
                                               const NeuronGraphContext& ctx,
                                               const OpSupportCheckParams& params) const {
  // For QLinearMatMul, the first tensor input is at index 0.
  std::unordered_set<int32_t> supported_types{onnx::TensorProto::UINT8,
                                              onnx::TensorProto::INT8,
                                              onnx::TensorProto::INT16,
                                              onnx::TensorProto::UINT16};

  if (params.use_fp16 || IsQuantizedOp(node, ctx)) {
    supported_types.insert(onnx::TensorProto::FLOAT);
  }

  if (!IsTypeSupported(node, ctx, 0, /*is_input*/ true, supported_types)) {
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
  if (ShouldLowerAsFullyConnected(node, ctx)) {
    LOGS_NEURON(VERBOSE) << "Const-B MatMul: lowering as NEURON_FULLY_CONNECTED";
    return BuildConstBFullyConnected(model_builder, node, ctx);
  }
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

// ---------------------------------------------------------------------------
// BuildConstBFullyConnected — MatMul (QDQ or float) with a constant rank-2 B.
// Modeled on the Gemm builder's QDQ path; wraps the FC in reshapes when A is
// rank-3 batch-1 so the FC sees the 2-D layout MDLA is built for.
// ---------------------------------------------------------------------------
NeuronStatus MatMulOpBuilder::BuildConstBFullyConnected(
    ModelBuilder& model_builder, const onnx::NodeProto& node,
    const NeuronGraphContext& ctx) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  const bool is_quant = IsQuantizedOp(node, ctx);
  const std::string& input1 = node.input(0);
  const std::string& output = node.output(0);
  const std::string actual_b_name = ResolveConstantMatMulB(node, ctx);

  float a_scale = 0.0f, b_scale = 0.0f, y_scale = 0.0f;
  int32_t a_zero_point = 0, b_zero_point = 0, y_zero_point = 0;
  Type b_type = Type::TENSOR_FLOAT32;
  Type out_type = operand_types.at(input1).type;
  const QDQNodeGroup* qdq = is_quant ? ctx.GetQDQGroup(output) : nullptr;
  if (is_quant) {
    NEURON_RETURN_IF_NOT(qdq != nullptr,
                          "QDQMatMul: no QDQ group found for output '", output, "'");
    if (qdq->input_quant_params.size() > 0 && qdq->input_quant_params[0].has_value()) {
      a_scale      = qdq->input_quant_params[0]->scale;
      a_zero_point = qdq->input_quant_params[0]->zero_point;
    }
    NEURON_RETURN_IF_NOT(qdq->input_quant_params.size() > 1 &&
                              qdq->input_quant_params[1].has_value(),
                          "QDQMatMul: B has no quant params");
    const auto& bqp = *qdq->input_quant_params[1];
    if (bqp.IsPerChannel()) {
      // Per-channel weights: ONNX quantizes [K,N] along axis 1 (the output
      // channel); after the FC transpose to [N,K] that is channelDim 0, which
      // is what NEURON_TENSOR_QUANT8_SYMM_PER_CHANNEL expects.  Symmetric only.
      NEURON_RETURN_IF_NOT(bqp.axis == 1,
                            "QDQMatMul: per-channel B axis must be 1, got ", bqp.axis);
      for (const int32_t zp : bqp.per_channel_zero_points) {
        NEURON_RETURN_IF_NOT(zp == 0,
                              "QDQMatMul: per-channel B must be symmetric (zp 0)");
      }
      b_type = Type::TENSOR_QUANT8_SYMM_PER_CHANNEL;
    } else {
      b_scale      = bqp.scale;
      b_zero_point = bqp.zero_point;
      b_type = bqp.IsSigned() ? Type::TENSOR_QUANT8_ASYMM_SIGNED : Type::TENSOR_QUANT8_ASYMM;
    }
    if (qdq->output_quant_params.has_value()) {
      y_scale      = qdq->output_quant_params->scale;
      y_zero_point = qdq->output_quant_params->zero_point;
      out_type = qdq->output_quant_params->IsSigned() ? Type::TENSOR_QUANT8_ASYMM_SIGNED
                                                       : Type::TENSOR_QUANT8_ASYMM;
    }
  }

  // B, transposed to FC's [units, input_size] layout.
  const auto& mat_b_tensor = *ctx.GetInitializer(actual_b_name);
  Shape onnx_b_shape;
  for (auto dim : mat_b_tensor.dims())
    onnx_b_shape.push_back(static_cast<uint32_t>(dim));
  const uint32_t k_dim = onnx_b_shape[0];
  const uint32_t n_dim = onnx_b_shape[1];
  const bool b_per_channel = b_type == Type::TENSOR_QUANT8_SYMM_PER_CHANNEL;
  if (b_per_channel) {
    SymmPerChannelQuantParams pc_params(qdq->input_quant_params[1]->per_channel_scales,
                                        /*channelDim=*/0);
    const OperandType onnx_b_operand_type(b_type, onnx_b_shape, std::move(pc_params));
    NEURON_RETURN_IF_ERROR(AddInitializerTransposed(model_builder, onnx_b_operand_type,
                                                    actual_b_name, /*is_per_tensor_u8s8=*/false));
  } else {
    const OperandType onnx_b_operand_type(b_type, onnx_b_shape, b_scale, b_zero_point);
    NEURON_RETURN_IF_ERROR(AddInitializerTransposed(model_builder, onnx_b_operand_type,
                                                    actual_b_name, /*is_per_tensor_u8s8=*/false));
  }

  // A: FC wants [batch, input_size]; unwrap a rank-3 batch-1 A.
  const Shape a_shape = ctx.GetShape(input1);
  const bool need_reshape = a_shape.size() == 3;
  std::string fc_in = input1;
  if (need_reshape) {
    fc_in = model_builder.GetUniqueName(node.name() + "/fc_in");
    NEURON_RETURN_IF_ERROR(model_builder.AddReshape(input1, {a_shape[1], k_dim}, fc_in));
  }
  const uint32_t rows = need_reshape ? a_shape[1] : a_shape[0];

  // Zero bias in the type FC requires.
  const std::string bias = model_builder.GetUniqueName(node.name() + node.op_type() + "_bias");
  if (b_type == Type::TENSOR_FLOAT32) {
    std::vector<float> buffer(n_dim, 0.f);
    NEURON_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(
        bias, buffer.data(), OperandType(Type::TENSOR_FLOAT32, Shape{n_dim})));
  } else {
    // Per NNAPI, a per-channel filter takes an INT32 bias with scale 0 (the
    // driver derives input_scale * filter_scale[i]); per-tensor uses the
    // explicit product scale.
    std::vector<int32_t> buffer(n_dim, 0);
    const float bias_scale = b_per_channel ? 0.0f : a_scale * b_scale;
    NEURON_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(
        bias, buffer.data(),
        OperandType(Type::TENSOR_INT32, Shape{n_dim}, bias_scale, 0)));
  }

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(fc_in));
  input_indices.push_back(operand_indices.at(actual_b_name));
  input_indices.push_back(operand_indices.at(bias));
  int32_t fuse_code = model_builder.FindActivation(node, ctx);
  ADD_SCALAR_OPERAND(model_builder, input_indices, fuse_code);

  const Shape fc_out_shape{rows, n_dim};
  if (!need_reshape) {
    const OperandType out_operand(out_type, fc_out_shape, y_scale, y_zero_point);
    return model_builder.AddOperation(NEURON_FULLY_CONNECTED, input_indices,
                                      {output}, {out_operand});
  }
  const std::string fc_out = model_builder.GetUniqueName(node.name() + "/fc_out");
  shaper.AddShape(fc_out, fc_out_shape);
  const OperandType fc_out_operand(out_type, fc_out_shape, y_scale, y_zero_point);
  NEURON_RETURN_IF_ERROR(model_builder.AddOperation(NEURON_FULLY_CONNECTED, input_indices,
                                                    {fc_out}, {fc_out_operand}));
  return model_builder.AddReshape(fc_out, {1, a_shape[1], n_dim}, output);
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
