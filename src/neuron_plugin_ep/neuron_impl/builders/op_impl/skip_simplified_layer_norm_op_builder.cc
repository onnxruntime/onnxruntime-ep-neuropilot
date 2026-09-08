// Copyright (c) MediaTek Inc. All rights reserved.
// Licensed under the MIT License.

// Builder for SkipSimplifiedLayerNormalization and SimplifiedLayerNormalization
// (RMSNorm family used in Qwen2 / LLaMA-style transformers).
// Standalone plugin EP version — uses onnx::NodeProto + NeuronGraphContext.
//
// Both ops implement RMSNorm:
//   SimplifiedLayerNorm:
//     output = input * gamma / rms(input)       where rms(x) = sqrt(mean(x²) + eps)
//
//   SkipSimplifiedLayerNorm:
//     sum    = input + skip
//     output = sum * gamma / rms(sum)
//
// Decomposition into primitive Neuron ops:
//   SkipSimplifiedLayerNorm only:
//     1. NEURON_ADD   (input, skip)          → x_sum  (also emits a SECOND ADD → out3)
//
//   Both continue from x (= x_sum or input).
//   All remaining operations work on a 2-D view [tokens, hidden] to avoid
//   potential hardware limitations with NEURON_MEAN on 3-D tensors.
//
//   NOTE: NEURON_DIV has documented accuracy issues on MDLA hardware.
//   We use NEURON_RSQRT + NEURON_MUL instead of NEURON_SQRT + NEURON_DIV.
//
//     2. NEURON_RESHAPE (x → x_2d)              (only when rank > 2)
//     3. NEURON_MUL   (x_2d, x_2d)             → sq          (x²)
//     4. NEURON_MEAN  (sq, [1], keepdims=1)     → mean_sq     [tokens, 1]
//     5. NEURON_ADD   (mean_sq, eps)            → mean_sq_eps
//     6. NEURON_RSQRT (mean_sq_eps)             → inv_rms     (1/rms)
//     7. NEURON_MUL   (x_2d, inv_rms)          → norm_2d
//     8. NEURON_MUL   (norm_2d, gamma_2d)      → out_2d
//     9. NEURON_RESHAPE (out_2d → output)       (only when rank > 2)

#include "base_op_builder.h"

namespace onnxruntime {
namespace neuron {

using namespace onnxruntime::neuron::op_builder_helpers;
using namespace onnxruntime::neuron::wrapper;

class SkipSimplifiedLayerNormOpBuilder : public BaseOpBuilder {
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
                                    const OpSupportCheckParams& params) const override;

  bool IsNodeUnitTypeSupported(const onnx::NodeProto& /*node*/) const override { return true; }
};

// ---------------------------------------------------------------------------

void SkipSimplifiedLayerNormOpBuilder::AddInitializersToSkip(
    ModelBuilder& model_builder,
    const onnx::NodeProto& node,
    const NeuronGraphContext& /*ctx*/) const {
  const bool is_skip = (node.op_type() == "SkipSimplifiedLayerNormalization");
  const size_t gamma_idx = is_skip ? 2u : 1u;
  if (static_cast<size_t>(node.input_size()) > gamma_idx && !node.input(static_cast<int>(gamma_idx)).empty())
    model_builder.AddInitializerToSkip(node.input(static_cast<int>(gamma_idx)));
}

bool SkipSimplifiedLayerNormOpBuilder::HasSupportedInputsImpl(
    const onnx::NodeProto& node,
    const NeuronGraphContext& ctx,
    const OpSupportCheckParams& /*params*/) const {
  if (node.input_size() < 1) return false;
  int32_t elem_type = ctx.GetElementType(node.input(0));
  if (elem_type != onnx::TensorProto::FLOAT) {
    LOGS_NEURON(VERBOSE) << node.op_type() << ": input must be FLOAT, got " << elem_type;
    return false;
  }
  return true;
}
// ---------------------------------------------------------------------------

NeuronStatus SkipSimplifiedLayerNormOpBuilder::AddToModelBuilderImpl(
    ModelBuilder& model_builder,
    const onnx::NodeProto& node,
    const NeuronGraphContext& ctx) const {
  const auto& op_type    = node.op_type();
  const std::string& output_name = node.output(0);

  auto& shaper                = model_builder.GetShaper();
  const auto& operand_indices = model_builder.GetOperandIndices();
  const bool  is_skip         = (op_type == "SkipSimplifiedLayerNormalization");

  const float eps = NeuronGraphContext::GetAttrFloat(node, "epsilon", 1e-5f);

  const std::string& input_name = node.input(0);
  const Shape input_shape = shaper[input_name];

  // -----------------------------------------------------------------------
  // Compute 2-D working shape to avoid potential NEURON_MEAN issues on
  // 3-D (or higher-rank) tensors.
  // -----------------------------------------------------------------------
  const size_t rank = input_shape.size();
  uint32_t flat_tokens = 1u;
  for (size_t i = 0; i + 1 < rank; ++i) flat_tokens *= input_shape[i];
  const uint32_t flat_hidden = input_shape.back();
  const Shape flat2d_shape{flat_tokens, flat_hidden};
  const bool need_reshape = (rank != 2u);

  // -----------------------------------------------------------------------
  // 1. (SkipSimplified only) ADD(input, skip) → x
  // -----------------------------------------------------------------------
  std::string x_name;
  if (is_skip) {
    const std::string& skip_name = node.input(1);
    x_name = model_builder.GetUniqueName(node.name() + "_rms_x");
    shaper.AddShape(x_name, input_shape);
    {
      const OperandType x_type(Type::TENSOR_FLOAT32, input_shape);
      std::vector<uint32_t> idx;
      idx.push_back(operand_indices.at(input_name));
      idx.push_back(operand_indices.at(skip_name));
      ADD_SCALAR_OPERAND(model_builder, idx, static_cast<int32_t>(NEURON_FUSED_NONE));
      NEURON_RETURN_IF_ERROR(model_builder.AddOperation(NEURON_ADD, idx, {x_name}, {x_type}));
    }

    // output[3] = input_skip_bias_sum: separate ADD so it is a distinct operand
    if (node.output_size() > 3 && !node.output(3).empty()) {
      const std::string& out3_name = node.output(3);
      shaper.AddShape(out3_name, input_shape);
      const OperandType out3_type(Type::TENSOR_FLOAT32, input_shape);
      std::vector<uint32_t> idx;
      idx.push_back(operand_indices.at(input_name));
      idx.push_back(operand_indices.at(skip_name));
      ADD_SCALAR_OPERAND(model_builder, idx, static_cast<int32_t>(NEURON_FUSED_NONE));
      NEURON_RETURN_IF_ERROR(model_builder.AddOperation(NEURON_ADD, idx, {out3_name}, {out3_type}));
    }
  } else {
    x_name = input_name;
  }

  // -----------------------------------------------------------------------
  // 2. Optionally flatten x to 2-D
  // -----------------------------------------------------------------------
  std::string x_2d_name;
  if (need_reshape) {
    x_2d_name = model_builder.GetUniqueName(node.name() + "_rms_x2d");
    NEURON_RETURN_IF_ERROR(model_builder.AddReshape(x_name, flat2d_shape, x_2d_name));
  } else {
    x_2d_name = x_name;
  }

  // -----------------------------------------------------------------------
  // 3. MUL(x_2d, x_2d) → sq   (element-wise square, 2-D)
  // -----------------------------------------------------------------------
  const std::string sq_name = model_builder.GetUniqueName(node.name() + "_rms_sq");
  shaper.AddShape(sq_name, flat2d_shape);
  {
    const OperandType sq_type(Type::TENSOR_FLOAT32, flat2d_shape);
    std::vector<uint32_t> idx;
    idx.push_back(operand_indices.at(x_2d_name));
    idx.push_back(operand_indices.at(x_2d_name));
    ADD_SCALAR_OPERAND(model_builder, idx, static_cast<int32_t>(NEURON_FUSED_NONE));
    NEURON_RETURN_IF_ERROR(model_builder.AddOperation(NEURON_MUL, idx, {sq_name}, {sq_type}));
  }

  // -----------------------------------------------------------------------
  // 4. NEURON_MEAN(sq, axis=[1], keepdims=1) → mean_sq  [tokens, 1]
  // -----------------------------------------------------------------------
  const int32_t axis_val = 1;
  const std::string axis_name =
      model_builder.GetUniqueName(node.name() + "_rms_axis");
  {
    const Shape       axis_shape{1};
    const OperandType axis_type(Type::TENSOR_INT32, axis_shape);
    NEURON_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(
        axis_name, &axis_val, axis_type));
  }

  const Shape mean_sq_shape{flat_tokens, 1u};
  const std::string mean_sq_name =
      model_builder.GetUniqueName(node.name() + "_rms_mean_sq");
  shaper.AddShape(mean_sq_name, mean_sq_shape);
  {
    const OperandType mean_sq_type(Type::TENSOR_FLOAT32, mean_sq_shape);
    std::vector<uint32_t> idx;
    idx.push_back(operand_indices.at(sq_name));
    idx.push_back(operand_indices.at(axis_name));
    ADD_SCALAR_OPERAND(model_builder, idx, static_cast<int32_t>(1));  // keepdims=1
    NEURON_RETURN_IF_ERROR(model_builder.AddOperation(NEURON_MEAN, idx, {mean_sq_name}, {mean_sq_type}));
  }

  // -----------------------------------------------------------------------
  // 5. ADD(mean_sq, eps) → mean_sq_eps
  // -----------------------------------------------------------------------
  const std::string eps_const_name =
      model_builder.GetUniqueName(node.name() + "_rms_eps");
  {
    // The operand is [tokens, 1]: materialise a full column.  Passing &eps
    // here made the persist-buffer copy read tokens*4 bytes from a single
    // stack float, every row but the first got garbage added to its
    // mean-square.
    const std::vector<float> eps_column(flat_tokens, eps);
    const OperandType eps_type(Type::TENSOR_FLOAT32, mean_sq_shape);
    NEURON_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(
        eps_const_name, eps_column.data(), eps_type));
  }

  const std::string mean_sq_eps_name =
      model_builder.GetUniqueName(node.name() + "_rms_mean_sq_eps");
  shaper.AddShape(mean_sq_eps_name, mean_sq_shape);
  {
    const OperandType mse_type(Type::TENSOR_FLOAT32, mean_sq_shape);
    std::vector<uint32_t> idx;
    idx.push_back(operand_indices.at(mean_sq_name));
    idx.push_back(operand_indices.at(eps_const_name));
    ADD_SCALAR_OPERAND(model_builder, idx, static_cast<int32_t>(NEURON_FUSED_NONE));
    NEURON_RETURN_IF_ERROR(model_builder.AddOperation(NEURON_ADD, idx, {mean_sq_eps_name}, {mse_type}));
  }

  // -----------------------------------------------------------------------
  // 6. RSQRT(mean_sq_eps) → inv_rms
  // -----------------------------------------------------------------------
  const std::string inv_rms_name =
      model_builder.GetUniqueName(node.name() + "_rms_inv_rms");
  shaper.AddShape(inv_rms_name, mean_sq_shape);
  {
    const OperandType inv_rms_type(Type::TENSOR_FLOAT32, mean_sq_shape);
    std::vector<uint32_t> idx;
    idx.push_back(operand_indices.at(mean_sq_eps_name));
    NEURON_RETURN_IF_ERROR(model_builder.AddOperation(NEURON_RSQRT, idx, {inv_rms_name}, {inv_rms_type}));
  }

  // -----------------------------------------------------------------------
  // 7. MUL(x_2d, inv_rms) → norm_2d  [tokens, H]
  // -----------------------------------------------------------------------
  const std::string norm_name =
      model_builder.GetUniqueName(node.name() + "_rms_norm");
  shaper.AddShape(norm_name, flat2d_shape);
  {
    const OperandType norm_type(Type::TENSOR_FLOAT32, flat2d_shape);
    std::vector<uint32_t> idx;
    idx.push_back(operand_indices.at(x_2d_name));
    idx.push_back(operand_indices.at(inv_rms_name));
    ADD_SCALAR_OPERAND(model_builder, idx, static_cast<int32_t>(NEURON_FUSED_NONE));
    NEURON_RETURN_IF_ERROR(model_builder.AddOperation(NEURON_MUL, idx, {norm_name}, {norm_type}));
  }

  // -----------------------------------------------------------------------
  // 8. MUL(norm_2d, gamma_2d) → out_2d
  // -----------------------------------------------------------------------
  const size_t      gamma_idx  = is_skip ? 2u : 1u;
  const std::string gamma_orig = node.input(static_cast<int>(gamma_idx));
  const std::string gamma_name =
      model_builder.GetUniqueName(node.name() + "_rms_gamma");
  {
    const auto* gamma_proto = ctx.GetInitializer(gamma_orig);
    NEURON_RETURN_IF_NOT(gamma_proto != nullptr,
                         op_type, ": gamma initializer not found");
    std::vector<float> gamma_data;
    NEURON_RETURN_IF_ERROR(NeuronGraphContext::GetFloatData(*gamma_proto, gamma_data));

    // Get original shape from gamma initializer dims
    Shape gamma_shape;
    for (auto d : gamma_proto->dims())
      gamma_shape.push_back(static_cast<uint32_t>(d));
    if (gamma_shape.empty()) gamma_shape.push_back(static_cast<uint32_t>(gamma_data.size()));
    // Unsqueeze gamma to rank-2 [1, hidden_size] to match norm_2d rank.
    while (gamma_shape.size() < 2u)
      gamma_shape.insert(gamma_shape.begin(), 1u);
    shaper.AddShape(gamma_name, gamma_shape);
    const OperandType gamma_type(Type::TENSOR_FLOAT32, gamma_shape);
    NEURON_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(
        gamma_name, gamma_data.data(), gamma_type));
  }

  const std::string out_2d_name = need_reshape
      ? model_builder.GetUniqueName(node.name() + "_rms_out2d")
      : output_name;

  {
    if (need_reshape) shaper.AddShape(out_2d_name, flat2d_shape);
    const OperandType out_type(Type::TENSOR_FLOAT32, flat2d_shape);
    std::vector<uint32_t> idx;
    idx.push_back(operand_indices.at(norm_name));
    idx.push_back(operand_indices.at(gamma_name));
    ADD_SCALAR_OPERAND(model_builder, idx, static_cast<int32_t>(NEURON_FUSED_NONE));
    NEURON_RETURN_IF_ERROR(model_builder.AddOperation(NEURON_MUL, idx, {out_2d_name}, {out_type}));
  }

  // -----------------------------------------------------------------------
  // 9. Reshape out_2d back to the original rank (only when rank > 2).
  // -----------------------------------------------------------------------
  if (need_reshape) {
    NEURON_RETURN_IF_ERROR(model_builder.AddReshape(out_2d_name, input_shape, output_name));
  }

  return NeuronStatus::OK();
}

// ---------------------------------------------------------------------------

void CreateSkipSimplifiedLayerNormOpBuilder(const std::string& op_type,
                                            OpBuilderRegistrations& op_registrations) {
  CreateSharedOpBuilderImpl<SkipSimplifiedLayerNormOpBuilder>(
      op_type, op_registrations,
      {"SkipSimplifiedLayerNormalization", "SimplifiedLayerNormalization"});
}

}  // namespace neuron
}  // namespace onnxruntime
