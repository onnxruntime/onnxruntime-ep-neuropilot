// expand_op_builder.cc
// Plugin EP version – uses onnx::NodeProto + NeuronGraphContext.
//
// Ported from in-tree expand_op_builder.cc.
//
// ONNX Expand mapping to Neuron backend.
//
// Key idea:
// Neuron may not provide a dedicated "Expand/BroadcastTo" operator.  We
// implement Expand by using NEURON_TILE with a multiplier derived from the
// target (expanded) shape divided by the input shape along each dimension.
//
// Translation notes:
//   node_unit.Inputs()[i].node_arg.Name()       → node.input(i)
//   node_unit.Outputs()[i].node_arg.Name()       → node.output(i)
//   node_unit.OpType()                           → node.op_type()
//   node_unit.Name()                             → node.name()
//   graph_viewer.GetConstantInitializer(name)    → ctx.GetInitializer(name)
//   Initializer / DataAsSpan                     → NeuronGraphContext::GetInt64Data / GetInt32Data
//   ORT_RETURN_IF_ERROR / ORT_RETURN_IF_NOT      → NEURON_RETURN_IF_ERROR / NEURON_RETURN_IF_NOT
//   ORT_MAKE_STATUS(ONNXRUNTIME, ...)            → NeuronStatus{false, MakeNeuronString(...)}
//   Status::OK()                                 → NeuronStatus::OK()

#include "base_op_builder.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include <onnx/onnx_pb.h>

#include "../op_builder_helpers.h"

namespace onnxruntime {
namespace neuron {
using namespace onnxruntime::neuron::op_builder_helpers;
using namespace onnxruntime::neuron::wrapper;

class ExpandOpBuilder : public BaseOpBuilder {
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

  bool IsQuantizedOp(const onnx::NodeProto& /*node*/,
                     const NeuronGraphContext& /*ctx*/) const override { return false; }

  int GetMinSupportedOpSet(const onnx::NodeProto& /*node*/) const override { return 8; }

  // Helper to validate broadcastability of input→output shapes.
  static bool IsBroadcastable(const Shape& in, const Shape& out) {
    if (in.size() > out.size()) return false;
    size_t in_rank  = in.size();
    size_t out_rank = out.size();
    for (size_t i = 0; i < out_rank; ++i) {
      uint32_t out_dim = out[out_rank - 1 - i];
      uint32_t in_dim  = (i < in_rank) ? in[in_rank - 1 - i] : 1U;
      if (out_dim == 0U || in_dim == 0U) return false;
      if (!(in_dim == out_dim || in_dim == 1U || out_dim == 1U)) return false;
    }
    return true;
  }
};

// ---------------------------------------------------------------------------
// AddInitializersToSkip
// ---------------------------------------------------------------------------
void ExpandOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder,
                                              const onnx::NodeProto& node,
                                              const NeuronGraphContext& ctx) const {
  // Skip the "shape" input (index 1) — we consume it to build the tile multiplier.
  if (node.input_size() < 2) return;
  const auto& shape_name = node.input(1);
  if (!shape_name.empty() && ctx.IsInitializer(shape_name))
    model_builder.AddInitializerToSkip(shape_name);
}

// ---------------------------------------------------------------------------
// AddToModelBuilderImpl
// ---------------------------------------------------------------------------
NeuronStatus ExpandOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                      const onnx::NodeProto& node,
                                                      const NeuronGraphContext& ctx) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  NEURON_RETURN_IF_NOT(node.input_size() == 2, "Expand expects 2 inputs: data and shape.");

  const auto& input       = node.input(0);
  const auto& shape_input = node.input(1);
  const auto& output      = node.output(0);

  NEURON_RETURN_IF_NOT(operand_indices.count(input) > 0,
                        "Expand: input not registered: ", input);

  const Shape& out_shape = shaper[output];
  const Shape& in_shape  = shaper[input];

  NEURON_RETURN_IF_NOT(IsBroadcastable(in_shape, out_shape),
                        "Expand: input shape ", Shape2String(in_shape),
                        " is not broadcastable to output shape ", Shape2String(out_shape));

  // The "shape" input must be a constant initializer.
  const auto* shape_tensor_proto = ctx.GetInitializer(shape_input);
  NEURON_RETURN_IF_NOT(shape_tensor_proto != nullptr,
                        "Expand requires 'shape' input to be a constant initializer.");

  if (shape_tensor_proto->dims_size() > 1) {
    return NeuronStatus{false, MakeNeuronString("Expand: shape input must be 1-D.")};
  }

  const int32_t shape_dtype = shape_tensor_proto->data_type();
  if (shape_dtype != onnx::TensorProto::INT64 && shape_dtype != onnx::TensorProto::INT32) {
    return NeuronStatus{false, MakeNeuronString("Expand: shape input must be int64 or int32.")};
  }

  std::vector<int64_t> shape_data;
  if (shape_dtype == onnx::TensorProto::INT64) {
    std::vector<int64_t> raw;
    NEURON_RETURN_IF_ERROR(NeuronGraphContext::GetInt64Data(*shape_tensor_proto, raw));
    for (auto v : raw) {
      if (v < 0)
        return NeuronStatus{false, MakeNeuronString("Expand: negative dimension in 'shape' is not allowed.")};
      shape_data.push_back(v);
    }
  } else {
    std::vector<int32_t> raw;
    NEURON_RETURN_IF_ERROR(NeuronGraphContext::GetInt32Data(*shape_tensor_proto, raw));
    for (auto v : raw) {
      if (v < 0)
        return NeuronStatus{false, MakeNeuronString("Expand: negative dimension in 'shape' is not allowed.")};
      shape_data.push_back(static_cast<int64_t>(v));
    }
  }

  // Compute tile multipliers: multiplier[i] = shape_data[i] / in_dim_i.
  // in_shape is right-aligned relative to shape_data (ONNX broadcasting semantics).
  NEURON_RETURN_IF_NOT(in_shape.size() <= shape_data.size(),
                        "Expand: input rank (", in_shape.size(),
                        ") > shape rank (", shape_data.size(), ")");

  std::vector<int32_t> multiplier_data;
  multiplier_data.reserve(shape_data.size());
  const size_t rank_diff = shape_data.size() - in_shape.size();
  for (size_t i = 0; i < shape_data.size(); ++i) {
    // Prepend virtual 1s for the leading dimensions missing from in_shape.
    int64_t in_dim    = (i >= rank_diff) ? static_cast<int64_t>(in_shape[i - rank_diff]) : 1LL;
    int64_t shape_dim = shape_data[i];
    if (in_dim == 0 || shape_dim % in_dim != 0) {
      return NeuronStatus{false,
          MakeNeuronString("Expand: shape_dim (", shape_dim, ") must be a multiple of in_dim (", in_dim, ") at index ", i)};
    }
    multiplier_data.push_back(static_cast<int32_t>(shape_dim / in_dim));
    LOGS_NEURON(VERBOSE) << "Expand tile multiplier[" << i << "] = " << multiplier_data.back();
  }

  // The multiplier tensor has the same rank as shape_data.
  const Shape multiplier_shape = {static_cast<uint32_t>(multiplier_data.size())};
  const auto& in_operand_type = operand_types.at(input);
  const float y_scale      = in_operand_type.operandType.scale;
  const int32_t y_zero_pt  = in_operand_type.operandType.zeroPoint;

  std::string multiplier_name = model_builder.GetUniqueName(shape_input + "_tile_multiplier");
  OperandType multiplier_operand_type(Type::TENSOR_INT32, multiplier_shape);
  NEURON_RETURN_IF_ERROR(
      model_builder.AddOperandFromPersistMemoryBuffer(multiplier_name, multiplier_data.data(),
                                                       multiplier_operand_type));

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  input_indices.push_back(model_builder.GetOperandIndices().at(multiplier_name));

  const OperandType output_operand_type(in_operand_type.type, out_shape, y_scale, y_zero_pt);
  NEURON_RETURN_IF_ERROR(model_builder.AddOperation(NEURON_TILE, input_indices,
                                                     {output}, {output_operand_type}));
  return NeuronStatus::OK();
}

// ---------------------------------------------------------------------------
// HasSupportedInputOutputsImpl
// ---------------------------------------------------------------------------
bool ExpandOpBuilder::HasSupportedInputOutputsImpl(const onnx::NodeProto& node,
                                                     const NeuronGraphContext& ctx,
                                                     const OpSupportCheckParams& /*params*/) const {
  // data (input 0): float32, float16, or quantized uint8/int8
  const bool data_float = IsTypeSupported(node, ctx, 0, /*is_input*/ true,
                                          {onnx::TensorProto::FLOAT,
                                           onnx::TensorProto::FLOAT16});
  const bool data_quant = IsQuantizedTypeSupported(node, ctx, 0, /*is_input*/ true,
                                                    {{onnx::TensorProto::UINT8, false},
                                                     {onnx::TensorProto::INT8, false}});
  if (!data_float && !data_quant) return false;

  // shape (input 1): int64 or int32
  if (!IsTypeSupported(node, ctx, 1, /*is_input*/ true,
                       {onnx::TensorProto::INT64, onnx::TensorProto::INT32})) {
    return false;
  }

  // output: same types as input 0
  const bool out_float = IsTypeSupported(node, ctx, 0, /*is_input*/ false,
                                         {onnx::TensorProto::FLOAT,
                                          onnx::TensorProto::FLOAT16});
  const bool out_quant = IsQuantizedTypeSupported(node, ctx, 0, /*is_input*/ false,
                                                   {{onnx::TensorProto::UINT8, false},
                                                    {onnx::TensorProto::INT8, false}});
  if (!out_float && !out_quant) return false;

  return true;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
void CreateExpandOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ExpandOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace neuron
}  // namespace onnxruntime
