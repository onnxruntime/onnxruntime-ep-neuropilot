// slice_op_builder.cc
// Plugin EP version – uses onnx::NodeProto + NeuronGraphContext.
//
// CHANGES FROM ORIGINAL:
//   node_unit.Inputs()[i].node_arg.Name() → node.input(i)
//   node_unit.Outputs()[i].node_arg.Name() → node.output(i)
//   model_builder.GetInitializerTensors().at(name) → *ctx.GetInitializer(name)
//   Initializer unpacked_tensor / SliceOp::PrepareForCompute →
//       inline implementation using NeuronGraphContext::GetInt64Data
//   graph_viewer.GetConstantInitializer(name) → ctx.IsInitializer(name)
//   CheckIsConstantInitializer → ctx.IsInitializer(name) + log
//   ORT_RETURN_IF_ERROR → NEURON_RETURN_IF_ERROR
//   ORT_MAKE_STATUS → NEURON_MAKE_STATUS
//   Status → NeuronStatus
//   Minimum supported opset: 10

#include "base_op_builder.h"
#include "../op_builder_helpers.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

namespace onnxruntime {
namespace neuron {
using namespace onnxruntime::neuron::op_builder_helpers;
using namespace onnxruntime::neuron::wrapper;

// ---------------------------------------------------------------------------
// Minimal inline replacement for SliceOp::PrepareForCompute.
// Normalises starts/ends/steps and computes the output shape.
// ---------------------------------------------------------------------------
namespace {

struct SliceComputeMetadata {
  std::vector<int64_t> starts_;
  std::vector<int64_t> ends_;
  std::vector<int64_t> steps_;
  std::vector<int64_t> output_dims_;
};

// Clamp idx into [0, dim] for positive step, or [-1, dim-1] for negative step.
inline int64_t ClampIndex(int64_t idx, int64_t dim, bool positive_step) {
  if (positive_step) {
    return std::max<int64_t>(0, std::min<int64_t>(idx, dim));
  } else {
    return std::max<int64_t>(-1, std::min<int64_t>(idx, dim - 1));
  }
}

NeuronStatus PrepareSlice(const std::vector<int64_t>& input_dims,
                            const std::vector<int64_t>& raw_starts,
                            const std::vector<int64_t>& raw_ends,
                            const std::vector<int64_t>& raw_axes,
                            const std::vector<int64_t>& raw_steps,
                            SliceComputeMetadata& meta) {
  const int64_t rank = static_cast<int64_t>(input_dims.size());

  // Build per-axis steps, starts, ends (defaults: step=1, start=0, end=dim).
  std::vector<int64_t> steps(rank, 1);
  std::vector<int64_t> starts(rank, 0);
  std::vector<int64_t> ends = input_dims;

  // Map axes: default is 0..N-1 if raw_axes is empty.
  std::vector<int64_t> axes;
  if (raw_axes.empty()) {
    axes.resize(raw_starts.size());
    std::iota(axes.begin(), axes.end(), 0);
  } else {
    axes = raw_axes;
    for (auto& a : axes) {
      if (a < 0) a += rank;
    }
  }

  for (size_t i = 0; i < axes.size(); ++i) {
    int64_t ax = axes[i];
    int64_t dim = input_dims[ax];
    int64_t step = raw_steps.empty() ? 1 : raw_steps[i];
    if (step == 0) return NeuronStatus{false, "Slice step cannot be 0"};

    int64_t start = raw_starts[i];
    int64_t end   = raw_ends[i];

    // Normalise negative indices.
    if (start < 0) start += dim;
    if (end < 0)   end   += dim;

    bool fwd = (step > 0);
    start = ClampIndex(start, dim, fwd);
    end   = ClampIndex(end,   dim, fwd);

    steps[ax]  = step;
    starts[ax] = start;
    ends[ax]   = end;
  }

  // Compute output dims.
  std::vector<int64_t> out_dims(rank);
  for (int64_t d = 0; d < rank; ++d) {
    int64_t step  = steps[d];
    int64_t start = starts[d];
    int64_t end   = ends[d];
    int64_t span  = (step > 0) ? (end - start) : (start - end);
    if (span <= 0) span = 0;
    out_dims[d] = (span + std::abs(step) - 1) / std::abs(step);
  }

  meta.starts_      = std::move(starts);
  meta.ends_        = std::move(ends);
  meta.steps_       = std::move(steps);
  meta.output_dims_ = std::move(out_dims);
  return NeuronStatus::OK();
}

}  // namespace

// ---------------------------------------------------------------------------

class SliceOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder,
                              const onnx::NodeProto& node,
                              const NeuronGraphContext& /*ctx*/) const override;

 private:
  // Only support Slice from opset 10 (inputs-based, not attributes-based).
  int GetMinSupportedOpSet(const onnx::NodeProto& /*node*/) const override { return 10; }

  NeuronStatus AddToModelBuilderImpl(ModelBuilder& model_builder,
                                      const onnx::NodeProto& node,
                                      const NeuronGraphContext& ctx) const override;

};

void SliceOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder,
                                             const onnx::NodeProto& node,
                                             const NeuronGraphContext& /*ctx*/) const {
  // Skip starts, ends, axes, steps — all consumed inline.
  model_builder.AddInitializerToSkip(node.input(1));  // starts
  model_builder.AddInitializerToSkip(node.input(2));  // ends
  if (node.input_size() > 3 && !node.input(3).empty())
    model_builder.AddInitializerToSkip(node.input(3));  // axes
  if (node.input_size() > 4 && !node.input(4).empty())
    model_builder.AddInitializerToSkip(node.input(4));  // steps
}

NeuronStatus SliceOpBuilder::AddToModelBuilderImpl(
    ModelBuilder& model_builder,
    const onnx::NodeProto& node,
    const NeuronGraphContext& ctx) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  const auto& input  = node.input(0);
  const auto& output = node.output(0);

  // Read the input shape.
  const auto input_shape_u32 = shaper[input];
  std::vector<int64_t> input_shape_64(input_shape_u32.cbegin(), input_shape_u32.cend());

  // Helper to read int64 data from an initializer by input index.
  // Handles both INT64 and INT32 initializers (INT32 is widened to INT64).
  const auto ReadInitializerInt64 = [&](int idx, std::vector<int64_t>& data) -> NeuronStatus {
    data.clear();
    if (node.input_size() <= idx || node.input(idx).empty()) return NeuronStatus::OK();
    const auto* tensor = ctx.GetInitializer(node.input(idx));
    if (!tensor) return NeuronStatus::OK();
    if (tensor->data_type() == onnx::TensorProto::INT64) {
      NEURON_RETURN_IF_ERROR(NeuronGraphContext::GetInt64Data(*tensor, data));
    } else if (tensor->data_type() == onnx::TensorProto::INT32) {
      std::vector<int32_t> data32;
      NEURON_RETURN_IF_ERROR(NeuronGraphContext::GetInt32Data(*tensor, data32));
      data.assign(data32.begin(), data32.end());
    } else {
      return NeuronStatus{false,
          MakeNeuronString("Slice: unsupported data type for starts/ends/axes/steps: ",
                           tensor->data_type())};
    }
    return NeuronStatus::OK();
  };

  std::vector<int64_t> input_starts, input_ends, input_axes, input_steps;
  NEURON_RETURN_IF_ERROR(ReadInitializerInt64(1, input_starts));
  NEURON_RETURN_IF_ERROR(ReadInitializerInt64(2, input_ends));
  NEURON_RETURN_IF_ERROR(ReadInitializerInt64(3, input_axes));
  NEURON_RETURN_IF_ERROR(ReadInitializerInt64(4, input_steps));

  SliceComputeMetadata meta;
  NEURON_RETURN_IF_ERROR(PrepareSlice(input_shape_64, input_starts, input_ends,
                                       input_axes, input_steps, meta));

  // Convert output dims to uint32.
  Shape neuron_output_shape;
  neuron_output_shape.reserve(meta.output_dims_.size());
  for (int64_t d : meta.output_dims_)
    neuron_output_shape.push_back(static_cast<uint32_t>(d));

  shaper.AddShape(output, neuron_output_shape);
  // Copy the full operand type (scale, ZP, per-channel params) from the input —
  // Slice is a pass-through for quantized tensors; Neuron rejects it if the
  // output has a different scale/ZP than the input ("requant unsupported").
  OperandType output_operand_type = operand_types.at(input);
  output_operand_type.SetDimensions(shaper[output]);

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));

  const Shape param_dimen = {static_cast<uint32_t>(input_shape_u32.size())};

  // Helper to add a param tensor (begins/ends/sizes/strides).
  const auto AddOperand = [&](const char* tag, const Shape& shape,
                               const std::vector<int64_t>& param_raw) -> NeuronStatus {
    std::vector<int32_t> param_data;
    param_data.reserve(param_raw.size());
    for (int64_t v : param_raw) param_data.push_back(static_cast<int32_t>(v));
    const std::string param_name = model_builder.GetUniqueName(std::string(node.name()) + tag);
    const OperandType param_operand_type(Type::TENSOR_INT32, shape);
    NEURON_RETURN_IF_ERROR(
        model_builder.AddOperandFromPersistMemoryBuffer(param_name, param_data.data(),
                                                        param_operand_type));
    input_indices.push_back(operand_indices.at(param_name));
    return NeuronStatus::OK();
  };

  NEURON_RETURN_IF_ERROR(AddOperand("starts", param_dimen, meta.starts_));

  // Choose between NEURON_SLICE (all steps == 1) and NEURON_STRIDED_SLICE.
  const bool all_steps_one = std::all_of(meta.steps_.cbegin(), meta.steps_.cend(),
                                          [](int64_t s) { return s == 1; });
  int op_code = all_steps_one ? NEURON_SLICE : NEURON_STRIDED_SLICE;

  if (all_steps_one) {
    // NEURON_SLICE: begin + sizes.
    NEURON_RETURN_IF_ERROR(AddOperand("sizes", param_dimen, meta.output_dims_));
  } else {
    // NEURON_STRIDED_SLICE: begin + ends + strides + masks.
    // Special-case end == -1: map to -(dim + 1) to avoid NNAPI treating it as a
    // count-from-end index.
    auto ends = meta.ends_;
    for (size_t i = 0; i < ends.size(); ++i) {
      if (ends[i] == -1) {
        ends[i] = -(static_cast<int64_t>(input_shape_u32[i]) + 1);
      }
    }
    NEURON_RETURN_IF_ERROR(AddOperand("ends",    param_dimen, ends));
    NEURON_RETURN_IF_ERROR(AddOperand("steps",   param_dimen, meta.steps_));
    ADD_SCALAR_OPERAND(model_builder, input_indices, 0);  // begin_mask
    ADD_SCALAR_OPERAND(model_builder, input_indices, 0);  // end_mask
    ADD_SCALAR_OPERAND(model_builder, input_indices, 0);  // shrink_axis_mask
  }

  return model_builder.AddOperation(op_code, input_indices, {output}, {output_operand_type});
}
void CreateSliceOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<SliceOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace neuron
}  // namespace onnxruntime
