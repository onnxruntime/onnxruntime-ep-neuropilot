// resize_op_builder.cc
// Plugin EP version – uses onnx::NodeProto + NeuronGraphContext.
//
// Ported from in-tree resize_op_builder.cc.
//
// Translation notes:
//   node_unit.Inputs()[i].node_arg.Name()       → node.input(i)
//   node_unit.Outputs()[i].node_arg.Name()       → node.output(i)
//   node_unit.OpType()                           → node.op_type()
//   node_unit.Name()                             → node.name()
//   node_unit.SinceVersion()                     → (opset from model; use op_capability check)
//   helper.Get("attr", val)                      → NeuronGraphContext::GetAttrXxx(node, "attr", val)
//   graph_viewer.GetConstantInitializer(name)    → ctx.GetInitializer(name)
//   ORT_RETURN_IF_ERROR / ORT_RETURN_IF_NOT      → NEURON_RETURN_IF_ERROR / NEURON_RETURN_IF_NOT
//   Status::OK()                                 → NeuronStatus::OK()
//
// Quantized path:
//   For QDQResize, x_scale and x_zp are initializers referenced by the
//   surrounding DQ/Q nodes.  In the plugin EP we retrieve them via
//   GetQuantizationScaleAndZeroPoint(ctx, scale_name, zp_name, ...).
//
//   AddInitializersToSkip:
//     - ROI (input 1), scales (input 2), sizes (input 3) are all skipped.
//     - For QDQResize: x quant params (DQ inputs) and y quant params (Q inputs)
//       are handled by the Q/DQL builders; here we only deal with the core Resize
//       node's own inputs, so no extra quant-param skips are needed at this level.

#include "base_op_builder.h"

#include <onnx/onnx_pb.h>

#include "../op_builder_helpers.h"

namespace onnxruntime {
namespace neuron {
using namespace onnxruntime::neuron::op_builder_helpers;
using namespace onnxruntime::neuron::wrapper;

class ResizeOpBuilder : public BaseOpBuilder {
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

  // Resize opset 10- is very different from opset 11+; only support 11+.
  int GetMinSupportedOpSet(const onnx::NodeProto& /*node*/) const override { return 11; }

  bool IsNodeUnitTypeSupported(const onnx::NodeProto& /*node*/) const override { return true; }

  bool IsQuantizedOp(const onnx::NodeProto& node,
                     const NeuronGraphContext& ctx) const override {
    return GetQuantizedOpType(node, ctx) == QuantizedOpType::QDQResize;
  }
};

// ---------------------------------------------------------------------------
// AddInitializersToSkip
// ---------------------------------------------------------------------------
void ResizeOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder,
                                              const onnx::NodeProto& node,
                                              const NeuronGraphContext& ctx) const {
  // We don't use ROI (input 1), scales (input 2), or sizes (input 3).
  if (node.input_size() > 1 && !node.input(1).empty())
    model_builder.AddInitializerToSkip(node.input(1));  // ROI
  if (node.input_size() > 2 && !node.input(2).empty())
    model_builder.AddInitializerToSkip(node.input(2));  // scales
  if (node.input_size() > 3 && !node.input(3).empty())
    model_builder.AddInitializerToSkip(node.input(3));  // sizes

  // For QDQResize: skip the quant params attached to x (if this is the core
  // Resize node and the DQ input carried them as sibling inputs).
  if (IsQuantizedOp(node, ctx)) {
    // QDQResize quant param layout (following ORT QDQ fusion):
    //   input(4)=x_scale, input(5)=x_zp, input(6)=y_scale, input(7)=y_zp
    if (node.input_size() > 4 && !node.input(4).empty())
      AddQuantizationScaleAndZeroPointToSkip(model_builder, node.input(4),
          node.input_size() > 5 ? node.input(5) : std::string{});
    if (node.input_size() > 6 && !node.input(6).empty())
      AddQuantizationScaleAndZeroPointToSkip(model_builder, node.input(6),
          node.input_size() > 7 ? node.input(7) : std::string{});
  }
}

// ---------------------------------------------------------------------------
// AddToModelBuilderImpl
// ---------------------------------------------------------------------------
NeuronStatus ResizeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                      const onnx::NodeProto& node,
                                                      const NeuronGraphContext& ctx) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  const auto& input  = node.input(0);
  const auto& output = node.output(0);

  const Shape& output_shape = shaper[output];
  const Shape& input_shape  = shaper[input];

  // Determine layout: NCHW if channel dim matches between input and output at index 1.
  const bool input_is_nchw = (output_shape.size() == 4 && input_shape.size() == 4 &&
                               output_shape[1] == input_shape[1]);

  // For QDQResize, validate that the registered operand's quant params match.
  if (IsQuantizedOp(node, ctx)) {
    // Retrieve scale/zp from the initializer inputs.
    // In the plugin EP, for QDQResize the x quant params are at input(4) and input(5).
    if (node.input_size() > 4 && !node.input(4).empty()) {
      float x_scale = 0.0f;
      int32_t x_zero_point = 0;
      const std::string& zp_name = (node.input_size() > 5) ? node.input(5) : "";
      NEURON_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(ctx, node.input(4), zp_name,
                                                               x_scale, x_zero_point));
      NEURON_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input, x_scale, x_zero_point));
    }
  }

  const bool is_linear_resize =
      NeuronGraphContext::GetAttrString(node, "mode", "nearest") == "linear";
  const int32_t operation_code =
      is_linear_resize ? NEURON_RESIZE_BILINEAR : NEURON_RESIZE_NEAREST_NEIGHBOR;

  const std::string coord_trans_mode =
      NeuronGraphContext::GetAttrString(node, "coordinate_transformation_mode", "half_pixel");
  const bool using_half_pixel    = (coord_trans_mode == "half_pixel");
  const bool using_align_corners = (coord_trans_mode == "align_corners");

  const int h_idx = input_is_nchw ? 2 : 1;
  const int w_idx = input_is_nchw ? 3 : 2;

  int32_t output_h = static_cast<int32_t>(output_shape[h_idx]);
  int32_t output_w = static_cast<int32_t>(output_shape[w_idx]);

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  ADD_SCALAR_OPERAND(model_builder, input_indices, output_w);
  ADD_SCALAR_OPERAND(model_builder, input_indices, output_h);
  ADD_SCALAR_OPERAND(model_builder, input_indices, static_cast<bool>(input_is_nchw));

  if (is_linear_resize) {
    if (using_align_corners || using_half_pixel) {
      ADD_SCALAR_OPERAND(model_builder, input_indices, using_align_corners);
      if (using_half_pixel)
        ADD_SCALAR_OPERAND(model_builder, input_indices, using_half_pixel);
    }
  }

  OperandType output_operand_type = operand_types.at(input);
  output_operand_type.SetDimensions(output_shape);
  NEURON_RETURN_IF_ERROR(model_builder.AddOperation(operation_code, input_indices,
                                                     {output}, {output_operand_type}));
  return NeuronStatus::OK();
}

// ---------------------------------------------------------------------------
// HasSupportedInputOutputsImpl
// ---------------------------------------------------------------------------
bool ResizeOpBuilder::HasSupportedInputOutputsImpl(const onnx::NodeProto& node,
                                                     const NeuronGraphContext& ctx,
                                                     const OpSupportCheckParams& params) const {
  if (!IsTypeSupported(node, ctx, 0, /*is_input*/ true,
                       {onnx::TensorProto::FLOAT,
                        onnx::TensorProto::UINT8})) {
    return false;
  }

  if (IsQuantizedOp(node, ctx)) {
    if (!IsQuantizedIOSupported(ctx, node, {0}, params, /*is_input*/ true))
      return false;
    if (!IsQuantizedIOSupported(ctx, node, {0}, params, /*is_input*/ false))
      return false;
  }

  return true;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
void CreateResizeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ResizeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace neuron
}  // namespace onnxruntime
