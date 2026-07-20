// gridsample_op_builder.cc
// Plugin EP version – uses onnx::NodeProto + NeuronGraphContext.
//
// Ported from in-tree gridsample_op_builder.cc.
//
// Implements ONNX GridSample op mapping to Neuron backend.
// Notes:
// - This mapping currently supports 2D grid sampling only:
//     X:    (N, C, H, W)
//     grid: (N, H_out, W_out, 2)
//     Y:    (N, C, H_out, W_out)
// - Attributes are converted and passed as scalar operands in the order:
//     [resamplingType:int32, borderType:int32, alignCorners:bool]
//   according to the following enum layouts:
//     NEAREST=0, BILINEAR=1  (resamplingType)
//     ZERO=0, DUPLICATE=1, REFLECTION=2  (borderType)
//
// Translation notes:
//   node_unit.Inputs()[i].node_arg.Name()       → node.input(i)
//   node_unit.Outputs()[i].node_arg.Name()       → node.output(i)
//   node_unit.OpType()                           → node.op_type()
//   node_unit.Name()                             → node.name()
//   helper.Get("attr", val)                      → NeuronGraphContext::GetAttrXxx(node, "attr", val)
//   ORT_RETURN_IF_ERROR / ORT_RETURN_IF_NOT      → NEURON_RETURN_IF_ERROR / NEURON_RETURN_IF_NOT
//   ORT_MAKE_STATUS(ONNXRUNTIME, ...)            → NeuronStatus{false, MakeNeuronString(...)}
//   Status::OK()                                 → NeuronStatus::OK()
//   NodeUnit::Type::SingleNode check             → IsNodeUnitTypeSupported always true
//     (plugin EP sends plain NodeProtos only, never QDQ groups)
//   ContainsDevice(params.devices, "mtk-dsp")   → model_builder.HasDevice("mtk-dsp") in builder;
//     op_capability uses params.devices directly via ContainsDevice.

#include "base_op_builder.h"

#include <algorithm>
#include <cctype>
#include <string>

#include <onnx/onnx_pb.h>

#include "../op_builder_helpers.h"
#include "../../neuron_lib/neuron_wrapper_device.h"

namespace onnxruntime {
namespace neuron {
using namespace onnxruntime::neuron::op_builder_helpers;
using namespace onnxruntime::neuron::wrapper;

namespace {
// Convert ONNX GridSample attributes to Neuron enum ints and align_corners bool.
// Returns NeuronStatus::OK() on success or an error otherwise.
NeuronStatus GetNeuronGridSampleAttributes(const onnx::NodeProto& node,
                                            int32_t& resample_type_i32,
                                            int32_t& border_type_i32,
                                            bool& align_corners_bool) {
  std::string mode         = NeuronGraphContext::GetAttrString(node, "mode", "linear");
  std::string padding_mode = NeuronGraphContext::GetAttrString(node, "padding_mode", "zeros");
  int64_t align_corners    = NeuronGraphContext::GetAttrInt(node, "align_corners", 0);

  // Normalize to lowercase.
  auto to_lower = [](std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(::tolower(c)); });
    return s;
  };
  mode         = to_lower(mode);
  padding_mode = to_lower(padding_mode);

  // Map mode → resamplingType.
  if (mode == "nearest") {
    resample_type_i32 = 0;  // NEAREST
  } else if (mode == "linear" || mode == "bilinear") {
    resample_type_i32 = 1;  // BILINEAR
  } else {
    return NeuronStatus{false,
        MakeNeuronString("GridSample: unsupported mode '", mode, "'. Supported: 'nearest', 'linear'.")};
  }

  // Map padding_mode → borderType.
  if (padding_mode == "zeros") {
    border_type_i32 = 0;  // ZERO
  } else if (padding_mode == "border") {
    border_type_i32 = 1;  // DUPLICATE
  } else if (padding_mode == "reflection") {
    border_type_i32 = 2;  // REFLECTION
  } else {
    return NeuronStatus{false,
        MakeNeuronString("GridSample: unsupported padding_mode '", padding_mode,
                         "'. Supported: 'zeros', 'border', 'reflection'.")};
  }

  align_corners_bool = (align_corners != 0);
  return NeuronStatus::OK();
}
}  // namespace

class GridSampleOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& /*model_builder*/,
                              const onnx::NodeProto& /*node*/,
                              const NeuronGraphContext& /*ctx*/) const override {
    // No constant initializers to skip for GridSample.
  }

 private:
  NeuronStatus AddToModelBuilderImpl(ModelBuilder& model_builder,
                                      const onnx::NodeProto& node,
                                      const NeuronGraphContext& ctx) const override;


  bool HasSupportedInputOutputsImpl(const onnx::NodeProto& node,
                                     const NeuronGraphContext& ctx,
                                     const OpSupportCheckParams& params) const override;

  bool IsNodeUnitTypeSupported(const onnx::NodeProto& /*node*/) const override {
    // Plugin EP sends plain NodeProtos only.
    return true;
  }

  bool IsQuantizedOp(const onnx::NodeProto& /*node*/,
                     const NeuronGraphContext& /*ctx*/) const override {
    return false;
  }

  int GetMinSupportedOpSet(const onnx::NodeProto& /*node*/) const override { return 16; }
};

// ---------------------------------------------------------------------------
// AddToModelBuilderImpl
// ---------------------------------------------------------------------------
NeuronStatus GridSampleOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                          const onnx::NodeProto& node,
                                                          const NeuronGraphContext& ctx) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  NEURON_RETURN_IF_NOT(node.input_size() == 2,
                        "GridSample expects 2 inputs but got: ", node.input_size());

  const std::string& x    = node.input(0);  // Input tensor X
  const std::string& grid = node.input(1);  // Sampling grid
  const std::string& y    = node.output(0); // Output tensor Y

  int32_t resample_type_i32 = 0;
  int32_t border_type_i32   = 0;
  bool align_corners_bool   = false;
  NEURON_RETURN_IF_ERROR(GetNeuronGridSampleAttributes(node, resample_type_i32,
                                                        border_type_i32, align_corners_bool));

  // Build input list: [X, grid, resamplingType:int32, borderType:int32, alignCorners:bool]
  std::vector<uint32_t> input_indices;
  input_indices.reserve(5);
  input_indices.push_back(operand_indices.at(x));
  input_indices.push_back(operand_indices.at(grid));

  ADD_SCALAR_OPERAND(model_builder, input_indices, resample_type_i32);
  ADD_SCALAR_OPERAND(model_builder, input_indices, border_type_i32);

  // align_corners bool scalar supported in Neuron >= 9.0.0
  if (GreaterEqualVersion(model_builder.GetNeuronVersion(), {9, 0, 0})) {
    uint32_t idx_bool = 0;
    NEURON_RETURN_IF_ERROR(model_builder.AddOperandFromScalar(align_corners_bool, idx_bool));
    input_indices.push_back(idx_bool);
  }

  const OperandType output_operand_type(operand_types.at(x).type, shaper[y],
                                         operand_types.at(x).operandType.scale,
                                         operand_types.at(x).operandType.zeroPoint);

  NEURON_RETURN_IF_ERROR(model_builder.AddOperation(NEURON_GRID_SAMPLE_2D,
                                                     input_indices,
                                                     {y},
                                                     {output_operand_type}));
  return NeuronStatus::OK();
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------// ---------------------------------------------------------------------------
// HasSupportedInputOutputsImpl
// ---------------------------------------------------------------------------
bool GridSampleOpBuilder::HasSupportedInputOutputsImpl(const onnx::NodeProto& node,
                                                         const NeuronGraphContext& ctx,
                                                         const OpSupportCheckParams& /*params*/) const {
  // X (input 0): float32 or float16
  if (!IsTypeSupported(node, ctx, 0, /*is_input*/ true,
                       {onnx::TensorProto::FLOAT, onnx::TensorProto::FLOAT16})) {
    return false;
  }
  // grid (input 1): float32 or float16
  if (!IsTypeSupported(node, ctx, 1, /*is_input*/ true,
                       {onnx::TensorProto::FLOAT, onnx::TensorProto::FLOAT16})) {
    return false;
  }
  // Y (output 0): float32 or float16
  if (!IsTypeSupported(node, ctx, 0, /*is_input*/ false,
                       {onnx::TensorProto::FLOAT, onnx::TensorProto::FLOAT16})) {
    return false;
  }
  return true;
}

void CreateGridSampleOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  CreateSharedOpBuilderImpl<GridSampleOpBuilder>(
      op_type, op_registrations,
      {
          "GridSample",
      });
}

}  // namespace neuron
}  // namespace onnxruntime
