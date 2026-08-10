// split_op_builder.cc
// Plugin EP version – uses onnx::NodeProto + NeuronGraphContext.
//
// CHANGES FROM ORIGINAL:
//   node_unit.Inputs()[i].node_arg.Name() → node.input(i)
//   node_unit.Outputs()[i].node_arg.Name() → node.output(i)
//   helper.Get("axis", ...) → NeuronGraphContext::GetAttrInt(node, "axis", ...)
//   node_unit.SinceVersion() → NeuronGraphContext::GetAttrInt(node, "since_version", ...)
//     (plugin EP uses opset_import from the graph; for safety we check the
//      num_outputs attribute presence to detect opset 18+ style nodes)
//   Initializer unpacked_tensor → ctx.GetInt64Data(*splits_tensor, ...)
//   ORT_RETURN_IF_NOT → NEURON_RETURN_IF_NOT
//   ORT_RETURN_IF_ERROR → NEURON_RETURN_IF_ERROR
//   Status → NeuronStatus
//   Split opset 13+ only (GetMinSupportedOpSet = 13)

#include <algorithm>
#include <numeric>

#include <functional>
#include "base_op_builder.h"

namespace onnxruntime {
namespace neuron {
using namespace onnxruntime::neuron::op_builder_helpers;
using namespace onnxruntime::neuron::wrapper;

class SplitOpBuilder : public BaseOpBuilder {
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
                                     const OpSupportCheckParams& /*params*/) const override {
    return IsTypeSupported(node, ctx, 0, /*is_input*/ true,
                           {onnx::TensorProto::FLOAT,
                            onnx::TensorProto::UINT8});
  }

  // Split opset 13- uses "split" as attribute. Currently not supported.
  int GetMinSupportedOpSet(const onnx::NodeProto& /*node*/) const override { return 13; }
};

// Add operator related

void SplitOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder,
                                            const onnx::NodeProto& node,
                                            const NeuronGraphContext& /*ctx*/) const {
  // Optional second input "split" (constant initializer with split sizes).
  if (node.input_size() > 1 && !node.input(1).empty()) {
    model_builder.AddInitializerToSkip(node.input(1));
  }
}

NeuronStatus SplitOpBuilder::AddToModelBuilderImpl(
    ModelBuilder& model_builder,
    const onnx::NodeProto& node,
    const NeuronGraphContext& ctx) const {
  const auto& input_name = node.input(0);

  const int32_t axis = static_cast<int32_t>(NeuronGraphContext::GetAttrInt(node, "axis", 0));

  // Determine number of outputs.
  // In opset 18+, num_outputs is a required attribute; in opset 13-17 it equals
  // the number of connected output edges.
  int32_t num_outputs = node.output_size();
  // If num_outputs attribute is present, prefer it (handles opset 18+ style).
  const auto* num_outputs_attr = NeuronGraphContext::GetAttr(node, "num_outputs");
  if (num_outputs_attr != nullptr) {
    num_outputs = static_cast<int32_t>(num_outputs_attr->i());
  }

  NEURON_RETURN_IF_NOT(num_outputs > 0,
                       "Split: num_outputs resolved to 0 for node: ", node.name());

  std::vector<std::string> output_names;
  output_names.reserve(static_cast<size_t>(num_outputs));
  for (int32_t i = 0; i < num_outputs; ++i) {
    output_names.push_back(node.output(i));
  }

  // An explicit "split" input may divide the axis unevenly; NEURON_SPLIT cannot
  // express that, so those lower to one NEURON_SLICE per output instead.
  if (node.input_size() > 1 && !node.input(1).empty()) {
    const auto* splits_tensor = ctx.GetConstantTensor(node.input(1));
    NEURON_RETURN_IF_NOT(splits_tensor != nullptr,
                         "Split: 'split' input is not a compile-time constant for node: ", node.name());
    std::vector<int64_t> splits;
    NEURON_RETURN_IF_ERROR(NeuronGraphContext::GetInt64Data(*splits_tensor, splits));
    const bool uneven = std::adjacent_find(splits.begin(), splits.end(),
                                           std::not_equal_to<int64_t>()) != splits.end();
    if (uneven) {
      return op_builder_helpers::AddNeuronUnevenSplit(model_builder, input_name, axis,
                                                      splits, output_names);
    }
  }

  NEURON_RETURN_IF_ERROR(op_builder_helpers::AddNeuronSplit(model_builder, input_name, axis, output_names));
  return NeuronStatus::OK();
}

// Operator support related
void CreateSplitOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<SplitOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace neuron
}  // namespace onnxruntime
