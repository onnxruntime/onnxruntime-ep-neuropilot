// unsqueeze_op_builder.cc
// Plugin EP version – uses onnx::NodeProto + NeuronGraphContext.
//
// CHANGES FROM ORIGINAL:
//   node_unit.Inputs()[i].node_arg.Name() → node.input(i)
//   node_unit.Outputs()[i].node_arg.Name() → node.output(i)
//   node_unit.SinceVersion() → detected via "since_version" attribute
//   graph_viewer.GetConstantInitializer(name) → ctx.IsInitializer(name)
//   GetShape(node_arg, shape) → ctx.GetShape(name)
//   ORT_RETURN_IF_ERROR → NEURON_RETURN_IF_ERROR
//   Status → NeuronStatus
//
// Neuron does not support Unsqueeze natively; it is lowered to RESHAPE.

#include "base_op_builder.h"
#include "../op_builder_helpers.h"

namespace onnxruntime {
namespace neuron {
using namespace onnxruntime::neuron::op_builder_helpers;
using namespace onnxruntime::neuron::wrapper;

class UnsqueezeOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder,
                              const onnx::NodeProto& node,
                              const NeuronGraphContext& /*ctx*/) const override;

 private:
  NeuronStatus AddToModelBuilderImpl(ModelBuilder& model_builder,
                                      const onnx::NodeProto& node,
                                      const NeuronGraphContext& ctx) const override;

};

void UnsqueezeOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder,
                                                 const onnx::NodeProto& node,
                                                 const NeuronGraphContext& /*ctx*/) const {
  // Unsqueeze opset 13+ stores axes in input[1] (INT64 initializer).
  // The GraphProto produced by OrtGraphToProto does not carry a "since_version"
  // attribute, so we cannot reliably detect the opset here.  Instead we simply
  // skip input[1] whenever it is present: for opset < 13 axes are an attribute
  // and input[1] is absent/empty; for opset 13+ input[1] is the axes tensor.
  if (node.input_size() > 1 && !node.input(1).empty()) {
    model_builder.AddInitializerToSkip(node.input(1));
  }
}

NeuronStatus UnsqueezeOpBuilder::AddToModelBuilderImpl(
    ModelBuilder& model_builder,
    const onnx::NodeProto& node,
    const NeuronGraphContext& ctx) const {
  auto& shaper(model_builder.GetShaper());
  const auto& input = node.input(0);

  // Neuron does not support Unsqueeze; implement it as RESHAPE using the axes to
  // compute the output shape.
  std::vector<int32_t> axes;
  NEURON_RETURN_IF_ERROR(GetAxesForSqueezeAndUnSqueeze(model_builder, node, ctx, axes));

  Shape input_shape = shaper[input];
  const size_t input_dims = input_shape.size();
  const uint32_t output_rank = static_cast<uint32_t>(input_dims + axes.size());

  // Normalise negative axes against the output rank.
  for (auto& axis : axes) {
    axis = static_cast<int32_t>(HandleNegativeAxis(axis, output_rank));
  }
  std::sort(axes.begin(), axes.end());

  // Build the output shape by copying the input dims and inserting 1's at the
  // requested positions (in sorted order so earlier insertions don't shift later ones).
  std::vector<int32_t> shape;
  shape.reserve(output_rank);
  for (uint32_t d : input_shape) shape.push_back(static_cast<int32_t>(d));
  for (size_t i = 0; i < axes.size(); i++) {
    shape.insert(shape.cbegin() + axes[i], 1);
  }

  return AddReshapeOperator(model_builder, node, ctx, input, shape);
}
void CreateUnsqueezeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<UnsqueezeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace neuron
}  // namespace onnxruntime
