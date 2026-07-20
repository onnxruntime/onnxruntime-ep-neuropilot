// squeeze_op_builder.cc
// Plugin EP version – uses onnx::NodeProto + NeuronGraphContext.
//
// CHANGES FROM ORIGINAL:
//   node_unit.Inputs()[i].node_arg.Name() → node.input(i)
//   node_unit.Outputs()[i].node_arg.Name() → node.output(i)
//   node_unit.SinceVersion() → read from "since_version" attribute (handled by
//       GetAxesForSqueezeAndUnSqueeze in op_builder_helpers)
//   graph_viewer.GetConstantInitializer(name) → ctx.IsInitializer(name)
//   GetShape(node_arg, shape) → ctx.GetShape(name)
//   ORT_RETURN_IF_ERROR → NEURON_RETURN_IF_ERROR
//   Status → NeuronStatus
// Handles: {"Squeeze"}

#include "base_op_builder.h"
#include "../op_builder_helpers.h"

namespace onnxruntime {
namespace neuron {
using namespace onnxruntime::neuron::op_builder_helpers;
using namespace onnxruntime::neuron::wrapper;

class SqueezeOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder,
                              const onnx::NodeProto& node,
                              const NeuronGraphContext& /*ctx*/) const override;

 private:
  NeuronStatus AddToModelBuilderImpl(ModelBuilder& model_builder,
                                      const onnx::NodeProto& node,
                                      const NeuronGraphContext& ctx) const override;


  bool HasSupportedInputOutputsImpl(const onnx::NodeProto& node,
                                     const NeuronGraphContext& ctx,
                                     const OpSupportCheckParams& /*params*/) const override {
    return IsTypeSupported(node, ctx, 0, /*is_input*/ true,
                           {onnx::TensorProto::FLOAT,
                            onnx::TensorProto::UINT8});
  }
};

void SqueezeOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder,
                                               const onnx::NodeProto& node,
                                               const NeuronGraphContext& /*ctx*/) const {
  // Squeeze opset 13+ stores axes in input[1] (INT64 initializer).
  // The GraphProto from OrtGraphToProto does not carry a "since_version" attribute,
  // so skip input[1] whenever it is present: for opset < 13 it is absent/empty.
  if (node.input_size() > 1 && !node.input(1).empty()) {
    model_builder.AddInitializerToSkip(node.input(1));
  }
}

NeuronStatus SqueezeOpBuilder::AddToModelBuilderImpl(
    ModelBuilder& model_builder,
    const onnx::NodeProto& node,
    const NeuronGraphContext& ctx) const {
  const auto& input  = node.input(0);
  const auto& output = node.output(0);

  std::vector<int32_t> axes;
  NEURON_RETURN_IF_ERROR(GetAxesForSqueezeAndUnSqueeze(model_builder, node, ctx, axes));
  return AddSqueezeOp(model_builder, node.name(), input, output, axes);
}
void CreateSqueezeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<SqueezeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace neuron
}  // namespace onnxruntime
