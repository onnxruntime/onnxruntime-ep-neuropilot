// reshape_op_builder.cc
// Plugin EP version – uses onnx::NodeProto + NeuronGraphContext.
//
// CHANGES FROM ORIGINAL:
//   node_unit.Inputs()[i].node_arg.Name() → node.input(i)
//   node_unit.Outputs()[i].node_arg.Name() → node.output(i)
//   GetConstantInitializer(name) → ctx.GetInitializer(name)
//   Initializer unpacked_tensor(shape_tensor) → NeuronGraphContext::GetInt64Data(...)
//   helper.Get("allowzero", 0) → NeuronGraphContext::GetAttrInt(node, "allowzero", 0)
//   GetShape(node_arg, shape) → ctx.GetShape(name)
//   GetType(node_arg, type) → ctx.GetElementType(name)
//   ORT_RETURN_IF_ERROR → NEURON_RETURN_IF_ERROR
//   Status → NeuronStatus

#include "base_op_builder.h"
#include "../op_builder_helpers.h"

namespace onnxruntime {
namespace neuron {
using namespace onnxruntime::neuron::op_builder_helpers;
using namespace onnxruntime::neuron::wrapper;

class ReshapeOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder,
                              const onnx::NodeProto& node,
                              const NeuronGraphContext& /*ctx*/) const override;

 private:
  NeuronStatus AddToModelBuilderImpl(ModelBuilder& model_builder,
                                      const onnx::NodeProto& node,
                                      const NeuronGraphContext& ctx) const override;


  int GetMinSupportedOpSet(const onnx::NodeProto& /*node*/) const override { return 5; }

  bool HasSupportedInputsImpl(const onnx::NodeProto& node,
                                     const NeuronGraphContext& ctx,
                                     const OpSupportCheckParams& /*params*/) const override {
    return IsTypeSupported(node, ctx, 0, /*is_input*/ true,
                           {onnx::TensorProto::FLOAT,
                            onnx::TensorProto::UINT8,
                            onnx::TensorProto::INT8});
  }
};

void ReshapeOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder,
                                               const onnx::NodeProto& node,
                                               const NeuronGraphContext& /*ctx*/) const {
  // Skip the shape input (input 1) — it is consumed directly, not added as a Neuron operand.
  model_builder.AddInitializerToSkip(node.input(1));
}

NeuronStatus ReshapeOpBuilder::AddToModelBuilderImpl(
    ModelBuilder& model_builder,
    const onnx::NodeProto& node,
    const NeuronGraphContext& ctx) const {
  auto& shaper(model_builder.GetShaper());

  const auto& input = node.input(0);

  // Read the new shape from the constant initializer (input 1).
  const auto* shape_tensor = ctx.GetInitializer(node.input(1));
  NEURON_RETURN_IF_NOT(shape_tensor != nullptr,
                        "Reshape: shape input must be a constant initializer");

  std::vector<int64_t> raw_shape;
  NEURON_RETURN_IF_ERROR(NeuronGraphContext::GetInt64Data(*shape_tensor, raw_shape));
  const uint32_t size = static_cast<uint32_t>(raw_shape.size());

  const auto input_shape = shaper[input];
  std::vector<int32_t> shape(size);
  for (uint32_t i = 0; i < size; i++) {
    int32_t dim = static_cast<int32_t>(raw_shape[i]);
    // Neuron reshape does not support 0 as dimension — map to input dim.
    shape[i] = (dim == 0 && i < input_shape.size()) ? static_cast<int32_t>(input_shape[i]) : dim;
  }

  return AddReshapeOperator(model_builder, node, ctx, input, shape);
}
void CreateReshapeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ReshapeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace neuron
}  // namespace onnxruntime
