// flatten_op_builder.cc
// Plugin EP version – uses onnx::NodeProto + NeuronGraphContext.
//
// CHANGES FROM ORIGINAL:
//   node_unit.Inputs()[i].node_arg.Name() → node.input(i)
//   node_unit.Outputs()[i].node_arg.Name() → node.output(i)
//   helper.Get("axis", 1) → NeuronGraphContext::GetAttrInt(node, "axis", 1)
//   GetShape(node_arg, shape) → ctx.GetShape(name)
//   HandleNegativeAxis(axis, rank) stays the same
//   Flatten is implemented as Reshape to 2D
//   Status → NeuronStatus

#include "base_op_builder.h"
#include "../op_builder_helpers.h"

namespace onnxruntime {
namespace neuron {
using namespace onnxruntime::neuron::op_builder_helpers;
using namespace onnxruntime::neuron::wrapper;

namespace {
// Compute the two output dimensions for a Flatten op given the input shape and axis attribute.
void GetFlattenOutputShape(const onnx::NodeProto& node, const Shape& input_shape,
                            int32_t& dim_1, int32_t& dim_2) {
  int32_t rank = static_cast<int32_t>(input_shape.size());
  int32_t axis = static_cast<int32_t>(NeuronGraphContext::GetAttrInt(node, "axis", 1));
  // axis == rank is valid but HandleNegativeAxis only handles negative; skip for non-negative.
  if (axis < 0)
    axis = static_cast<int32_t>(HandleNegativeAxis(axis, rank));

  dim_1 = 1;
  for (int32_t i = 0; i < axis; ++i)
    dim_1 *= static_cast<int32_t>(input_shape[i]);
  dim_2 = 1;
  for (int32_t i = axis; i < rank; ++i)
    dim_2 *= static_cast<int32_t>(input_shape[i]);
}
}  // namespace

class FlattenOpBuilder : public BaseOpBuilder {
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

NeuronStatus FlattenOpBuilder::AddToModelBuilderImpl(
    ModelBuilder& model_builder,
    const onnx::NodeProto& node,
    const NeuronGraphContext& ctx) const {
  const auto& input = node.input(0);

  // Flatten is essentially a reshape to a 2D tensor.
  const auto input_shape = ctx.GetShape(input);
  int32_t dim_1 = 1;
  int32_t dim_2 = 1;
  GetFlattenOutputShape(node, input_shape, dim_1, dim_2);

  // If the input has a dynamic dimension, replace 0 with -1.
  // We cannot have both dim_1 and dim_2 be 0 (checked by the op_capability library).
  dim_1 = (dim_1 == 0) ? -1 : dim_1;
  dim_2 = (dim_2 == 0) ? -1 : dim_2;

  std::vector<int32_t> shape{dim_1, dim_2};
  return AddReshapeOperator(model_builder, node, ctx, input, shape);
}
void CreateFlattenOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<FlattenOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace neuron
}  // namespace onnxruntime
