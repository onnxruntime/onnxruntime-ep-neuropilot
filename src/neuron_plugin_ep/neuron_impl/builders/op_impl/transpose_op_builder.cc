// transpose_op_builder.cc
// Plugin EP version – uses onnx::NodeProto + NeuronGraphContext.
//
// CHANGES FROM ORIGINAL:
//   node_unit.Inputs()[i].node_arg.Name() → node.input(i)
//   node_unit.Outputs()[i].node_arg.Name() → node.output(i)
//   helper.Get("perm", vector<int32_t>{}) → NeuronGraphContext::GetAttrInts(node, "perm")
//   GetShape(node_arg, shape) → ctx.GetShape(name)
//   GetType(node_arg, type) → ctx.GetElementType(name)
//   ORT_RETURN_IF_NOT → NEURON_RETURN_IF_NOT
//   ORT_RETURN_IF_ERROR → NEURON_RETURN_IF_ERROR
//   Status → NeuronStatus
//   Quantized path removed (not needed in plugin EP non-QDQ flow)

#include "base_op_builder.h"
#include "../op_builder_helpers.h"

namespace onnxruntime {
namespace neuron {
using namespace onnxruntime::neuron::op_builder_helpers;
using namespace onnxruntime::neuron::wrapper;

class TransposeOpBuilder : public BaseOpBuilder {
 private:
  NeuronStatus AddToModelBuilderImpl(ModelBuilder& model_builder,
                                      const onnx::NodeProto& node,
                                      const NeuronGraphContext& ctx) const override;


  bool HasSupportedInputOutputsImpl(const onnx::NodeProto& node,
                                     const NeuronGraphContext& ctx,
                                     const OpSupportCheckParams& /*params*/) const override {
    return IsTypeSupported(node, ctx, 0, /*is_input*/ true,
                           {onnx::TensorProto::FLOAT,
                            onnx::TensorProto::UINT8,
                            onnx::TensorProto::INT8});
  }
};

NeuronStatus TransposeOpBuilder::AddToModelBuilderImpl(
    ModelBuilder& model_builder,
    const onnx::NodeProto& node,
    const NeuronGraphContext& ctx) const {
  auto& shaper(model_builder.GetShaper());

  const auto& input  = node.input(0);
  const auto& output = node.output(0);

  // Build the permutation vector.
  auto perm_int64 = NeuronGraphContext::GetAttrInts(node, "perm");
  auto input_dims  = static_cast<int32_t>(shaper[input].size());
  std::vector<int32_t> perm;
  if (perm_int64.empty()) {
    // Default: reverse all dimensions.
    perm.reserve(input_dims);
    for (int32_t i = input_dims - 1; i >= 0; i--)
      perm.push_back(i);
  } else {
    perm.reserve(perm_int64.size());
    for (int64_t v : perm_int64) perm.push_back(static_cast<int32_t>(v));
    NEURON_RETURN_IF_NOT(static_cast<int32_t>(perm.size()) == input_dims,
                          "Perm and input should have the same number of dimensions");
  }

  (void)ctx;  // ctx not needed for non-quantized path

  std::string perm_name = model_builder.GetUniqueName(node.name() + input + "perm");
  NEURON_RETURN_IF_ERROR(
      op_builder_helpers::AddNeuronTranspose(model_builder, input, perm_name,
                                              gsl::make_span(perm), output));
  return NeuronStatus::OK();
}
void CreateTransposeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<TransposeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace neuron
}  // namespace onnxruntime
