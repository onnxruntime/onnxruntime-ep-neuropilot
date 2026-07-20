// identity_op_builder.cc
// Plugin EP version – uses onnx::NodeProto + NeuronGraphContext.

#include "base_op_builder.h"

namespace onnxruntime {
namespace neuron {
using namespace onnxruntime::neuron::op_builder_helpers;
using namespace onnxruntime::neuron::wrapper;

class IdentityOpBuilder : public BaseOpBuilder {
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

NeuronStatus IdentityOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                       const onnx::NodeProto& node,
                                                       const NeuronGraphContext& ctx) const {
  return AddIdentityOp(model_builder, node, ctx);
}

void CreateIdentityOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<IdentityOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace neuron
}  // namespace onnxruntime
