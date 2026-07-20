// minmax_op_builder.cc
// Plugin EP version – uses onnx::NodeProto + NeuronGraphContext.
// Shared by: Min, Max.

#include "base_op_builder.h"

namespace onnxruntime {
namespace neuron {
using namespace onnxruntime::neuron::op_builder_helpers;
using namespace onnxruntime::neuron::wrapper;

class MinMaxOpBuilder : public BaseOpBuilder {
 private:
  NeuronStatus AddToModelBuilderImpl(ModelBuilder& model_builder,
                                      const onnx::NodeProto& node,
                                      const NeuronGraphContext& ctx) const override;


  // Min/Max opset 5- uses consumed_inputs attribute which is not supported
  int GetMinSupportedOpSet(const onnx::NodeProto& /*node*/) const override { return 6; }
};

NeuronStatus MinMaxOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                     const onnx::NodeProto& node,
                                                     const NeuronGraphContext& ctx) const {
  const std::string& input1 = node.input(0);
  const std::string& input2 = node.input(1);
  return AddMinMaxOperator(model_builder, node, ctx, input1, input2);
}
void CreateMinMaxOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  CreateSharedOpBuilderImpl<MinMaxOpBuilder>(
      op_type, op_registrations,
      {
          "Min",
          "Max",
      });
}

}  // namespace neuron
}  // namespace onnxruntime
