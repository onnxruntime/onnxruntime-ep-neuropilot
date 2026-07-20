// batchnorm_op_builder.cc
// Plugin EP version – uses onnx::NodeProto + NeuronGraphContext.
//
// CHANGES FROM ORIGINAL:
//   node_unit.Inputs()[i].node_arg.Name() → node.input(i)
//   node_unit.Outputs()[i].node_arg.Name() → node.output(i)
//   helper.Get("epsilon", ...) → NeuronGraphContext::GetAttrFloat(node, "epsilon", ...)
//   Initializer unpacked_xxx → NeuronGraphContext::GetFloatData(*tensor, ...)
//   ORT_RETURN_IF_ERROR → NEURON_RETURN_IF_ERROR
//   Status → NeuronStatus

#include <cmath>
#include <vector>

#include "base_op_builder.h"

namespace onnxruntime {
namespace neuron {
using namespace onnxruntime::neuron::op_builder_helpers;
using namespace onnxruntime::neuron::wrapper;

class BatchNormalizationOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder,
                              const onnx::NodeProto& node,
                              const NeuronGraphContext& ctx) const override;

 private:
  NeuronStatus AddToModelBuilderImpl(ModelBuilder& model_builder,
                                      const onnx::NodeProto& node,
                                      const NeuronGraphContext& ctx) const override;


  // BatchNormalization opset 6- has unsupported attributes
  int GetMinSupportedOpSet(const onnx::NodeProto& /*node*/) const override { return 7; }
};

// Add operator related

void BatchNormalizationOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder,
                                                         const onnx::NodeProto& node,
                                                         const NeuronGraphContext& /*ctx*/) const {
  // skip everything except input0 for BatchNormalization
  model_builder.AddInitializerToSkip(node.input(1));  // scale
  model_builder.AddInitializerToSkip(node.input(2));  // B
  model_builder.AddInitializerToSkip(node.input(3));  // mean
  model_builder.AddInitializerToSkip(node.input(4));  // var
}

NeuronStatus BatchNormalizationOpBuilder::AddToModelBuilderImpl(
    ModelBuilder& model_builder,
    const onnx::NodeProto& node,
    const NeuronGraphContext& ctx) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_types(model_builder.GetOperandTypes());

  const auto& input  = node.input(0);
  const auto& output = node.output(0);
  const auto input_shape = shaper[input];

  const auto* scale_tensor = ctx.GetInitializer(node.input(1));
  const auto* bias_tensor  = ctx.GetInitializer(node.input(2));
  const auto* mean_tensor  = ctx.GetInitializer(node.input(3));
  const auto* var_tensor   = ctx.GetInitializer(node.input(4));

  NEURON_RETURN_IF_NOT(scale_tensor && bias_tensor && mean_tensor && var_tensor,
                       "BatchNormalization: scale/B/mean/var must be constant initializers");

  const auto eps = NeuronGraphContext::GetAttrFloat(node, "epsilon", 1e-5f);

  std::vector<float> scale_data, bias_data, mean_data, var_data;
  NEURON_RETURN_IF_ERROR(NeuronGraphContext::GetFloatData(*scale_tensor, scale_data));
  NEURON_RETURN_IF_ERROR(NeuronGraphContext::GetFloatData(*bias_tensor,  bias_data));
  NEURON_RETURN_IF_ERROR(NeuronGraphContext::GetFloatData(*mean_tensor,  mean_data));
  NEURON_RETURN_IF_ERROR(NeuronGraphContext::GetFloatData(*var_tensor,   var_data));

  const uint32_t size = static_cast<uint32_t>(scale_data.size());
  std::vector<float> a, b;
  a.reserve(size);
  b.reserve(size);
  for (uint32_t i = 0; i < size; i++) {
    float inv_std = 1.0f / std::sqrt(var_data[i] + eps);
    a.push_back(scale_data[i] * inv_std);
    b.push_back(-scale_data[i] * mean_data[i] * inv_std + bias_data[i]);
  }

  const auto tensor_a_name = model_builder.GetUniqueName(node.name() + input + "_imm_a");
  const auto tensor_b_name = model_builder.GetUniqueName(node.name() + input + "_imm_b");
  const auto tensor_imm_product_name = model_builder.GetUniqueName(node.name() + input + "_imm_mul");

  bool use_nchw = model_builder.UseNCHW();

  Shape tensor_a_dimen = {size};
  if (use_nchw) {
    // NCHW: broadcast on channel dim C (dim 1).
    // input rank 2 → {size}, rank 3 → {size, 1}, rank 4 → {size, 1, 1}
    const auto input_rank = input_shape.size();
    for (size_t i = 2; i < input_rank; i++)
      tensor_a_dimen.push_back(1);
  }

  shaper.AddShape(tensor_a_name, tensor_a_dimen);
  shaper.AddShape(tensor_b_name, tensor_a_dimen);
  const OperandType a_operand_type(operand_types.at(input).type, tensor_a_dimen);
  NEURON_RETURN_IF_ERROR(
      model_builder.AddOperandFromPersistMemoryBuffer(tensor_a_name, a.data(), a_operand_type));
  const OperandType b_operand_type(operand_types.at(input).type, tensor_a_dimen);
  NEURON_RETURN_IF_ERROR(
      model_builder.AddOperandFromPersistMemoryBuffer(tensor_b_name, b.data(), b_operand_type));

  int32_t fuse_code = model_builder.FindActivation(node, ctx);
  NEURON_RETURN_IF_ERROR(AddNeuronBatchNormalization(model_builder, input, tensor_a_name, tensor_b_name,
                                                     tensor_imm_product_name, output, fuse_code));
  return NeuronStatus::OK();
}

// Operator support related
void CreateBatchNormalizationOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<BatchNormalizationOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace neuron
}  // namespace onnxruntime
