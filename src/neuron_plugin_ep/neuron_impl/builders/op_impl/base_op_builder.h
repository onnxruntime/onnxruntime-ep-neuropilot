// base_op_builder.h
// Plugin EP version – BaseOpBuilder uses onnx::NodeProto + NeuronGraphContext.

#pragma once

#include <onnx/onnx_pb.h>

#include <set>
#include <tuple>
#include <unordered_set>

#include "../op_builder.h"
#include "../op_builder_factory.h"
#include "../op_builder_helpers.h"
#include "../../neuron_graph_context.h"
#include "../../neuron_lib/neuron_types.h"

namespace onnxruntime {
namespace neuron {

template <class T>
void CreateSharedOpBuilderImpl(const std::string& op_type,
                               OpBuilderRegistrations& op_registrations,
                               const std::vector<std::string>& op_types) {
  if (op_registrations.op_builder_map.count(op_type)) return;
  op_registrations.builders.push_back(std::make_unique<T>());
  for (const auto& op : op_types) {
    op_registrations.op_builder_map.emplace(op, op_registrations.builders.back().get());
  }
}

class BaseOpBuilder : public IOpBuilder {
 public:
  virtual ~BaseOpBuilder() = default;

  // ---- IOpBuilder interface -------------------------------------------------
  void AddInitializersToSkip(ModelBuilder& /*model_builder*/,
                              const onnx::NodeProto& /*node*/,
                              const NeuronGraphContext& /*ctx*/) const override {}

  NeuronStatus AddToModelBuilder(ModelBuilder& model_builder,
                                  const onnx::NodeProto& node,
                                  const NeuronGraphContext& ctx) const override final;

  bool IsOpSupported(const onnx::NodeProto& node,
                     const NeuronGraphContext& ctx,
                     const neuron::OpSupportCheckParams& params) const override final;

  // is this op worth a dedicated NPU dispatch
  virtual bool IsComputeHeavy(const onnx::NodeProto& /*node*/) const { return false; }

 protected:
  // Subclasses implement the actual model-building here.
  virtual NeuronStatus AddToModelBuilderImpl(ModelBuilder& model_builder,
                                              const onnx::NodeProto& node,
                                              const NeuronGraphContext& ctx) const = 0;

  virtual bool IsQuantizedOp(const onnx::NodeProto& /*node*/,
                              const NeuronGraphContext& /*ctx*/) const { return false; }

  virtual NeuronRuntimeVersion GetMinSupportedNeuronVersion(
      const onnx::NodeProto& /*node*/,
      const neuron::OpSupportCheckParams& /*params*/) const {
    return {4, 1, 0};
  }

  virtual bool HasSupportedInputsImpl(
      const onnx::NodeProto& node,
      const NeuronGraphContext& ctx,
      const neuron::OpSupportCheckParams& params) const;

  virtual bool HasSupportedOutputsImpl(
      const onnx::NodeProto& node,
      const NeuronGraphContext& ctx,
      const neuron::OpSupportCheckParams& params) const;

  // Check that a specific input/output has an acceptable ONNX data type.
  virtual bool IsTypeSupported(
      const onnx::NodeProto& node,
      const NeuronGraphContext& ctx,
      size_t index,
      bool is_input,
      const std::unordered_set<int32_t>& supported_types) const;

  virtual bool IsQuantizedTypeSupported(
      const onnx::NodeProto& node,
      const NeuronGraphContext& ctx,
      size_t index,
      bool is_input,
      const std::set<std::tuple<int32_t, bool>>& supported_quantized_types) const;

  virtual int GetMinSupportedOpSet(const onnx::NodeProto& /*node*/) const { return 1; }
  virtual int GetMaxSupportedOpSet(const onnx::NodeProto& /*node*/) const { return 21; }

  // Whether a QDQ group node is supported (default: false for pure nodes).
  virtual bool IsNodeUnitTypeSupported(const onnx::NodeProto& /*node*/) const { return true; }

 private:
  bool HasSupportedOpSet(const onnx::NodeProto& node) const;
  bool HasSupportedInputs(const onnx::NodeProto& node,
                          const NeuronGraphContext& ctx,
                          const neuron::OpSupportCheckParams& params) const;
  bool HasSupportedOutputs(const onnx::NodeProto& node,
                           const NeuronGraphContext& ctx,
                           const neuron::OpSupportCheckParams& params) const;
};

}  // namespace neuron
}  // namespace onnxruntime
