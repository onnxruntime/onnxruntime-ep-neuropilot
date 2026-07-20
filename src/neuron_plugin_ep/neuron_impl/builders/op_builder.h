// op_builder.h
// Plugin EP version – IOpBuilder interface uses onnx::NodeProto + NeuronGraphContext
// instead of ORT internal NodeUnit + GraphViewer.

#pragma once

#include <onnx/onnx_pb.h>

#include "../neuron_graph_context.h"
#include "../neuron_lib/neuron_types.h"
#include "../neuron_lib/neuron_wrapper_device.h"
#include "neuron_model_builder.h"

namespace onnxruntime {
namespace neuron {

struct OpSupportCheckParams {
  NeuronRuntimeVersion version = {
      .major = 1,
      .minor = 0,
      .patch = 0};
  neuron::wrapper::DeviceVector devices;
  bool use_nchw = false;
};

class IOpBuilder {
 public:
  virtual ~IOpBuilder() = default;

  // If the operator needs to skip certain initializers (e.g., weight tensors
  // that will be embedded directly), register them here.
  virtual void AddInitializersToSkip(ModelBuilder& model_builder,
                                     const onnx::NodeProto& node,
                                     const NeuronGraphContext& ctx) const = 0;

  // Add the operator to the Neuron model.
  virtual NeuronStatus AddToModelBuilder(ModelBuilder& model_builder,
                                          const onnx::NodeProto& node,
                                          const NeuronGraphContext& ctx) const = 0;

  // Check whether this operator is supported by the Neuron EP.
  virtual bool IsOpSupported(const onnx::NodeProto& node,
                              const NeuronGraphContext& ctx,
                              const OpSupportCheckParams& params) const = 0;
};

}  // namespace neuron
}  // namespace onnxruntime
