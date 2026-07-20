// op_capability_interface.h
// Plugin EP version – uses onnx::NodeProto + NeuronGraphContext (no ORT internals).
//
// Defines the IOpCapability interface, OpCapabilityRegistrations registry,
// and the CreateOpCapabilityRegistrationsFn function pointer type.
// This header is shared between the main EP (loader) and the capability .so (factory).

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include <onnx/onnx_pb.h>

// Full definitions needed by both the capability .so and the main EP loader.
#include "../neuron_graph_context.h"
#include "../builders/op_builder.h"

namespace onnxruntime {
namespace neuron {
namespace op_capability {

// Per-operator capability checker interface.
// Each op capability class implements this to express its hardware constraints,
// mirroring what BaseOpBuilder::IsOpSupportedImpl does in the main EP.
class IOpCapability {
 public:
  virtual ~IOpCapability() = default;

  // Mirrors BaseOpBuilder::IsOpSupportedImpl.
  // Called after base checks (opset, I/O types, version) have passed.
  virtual bool IsOpSupported(const onnx::NodeProto& node,
                              const NeuronGraphContext& ctx,
                              const OpSupportCheckParams& params) const = 0;

  // Mirrors BaseOpBuilder::IsNodeUnitTypeSupported.
  // Default: return true (plugin EP only sends plain NodeProtos).
  virtual bool IsNodeUnitTypeSupported(const onnx::NodeProto& /*node*/) const {
    return true;
  }
};

// Registry: maps ONNX op_type string → IOpCapability instance.
struct OpCapabilityRegistrations {
  std::unordered_map<std::string, std::unique_ptr<IOpCapability>> capabilities;
};

// Shared library entry point signature.
// The .so exports this function; the main EP loads it via dlsym.
using CreateOpCapabilityRegistrationsFn = void (*)(OpCapabilityRegistrations&);

}  // namespace op_capability
}  // namespace neuron
}  // namespace onnxruntime
