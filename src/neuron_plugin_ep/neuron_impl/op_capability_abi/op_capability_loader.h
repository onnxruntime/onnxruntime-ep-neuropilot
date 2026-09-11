// op_capability_loader.h
// Plugin EP side of the C ABI in neuron_op_capability_abi.h: loads
// libneuron_op_capability.so via dlopen and resolves its entry points.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "neuron_op_capability_abi.h"

namespace onnx {
class GraphProto;
}

namespace onnxruntime {
namespace neuron {
namespace op_capability {

// Singleton owning the dlopen handle and the resolved entry points.
class OpCapabilityLoader {
 public:
  static OpCapabilityLoader& Instance();

  // True if the library loaded, reported a compatible ABI version, and exposed
  // every required entry point.
  bool IsLoaded() const { return loaded_; }

  // Whether a capability is registered for an ONNX op type. NEURON_CAP_FALSE
  // means "no constraints known", not "rejected"; NEURON_CAP_ERROR must be
  // treated as a rejection.
  int32_t HasCapability(const std::string& op_type) const;

  // --- Raw ABI passthrough. Prefer CapabilityGraphHandle below. -------------
  NeuronCapGraph* CreateGraph(const void* bytes, size_t length) const;
  void ReleaseGraph(NeuronCapGraph* graph) const;

  int32_t IsNodeUnitTypeSupported(NeuronCapGraph* graph, int32_t node_index) const;
  int32_t IsNodeSupported(NeuronCapGraph* graph,
                          int32_t node_index,
                          const NeuronCapParams* params) const;

  OpCapabilityLoader(const OpCapabilityLoader&) = delete;
  OpCapabilityLoader& operator=(const OpCapabilityLoader&) = delete;

 private:
  OpCapabilityLoader();
  ~OpCapabilityLoader();

  void LoadLibrary();

  void* lib_handle_{nullptr};
  bool loaded_{false};

  decltype(&NeuronCapGetAbiVersion) get_abi_version_{nullptr};
  decltype(&NeuronCapCreateGraph) create_graph_{nullptr};
  decltype(&NeuronCapReleaseGraph) release_graph_{nullptr};
  decltype(&NeuronCapHasCapability) has_capability_{nullptr};
  decltype(&NeuronCapIsNodeUnitTypeSupported) is_node_unit_type_supported_{nullptr};
  decltype(&NeuronCapIsNodeSupported) is_node_supported_{nullptr};
};

// RAII wrapper around a NeuronCapGraph. Serializes a GraphProto and hands the
// bytes to the library, which parses them with its own protobuf. Bulk tensor
// payloads are stripped before serializing and restored afterwards (see
// graph_payload_elision.h) -- hence the non-const reference; the graph is
// unchanged by the time the constructor returns. valid() is false if the
// library is not loaded or the graph could not be parsed.
class CapabilityGraphHandle {
 public:
  CapabilityGraphHandle() = default;

  // Serializes `graph` and creates the library-side handle.
  explicit CapabilityGraphHandle(onnx::GraphProto& graph);

  ~CapabilityGraphHandle();

  CapabilityGraphHandle(CapabilityGraphHandle&& other) noexcept;
  CapabilityGraphHandle& operator=(CapabilityGraphHandle&& other) noexcept;

  CapabilityGraphHandle(const CapabilityGraphHandle&) = delete;
  CapabilityGraphHandle& operator=(const CapabilityGraphHandle&) = delete;

  bool valid() const { return handle_ != nullptr; }
  NeuronCapGraph* get() const { return handle_; }

 private:
  void Reset();

  NeuronCapGraph* handle_{nullptr};
};

}  // namespace op_capability
}  // namespace neuron
}  // namespace onnxruntime
