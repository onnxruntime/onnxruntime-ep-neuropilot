// neuron_ep.h
// NeuronEp: per-session execution provider.
// Calls NeuronWrapper_* functions directly (linked at build time).

#pragma once

#include <memory>
#include <string>

#include "neuron_ep_utils.h"
#include "neuron_execution_provider_wrapper.h"

class NeuronEpFactory;  // forward declaration

// ---------------------------------------------------------------------------
// NeuronEp
// One instance per ORT session.  Inherits OrtEp (C vtable struct) and ApiPtrs.
// ---------------------------------------------------------------------------
class NeuronEp : public OrtEp, public ApiPtrs {
 public:
  struct Config {
    // Provider options forwarded verbatim to NeuronWrapper_Create.
    // Pointer is NOT owned; it is owned by NeuronEpFactory.
    const OrtKeyValuePairs* provider_options{nullptr};
  };

  NeuronEp(NeuronEpFactory& factory,
           const std::string& ep_name,
           const Config& config,
           const OrtLogger& logger);

  ~NeuronEp();

 private:
  // ---- OrtEp vtable implementations (all static, ORT_API_CALL) ------------
  static const char* ORT_API_CALL GetNameImpl(const OrtEp* ep) noexcept;

  static OrtStatus* ORT_API_CALL GetCapabilityImpl(
      OrtEp* ep,
      const OrtGraph* graph,
      OrtEpGraphSupportInfo* support_info) noexcept;

  static OrtStatus* ORT_API_CALL CompileImpl(
      OrtEp* ep,
      const OrtGraph** graphs,
      const OrtNode** fused_nodes,
      size_t count,
      OrtNodeComputeInfo** node_compute_infos,
      OrtNode** ep_context_nodes) noexcept;

  static void ORT_API_CALL ReleaseNodeComputeInfosImpl(
      OrtEp* ep,
      OrtNodeComputeInfo** infos,
      size_t num) noexcept;

  static OrtStatus* ORT_API_CALL GetPreferredDataLayoutImpl(
      OrtEp* ep,
      OrtEpDataLayout* preferred_data_layout) noexcept;

  static OrtStatus* ORT_API_CALL ShouldConvertDataLayoutForOpImpl(
      OrtEp* ep,
      const char* domain,
      const char* op_type,
      OrtEpDataLayout target_data_layout,
      int* should_convert) noexcept;

  static OrtStatus* ORT_API_CALL CreateAllocatorImpl(
      OrtEp* ep,
      const OrtMemoryInfo* memory_info,
      OrtAllocator** allocator) noexcept;

  static OrtStatus* ORT_API_CALL CreateSyncStreamForDeviceImpl(
      OrtEp* ep,
      const OrtMemoryDevice* memory_device,
      OrtSyncStreamImpl** stream) noexcept;

  // ---- Per-session state --------------------------------------------------
  NeuronEpFactory&    factory_;
  std::string         name_;
  Config              config_;
  const OrtLogger&    logger_;

  NeuronWrapperHandle neuron_handle_{nullptr};
};
