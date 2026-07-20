// neuron_ep.cc
// NeuronEp implementation.
// Graph-partitioning and compilation are handled by calling NeuronWrapper_*
// functions directly (linked at build time from neuron_impl sources).

#include "neuron_ep.h"

#include <string>
#include <unordered_set>

#include "neuron_ep_factory.h"
#include "neuron_ep_stream_support.h"

// ===========================================================================
// NeuronEp
// ===========================================================================

NeuronEp::NeuronEp(NeuronEpFactory& factory,
                   const std::string& ep_name,
                   const Config& config,
                   const OrtLogger& logger)
    : OrtEp{},
      ApiPtrs{static_cast<const ApiPtrs&>(factory)},
      factory_{factory},
      name_{ep_name},
      config_{config},
      logger_{logger} {
  ort_version_supported = ORT_API_VERSION;

  // Create the per-session Neuron handle.  provider_options carries flags
  // like NEURON_FLAG_USE_FP16, NEURON_FLAG_CPU_DISABLED, etc.
  neuron_handle_ = NeuronWrapper_Create(
      &ort_api, &ep_api, &model_editor_api,
      config_.provider_options);
  if (!neuron_handle_) {
    IGNORE_ORTSTATUS(ort_api.Logger_LogMessage(
        &logger_, ORT_LOGGING_LEVEL_ERROR,
        "NeuronEP: NeuronWrapper_Create returned null",
        ORT_FILE, __LINE__, __FUNCTION__));
  }

  // Populate the C vtable.
  GetName                      = GetNameImpl;
  GetCapability                = GetCapabilityImpl;
  Compile                      = CompileImpl;
  ReleaseNodeComputeInfos      = ReleaseNodeComputeInfosImpl;
  GetPreferredDataLayout       = GetPreferredDataLayoutImpl;
  ShouldConvertDataLayoutForOp = ShouldConvertDataLayoutForOpImpl;
  CreateAllocator              = CreateAllocatorImpl;
  CreateSyncStreamForDevice    = CreateSyncStreamForDeviceImpl;

  IGNORE_ORTSTATUS(ort_api.Logger_LogMessage(
      &logger_, ORT_LOGGING_LEVEL_INFO,
      ("NeuronEp created: " + name_).c_str(),
      ORT_FILE, __LINE__, __FUNCTION__));
}

NeuronEp::~NeuronEp() {
  if (neuron_handle_) {
    NeuronWrapper_Destroy(neuron_handle_);
    neuron_handle_ = nullptr;
  }
}

// ---- GetName ---------------------------------------------------------------
/*static*/
const char* ORT_API_CALL NeuronEp::GetNameImpl(const OrtEp* ep) noexcept {
  return static_cast<const NeuronEp*>(ep)->name_.c_str();
}

// ---- GetCapability ---------------------------------------------------------
// ORT calls this during graph partitioning.  We delegate to the wrapper which
// walks the graph using internal ORT types and registers supported subgraphs
// via ep_api->EpGraphSupportInfo_AddNodesToFuse().
/*static*/
OrtStatus* ORT_API_CALL NeuronEp::GetCapabilityImpl(OrtEp* ep_ptr,
                                                      const OrtGraph* graph,
                                                      OrtEpGraphSupportInfo* support_info) noexcept {
  auto& ep = *static_cast<NeuronEp*>(ep_ptr);

  if (!ep.neuron_handle_) {
    return ep.ort_api.CreateStatus(
        ORT_EP_FAIL,
        "NeuronEP: handle not initialised; cannot determine graph capability.");
  }

  return NeuronWrapper_GetCapability(
      ep.neuron_handle_, &ep.ort_api, &ep.ep_api, graph, support_info);
}

// ---- Compile ---------------------------------------------------------------
// ORT calls this once per fused subgraph after partitioning.  The wrapper
// compiles the subgraph via the Neuron SDK and returns OrtNodeComputeInfo
// objects that ORT invokes at inference time.
/*static*/
OrtStatus* ORT_API_CALL NeuronEp::CompileImpl(OrtEp* ep_ptr,
                                               const OrtGraph** graphs,
                                               const OrtNode** fused_nodes,
                                               size_t count,
                                               OrtNodeComputeInfo** node_compute_infos,
                                               OrtNode** ep_context_nodes) noexcept {
  auto& ep = *static_cast<NeuronEp*>(ep_ptr);

  if (!ep.neuron_handle_) {
    return ep.ort_api.CreateStatus(
        ORT_EP_FAIL,
        "NeuronEP: handle not initialised; cannot compile.");
  }

  return NeuronWrapper_Compile(
      ep.neuron_handle_,
      &ep.ort_api, &ep.ep_api, &ep.model_editor_api,
      graphs, fused_nodes, count,
      node_compute_infos, ep_context_nodes);
}

// ---- ReleaseNodeComputeInfos -----------------------------------------------
// Free the OrtNodeComputeInfo objects that were allocated by Compile().
// The wrapper is responsible for the actual deallocation since it allocated them.
/*static*/
void ORT_API_CALL NeuronEp::ReleaseNodeComputeInfosImpl(OrtEp* ep_ptr,
                                                         OrtNodeComputeInfo** infos,
                                                         size_t num) noexcept {
  auto& ep = *static_cast<NeuronEp*>(ep_ptr);
  if (ep.neuron_handle_) {
    NeuronWrapper_ReleaseNodeComputeInfos(ep.neuron_handle_, infos, num);
  }
}

// ---- GetPreferredDataLayout ------------------------------------------------
// Replaces IExecutionProvider::GetPreferredLayout() from v1.20.2.
// Returns NHWC for Neuron (typical for mobile NPUs).
/*static*/
OrtStatus* ORT_API_CALL NeuronEp::GetPreferredDataLayoutImpl(
    OrtEp* ep_ptr,
    OrtEpDataLayout* preferred_data_layout) noexcept {
  auto& ep = *static_cast<NeuronEp*>(ep_ptr);

  if (ep.neuron_handle_) {
    *preferred_data_layout = static_cast<OrtEpDataLayout>(
        NeuronWrapper_GetPreferredLayout(ep.neuron_handle_));
  } else {
    *preferred_data_layout = OrtEpDataLayout_NHWC;
  }
  return nullptr;
}

// ---- ShouldConvertDataLayoutForOp ------------------------------------------
// Replaces the ORT-core ort_transpose_optimization.cc Softmax patch from
// v1.20.2.  The old patch registered a global handler that returned false for
// Softmax; this vtable function achieves the same effect per-EP without
// patching ORT core.
/*static*/
OrtStatus* ORT_API_CALL NeuronEp::ShouldConvertDataLayoutForOpImpl(
    OrtEp* ep_ptr,
    const char* domain,
    const char* op_type,
    OrtEpDataLayout target_data_layout,
    int* should_convert) noexcept {
  // Spatial ops that require NHWC on Neuron hardware: force NCHW→NHWC.
  // Other ops: return -1 (no opinion) so ORT uses its built-in layout-sensitive
  // list.  Returning 1 for ALL ops causes ORT to apply 4-D permutations to
  // non-4-D tensors (e.g. Gemm weight is rank-2), crashing the transpose
  // optimizer with "Permutation length 4 does not match rank 2".
  static const std::unordered_set<std::string> kForceNHWCOps = {
      "Resize", "Upsample",
      "AveragePool", "MaxPool", "GlobalAveragePool", "GlobalMaxPool",
      "QLinearAveragePool",
      "BatchNormalization",
      "InstanceNormalization",
  };

  if (op_type && kForceNHWCOps.count(op_type)) {
    *should_convert = 1;  // force NCHW→NHWC
    return nullptr;
  }

  auto& ep = *static_cast<NeuronEp*>(ep_ptr);
  if (ep.neuron_handle_) {
    int skip = NeuronWrapper_ShouldSkipLayoutConversion(
        ep.neuron_handle_, domain, op_type,
        static_cast<int>(target_data_layout));
    // skip=1 → explicitly don't convert; skip=0 → no opinion (let ORT decide)
    *should_convert = skip ? 0 : -1;
  } else {
    *should_convert = -1;  // no opinion
  }
  return nullptr;
}

// ---- CreateAllocator -------------------------------------------------------
// Delegates to the factory's shared allocator (malloc/free backed).
// Neuron EP uses CPU-accessible memory so no device-specific allocator needed.
/*static*/
OrtStatus* ORT_API_CALL NeuronEp::CreateAllocatorImpl(OrtEp* ep_ptr,
                                                       const OrtMemoryInfo* memory_info,
                                                       OrtAllocator** allocator) noexcept {
  auto& ep = *static_cast<NeuronEp*>(ep_ptr);
  return ep.factory_.CreateAllocator(&ep.factory_, memory_info, nullptr, allocator);
}

// ---- CreateSyncStreamForDevice (optional) ----------------------------------
/*static*/
OrtStatus* ORT_API_CALL NeuronEp::CreateSyncStreamForDeviceImpl(
    OrtEp* ep_ptr,
    const OrtMemoryDevice* memory_device,
    OrtSyncStreamImpl** stream) noexcept {
  auto& ep = *static_cast<NeuronEp*>(ep_ptr);

  auto mem_type = ep.ep_api.MemoryDevice_GetMemoryType(memory_device);
  if (mem_type != OrtDeviceMemoryType_DEFAULT) {
    return ep.ort_api.CreateStatus(
        ORT_INVALID_ARGUMENT,
        "NeuronEP: stream requested for unsupported memory type.");
  }

  auto s = std::make_unique<NeuronStreamImpl>(ep.factory_, nullptr);
  *stream = s.release();
  return nullptr;
}
