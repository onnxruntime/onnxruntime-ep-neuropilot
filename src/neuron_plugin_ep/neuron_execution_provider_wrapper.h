// neuron_execution_provider_wrapper.h
// C interface implemented by neuron_impl and called directly by NeuronEp.
//
// neuron_impl sources are compiled into the same plugin .so as the EP shell,
// so these are ordinary linked symbols — not dlopen/dlsym.
//
// Design notes:
//   - All types are ORT public C API types (OrtGraph*, OrtNode*, etc.).
//   - neuron_impl uses ort_graph_to_proto.h to convert OrtGraph* → onnx::GraphProto
//     without any reinterpret_cast to ORT internal types.
//   - OrtNodeComputeInfo objects returned by NeuronWrapper_Compile are freed
//     via NeuronWrapper_ReleaseNodeComputeInfos; the EP shell must not free them.

#pragma once

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

typedef void* NeuronWrapperHandle;

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

// Create a Neuron wrapper instance.  All pointer arguments are borrowed
// for the lifetime of the handle.
//
// provider_options  – key/value pairs forwarded to the Neuron SDK.
//   Recognised keys:
//     "NEURON_FLAG_USE_FP16"             – value "1" enables FP16 inference
//     "NEURON_FLAG_USE_NCHW"             – value "1" forces NCHW layout
//     "NEURON_FLAG_CPU_DISABLED"         – value "1" disables CPU fallback
//     "NEURON_FLAG_CPU_ONLY"             – value "1" forces CPU backend only
//     "NEURON_FLAG_MDLA_ONLY"            – value "1" targets MDLA accelerator only
//     "NEURON_FLAG_MIN_GROUP_SIZE"       – minimum subgraph node count
//     "NEURON_FLAG_OPTIMIZATION_STRING"  – Neuron SDK optimisation hint string
//     "NEURON_FLAG_FORCE_SKIP_OPS_STRING"– comma-separated list of ops to skip
//
// Returns nullptr on failure (logs internally).
NeuronWrapperHandle NeuronWrapper_Create(
    const OrtApi*            ort_api,
    const OrtEpApi*          ep_api,
    const OrtModelEditorApi* model_editor_api,
    const OrtKeyValuePairs*  provider_options);

// Destroy a handle returned by NeuronWrapper_Create.
void NeuronWrapper_Destroy(NeuronWrapperHandle handle);

// ---------------------------------------------------------------------------
// Graph partitioning  (called from OrtEp::GetCapability)
// ---------------------------------------------------------------------------

// Walk `graph` and register supported subgraphs via the ep_api.
// For each supported node group, calls:
//   ep_api->EpGraphSupportInfo_AddNodesToFuse(support_info, nodes, n, opts)
OrtStatus* NeuronWrapper_GetCapability(
    NeuronWrapperHandle      handle,
    const OrtApi*            ort_api,
    const OrtEpApi*          ep_api,
    const OrtGraph*          graph,
    OrtEpGraphSupportInfo*   support_info);

// ---------------------------------------------------------------------------
// Compilation  (called from OrtEp::Compile)
// ---------------------------------------------------------------------------

// Compile `count` fused subgraphs into executable kernels.
// Allocates one OrtNodeComputeInfo per fused node and writes them to
// node_compute_infos[0..count-1].
// If ep_context_nodes is non-null, optionally writes EPContext nodes to it.
OrtStatus* NeuronWrapper_Compile(
    NeuronWrapperHandle          handle,
    const OrtApi*                ort_api,
    const OrtEpApi*              ep_api,
    const OrtModelEditorApi*     model_editor_api,
    const OrtGraph* const*       graphs,
    const OrtNode*  const*       fused_nodes,
    size_t                       count,
    OrtNodeComputeInfo**         node_compute_infos,
    OrtNode**                    ep_context_nodes);

// Release OrtNodeComputeInfo objects previously allocated by NeuronWrapper_Compile.
void NeuronWrapper_ReleaseNodeComputeInfos(
    NeuronWrapperHandle  handle,
    OrtNodeComputeInfo** node_compute_infos,
    size_t               count);

// ---------------------------------------------------------------------------
// Layout preferences  (called from OrtEp vtable)
// ---------------------------------------------------------------------------

// Returns the OrtEpDataLayout value the EP prefers.
//   0 = OrtEpDataLayout_NCHW (default)
//   1 = OrtEpDataLayout_NHWC
int NeuronWrapper_GetPreferredLayout(NeuronWrapperHandle handle);

// Returns 1 if layout conversion should be SKIPPED for this op, 0 otherwise.
// Used to suppress layout transformation for ops like Softmax.
// Replaces the ORT-core ort_transpose_optimization.cc patch from v1.20.2.
int NeuronWrapper_ShouldSkipLayoutConversion(
    NeuronWrapperHandle handle,
    const char*         domain,
    const char*         op_type,
    int                 target_layout);

#ifdef __cplusplus
}
#endif
