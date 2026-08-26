/*
 * neuron_op_capability_abi.h
 *
 * The complete interface between the plugin EP and libneuron_op_capability.so.
 * This header is the ONLY thing the two sides share at build time, and it
 * deliberately includes nothing but <stddef.h>/<stdint.h>.
 *
 * The graph crosses the boundary as serialized GraphProto bytes, parsed by each
 * side with its own protobuf (the wire format is stable across versions). Plain
 * C linkage; no ownership, no exceptions, and no C++ types cross the boundary.
 * Cost: one serialize+parse per GetCapability() call -- not per node.
 *
 * VERSIONING: bump NEURON_CAP_ABI_VERSION for ANY change to the signatures,
 * structs, or semantics below. The EP checks it for exact equality at load time
 * and refuses a mismatched library. NeuronCapParams::struct_size is a second
 * tripwire for a change that forgot the bump (see the struct's comment).
 */

#ifndef NEURON_OP_CAPABILITY_ABI_H_
#define NEURON_OP_CAPABILITY_ABI_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Incompatible-change counter. See VERSIONING above. */
#define NEURON_CAP_ABI_VERSION 1u

/* Name the EP passes to dlopen(). */
#define NEURON_CAP_LIBRARY_NAME "libneuron_op_capability.so"

/*
 * Export marker for the entry points below. The library is built with
 * -fvisibility=hidden, so they must opt back in. NEURON_CAP_BUILD_LIBRARY is
 * defined only by the capability library's build; the EP resolves via dlsym.
 */
#if defined(NEURON_CAP_BUILD_LIBRARY) && (defined(__GNUC__) || defined(__clang__))
#define NEURON_CAP_API __attribute__((visibility("default")))
#else
#define NEURON_CAP_API
#endif

/*
 * Tri-state result for the predicate entry points. ERROR keeps a failing
 * library distinguishable from a genuinely unsupported node; callers treat it
 * as unsupported but log it loudly.
 */
#define NEURON_CAP_ERROR (-1)
#define NEURON_CAP_FALSE (0)
#define NEURON_CAP_TRUE  (1)

/*
 * Opaque handle to a library-side parsed graph. Owned by the library; the
 * layout is not part of the ABI.
 */
typedef struct NeuronCapGraph NeuronCapGraph;

/*
 * Support-check parameters, flattened to POD.
 *
 * struct_size must be set to sizeof(NeuronCapParams) by the caller; the library
 * rejects any other value. It is not a forward-compatibility mechanism -- it
 * catches a struct edit made without the required version bump.
 *
 * device_names and the strings it points to are borrowed for the duration of
 * the call only; the library does not retain them.
 */
typedef struct NeuronCapParams {
  uint32_t struct_size;

  /* Neuron runtime version the model will run against. */
  uint32_t version_major;
  uint32_t version_minor;
  uint32_t version_patch;

  /* Borrowed array of NUL-terminated device names; may be NULL if count == 0. */
  const char* const* device_names;
  size_t device_count;

  int32_t use_nchw; /* NEURON_FLAG_USE_NCHW */
  int32_t use_fp16; /* NEURON_FLAG_USE_FP16 */
} NeuronCapParams;

/*
 * Returns NEURON_CAP_ABI_VERSION as compiled into the library. The EP must call
 * this first and refuse to use the library on mismatch.
 */
NEURON_CAP_API uint32_t NeuronCapGetAbiVersion(void);

/*
 * Parses a serialized onnx::GraphProto. The data is copied; the caller may free
 * it on return. Returns NULL on malformed input (treated as untrusted). Every
 * returned handle must be released with NeuronCapReleaseGraph.
 */
NEURON_CAP_API NeuronCapGraph* NeuronCapCreateGraph(const void* bytes, size_t length);

/* Releases a handle from NeuronCapCreateGraph. NULL is accepted and ignored. */
NEURON_CAP_API void NeuronCapReleaseGraph(NeuronCapGraph* graph);

/*
 * Whether the library has a capability registered for an ONNX op type.
 * NEURON_CAP_FALSE means "no constraints known", which the EP treats as
 * supported -- it is not a rejection. Does not depend on any graph.
 */
NEURON_CAP_API int32_t NeuronCapHasCapability(const char* op_type);

/*
 * Whether the node's unit type (e.g. its QDQ arrangement) is one the capability
 * handles. node_index indexes graph.node() of the GraphProto that produced
 * `graph`. Returns NEURON_CAP_ERROR for an out-of-range index.
 */
NEURON_CAP_API int32_t NeuronCapIsNodeUnitTypeSupported(NeuronCapGraph* graph, int32_t node_index);

/*
 * The full capability check for a node. node_index as above; params must be
 * non-NULL with struct_size == sizeof(NeuronCapParams). Returns
 * NEURON_CAP_ERROR for an out-of-range index or a malformed params.
 */
NEURON_CAP_API int32_t NeuronCapIsNodeSupported(NeuronCapGraph* graph,
                                                int32_t node_index,
                                                const NeuronCapParams* params);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* NEURON_OP_CAPABILITY_ABI_H_ */
