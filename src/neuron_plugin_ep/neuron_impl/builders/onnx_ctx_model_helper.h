// onnx_ctx_model_helper.h - Plugin EP version.
// EPContext (DLA caching) helpers using proto types instead of ORT internals.
#pragma once
#include <filesystem>
#include <string>
#include <vector>
#include <onnx/onnx_pb.h>
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT
#include "../neuron_types_internal.h"
#include "../neuron_lib/neuron_types.h"
#include "../neuron_graph_context.h"

namespace onnxruntime {
namespace neuron {

// EPContext node attribute constants (same as original).
inline const std::string EP_CONTEXT_OP  = "EPContext";
// Value written into the "source" attribute, and the name the reader matches
// first.  Keep in sync with NeuronEpFactory::kEpName.
inline const std::string EP_CONTEXT_SOURCE_NAME = "NeuroPilotExecutionProvider";
inline const std::string SOURCE         = "source";
inline const std::string EP_CACHE_CONTEXT = "ep_cache_context";
inline const std::string MAIN_CONTEXT   = "main_context";
inline const std::string EMBED_MODE     = "embed_mode";
inline const std::string EP_SDK_VER     = "ep_sdk_version";
inline const std::string PARTITION_NAME = "partition_name";
inline const std::string ADAPTER_DLA    = "adapter_dla";

// DLA extension names / codes.
inline constexpr const char* RESTORE_DLA_EXTENSION_NAME = "com.mediatek.compiled_network";
inline constexpr uint16_t ADAPTER_EXTENSION_HINT_OPERATION_TYPE_RESTORE     = 0x0000;
inline constexpr uint16_t ADAPTER_EXTENSION_HINT_OPERAND_RESTORE_DATA       = 0x0100;
inline constexpr uint16_t ADAPTER_EXTENSION_HINT_OPERAND_RESTORE_DATA_NO_COPY = 0x0200;

// Returns true when the node is an EPContext node this EP owns, i.e. it is an
// EPContext node whose "source" names this EP and which actually carries a
// cache payload.  Checking "source" matters: a model compiled for several EPs
// holds one EPContext node per EP, and claiming another EP's node would fail
// later in the restore with an opaque Neuron error.
bool IsValidDlaNode(const onnx::NodeProto& node);

// True when `source` names this EP.  Accepts the legacy in-tree Neuron EP names
// so context models produced before the plugin rename still load.
bool IsOwnedEpContextSource(const std::string& source);

// Which DLA format this Neuron runtime produces and consumes.
//
// >= 4.3.0 uses the adapter ("compiled network") format: stored with
// NeuronCompilation_storeCompiledNetwork and restored whole with
// NeuronModel_restoreFromCompiledNetwork.
//
// Older runtimes emit a DLA file via the "--dla-file" optimization string,
// restored by building a one-operation model around the
// com.mediatek.compiled_network RESTORE extension.
bool IsAdapterFormattedDlaVersion(const NeuronRuntimeVersion& v);

// Returns true when the node uses adapter-formatted (compiled network) DLA.
bool IsAdapterFormattedDlaNode(const onnx::NodeProto& node);

bool GraphHasDlaNode(const NeuronGraphContext& ctx);
bool GraphHasAdapterFormattedDlaNode(const NeuronGraphContext& ctx);

// Extracts the DLA blob an EPContext node carries.
// embed_mode=1: ep_cache_context is the blob itself.
// embed_mode=0: ep_cache_context is a file name relative to `ep_context_dir`
//               (the directory holding the context .onnx).
NeuronStatus GetDlaPayload(const onnx::NodeProto& node,
                            const std::filesystem::path& ep_context_dir,
                            std::string& payload);

// --- OrtNode/OrtGraph overloads -------------------------------------------
// GetCapability must decide whether this is a context model *before* paying for
// an OrtGraphToProto conversion, which would copy the whole (multi-MB) DLA blob
// out of the ep_cache_context attribute just to throw it away.  These read the
// handful of small attributes needed for that decision straight off the OrtNode.

// Reads a string attribute; returns `def` when the attribute is absent.
std::string GetOrtNodeAttrString(const OrtApi& ort_api, const OrtNode& node,
                                 const char* name, const std::string& def = "");

bool IsValidDlaNode(const OrtApi& ort_api, const OrtNode& node);
bool GraphHasDlaNode(const OrtApi& ort_api, const OrtGraph& graph);

std::filesystem::path GetEpContextPath(bool is_dla_model,
                                        const std::string& user_context_cache_path,
                                        const std::string& model_path);

// Builds one EPContext OrtNode per fused node, writing them to
// ep_context_nodes[0..count-1].  ORT owns the returned nodes and assembles the
// context model itself; the EP only supplies the nodes.
//
// Neuron compiles each partition independently, so every EPContext node
// carries its own DLA and is its own main context.  With embed_mode=0 each
// blob is written to a sidecar file next to `context_model_path` and only its
// file name is stored in the attribute.
//
// ORT requires an EPContext node for *every* fused node when context
// generation is enabled -- returning a null entry fails the session with
// "OrtEp::Compile() returned a NULL EPContext node".
// partition_names/input_names/output_names are parallel arrays, one entry per
// fused node.  The I/O names must be the *model's* real inputs and outputs, not
// the fused node's raw inputs: those also list every initializer, and the
// restore operation would then be handed every weight tensor as an operand.
NeuronStatus CreateEpContextNodes(
    const OrtApi& ort_api,
    const OrtModelEditorApi& model_editor_api,
    const std::vector<std::string>& partition_names,
    const std::vector<std::vector<std::string>>& input_names,
    const std::vector<std::vector<std::string>>& output_names,
    OrtNode** ep_context_nodes,
    const NeuronRuntimeVersion& neuron_version,
    const std::vector<std::filesystem::path>& compiled_dla_paths,
    const std::filesystem::path& context_model_path,
    bool context_embed_mode);

// Resolves where ORT will write the context model, which is what the sidecar
// .bin is placed next to: the user's ep.context_file_path when set, otherwise
// the source model path (ORT appends _ctx.onnx itself).
std::filesystem::path GetContextModelPath(const std::string& user_context_cache_path,
                                           const std::string& source_model_path);

} // namespace neuron
} // namespace onnxruntime
