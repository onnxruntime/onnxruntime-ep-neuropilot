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
static const std::string EP_CONTEXT_OP  = "EPContext";
static const std::string SOURCE         = "source";
static const std::string EP_CACHE_CONTEXT = "ep_cache_context";
static const std::string MAIN_CONTEXT   = "main_context";
static const std::string EMBED_MODE     = "embed_mode";
static const std::string EP_SDK_VER     = "ep_sdk_version";
static const std::string PARTITION_NAME = "partition_name";
static const std::string ADAPTER_DLA    = "adapter_dla";

// DLA extension names / codes.
static const char* RESTORE_DLA_EXTENSION_NAME = "com.mediatek.compiled_network";
static const uint16_t ADAPTER_EXTENSION_HINT_OPERATION_TYPE_RESTORE     = 0x0000;
static const uint16_t ADAPTER_EXTENSION_HINT_OPERAND_RESTORE_DATA       = 0x0100;
static const uint16_t ADAPTER_EXTENSION_HINT_OPERAND_RESTORE_DATA_NO_COPY = 0x0200;

// Returns true when the node is an EPContext node carrying DLA content.
inline bool IsValidDlaNode(const onnx::NodeProto& node) {
  return node.op_type() == EP_CONTEXT_OP;
}

// Returns true when the node uses adapter-formatted (compiled network) DLA.
bool IsAdapterFormattedDlaNode(const onnx::NodeProto& node);

bool GraphHasDlaNode(const NeuronGraphContext& ctx);
bool GraphHasAdapterFormattedDlaNode(const NeuronGraphContext& ctx);

std::filesystem::path GetEpContextPath(bool is_dla_model,
                                        const std::string& user_context_cache_path,
                                        const std::string& model_path);

NeuronStatus MakeEpContextModel(
    const OrtModelEditorApi* model_editor_api,
    const NeuronRuntimeVersion& neuron_version,
    const std::filesystem::path& compiled_dla_path,
    const std::filesystem::path& context_cache_path,
    bool context_embed_mode);

} // namespace neuron
} // namespace onnxruntime
