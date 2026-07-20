// onnx_ctx_model_helper.cc - Plugin EP version.
#include "onnx_ctx_model_helper.h"
#include <filesystem>

namespace onnxruntime {
namespace neuron {

bool IsAdapterFormattedDlaNode(const onnx::NodeProto& node) {
  if (node.op_type() != EP_CONTEXT_OP) return false;
  for (const auto& attr : node.attribute()) {
    if (attr.name() == ADAPTER_DLA && attr.i() != 0) return true;
  }
  return false;
}

bool GraphHasDlaNode(const NeuronGraphContext& ctx) {
  for (int i = 0; i < ctx.graph.node_size(); ++i) {
    if (IsValidDlaNode(ctx.graph.node(i))) return true;
  }
  return false;
}

bool GraphHasAdapterFormattedDlaNode(const NeuronGraphContext& ctx) {
  for (int i = 0; i < ctx.graph.node_size(); ++i) {
    if (IsAdapterFormattedDlaNode(ctx.graph.node(i))) return true;
  }
  return false;
}

std::filesystem::path GetEpContextPath(bool /*is_dla_model*/,
                                        const std::string& user_context_cache_path,
                                        const std::string& model_path) {
  if (!user_context_cache_path.empty()) return user_context_cache_path;
  std::filesystem::path p(model_path);
  return p.replace_extension(".neuron_ep_context");
}

NeuronStatus MakeEpContextModel(
    const OrtModelEditorApi* /*model_editor_api*/,
    const NeuronRuntimeVersion& /*neuron_version*/,
    const std::filesystem::path& /*compiled_dla_path*/,
    const std::filesystem::path& /*context_cache_path*/,
    bool /*context_embed_mode*/) {
  // TODO: implement EP context caching via OrtModelEditorApi.
  return NeuronStatus::OK();
}

} // namespace neuron
} // namespace onnxruntime
