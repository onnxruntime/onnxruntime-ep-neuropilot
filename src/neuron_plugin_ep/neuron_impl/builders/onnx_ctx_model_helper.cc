// onnx_ctx_model_helper.cc - Plugin EP version.
#include "onnx_ctx_model_helper.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <vector>

#include "helper.h"

namespace onnxruntime {
namespace neuron {

namespace {

std::string ToLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s;
}

}  // namespace

bool IsOwnedEpContextSource(const std::string& source) {
  const std::string s = ToLower(source);
  return s == ToLower(EP_CONTEXT_SOURCE_NAME) || s == "neuropilot" ||
         // in-tree Neuron EP, before the plugin rename
         s == "neuron" || s == "neuronexecutionprovider";
}

std::string GetOrtNodeAttrString(const OrtApi& ort_api, const OrtNode& node,
                                 const char* name, const std::string& def) {
  const OrtOpAttr* attr = nullptr;
  Ort::Status st{ort_api.Node_GetAttributeByName(&node, name, &attr)};
  if (!st.IsOK() || attr == nullptr) return def;

  // ReadOpAttr reports the required byte count in `out` when the buffer is too
  // small, so size the buffer from a deliberately failing probe.  The probe
  // passes a real (if zero-length) buffer rather than nullptr: for an empty
  // string ReadOpAttr takes the success path and memcpy's 0 bytes into it.
  char probe_buf[1];
  size_t needed = 0;
  Ort::Status probe{
      ort_api.ReadOpAttr(attr, ORT_OP_ATTR_STRING, probe_buf, 0, &needed)};
  if (needed == 0) return probe.IsOK() ? std::string{} : def;

  std::vector<char> buf(needed);
  Ort::Status read{
      ort_api.ReadOpAttr(attr, ORT_OP_ATTR_STRING, buf.data(), buf.size(), &needed)};
  if (!read.IsOK()) return def;
  return std::string(buf.data(), needed);
}

bool IsValidDlaNode(const OrtApi& ort_api, const OrtNode& node) {
  const char* op_type = nullptr;
  Ort::Status st{ort_api.Node_GetOperatorType(&node, &op_type)};
  if (!st.IsOK() || op_type == nullptr || EP_CONTEXT_OP != op_type) return false;

  if (!IsOwnedEpContextSource(GetOrtNodeAttrString(ort_api, node, SOURCE.c_str())))
    return false;

  // Presence only -- never read the payload here, it can be many megabytes.
  const OrtOpAttr* payload = nullptr;
  Ort::Status ps{
      ort_api.Node_GetAttributeByName(&node, EP_CACHE_CONTEXT.c_str(), &payload)};
  return ps.IsOK() && payload != nullptr;
}

bool GraphHasDlaNode(const OrtApi& ort_api, const OrtGraph& graph) {
  size_t num_nodes = 0;
  Ort::Status st{ort_api.Graph_GetNumNodes(&graph, &num_nodes)};
  if (!st.IsOK() || num_nodes == 0) return false;

  std::vector<const OrtNode*> nodes(num_nodes, nullptr);
  Ort::Status gs{ort_api.Graph_GetNodes(&graph, nodes.data(), nodes.size())};
  if (!gs.IsOK()) return false;

  for (const OrtNode* n : nodes) {
    if (n != nullptr && IsValidDlaNode(ort_api, *n)) return true;
  }
  return false;
}

bool IsAdapterFormattedDlaVersion(const NeuronRuntimeVersion& v) {
  return GreaterEqualVersion(v, {4, 3, 0});
}

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

std::filesystem::path GetContextModelPath(const std::string& user_context_cache_path,
                                           const std::string& source_model_path) {
  if (!user_context_cache_path.empty()) return user_context_cache_path;
  return source_model_path;  // ORT appends _ctx.onnx itself
}

namespace {

// Owns the OrtOpAttr list while it is being assembled.  CreateNode takes
// ownership of every attribute it is handed (it copies the AttributeProto, then
// releases the original and nulls the array slot), so anything still owned here
// when CreateNode did not run must be released by us.
class OpAttrList {
 public:
  explicit OpAttrList(const OrtApi& ort_api) : ort_api_(ort_api) {}
  ~OpAttrList() {
    for (OrtOpAttr* a : attrs_)
      if (a != nullptr) ort_api_.ReleaseOpAttr(a);
  }
  OpAttrList(const OpAttrList&) = delete;
  OpAttrList& operator=(const OpAttrList&) = delete;

  NeuronStatus AddString(const std::string& name, const std::string& value) {
    OrtOpAttr* a = nullptr;
    Ort::Status st{ort_api_.CreateOpAttr(name.c_str(), value.data(),
                                         static_cast<int>(value.size()),
                                         ORT_OP_ATTR_STRING, &a)};
    NEURON_RETURN_IF_NOT(st.IsOK() && a != nullptr,
                         MakeNeuronString("CreateOpAttr failed for ", name));
    attrs_.push_back(a);
    return NeuronStatus::OK();
  }

  NeuronStatus AddInt(const std::string& name, int64_t value) {
    OrtOpAttr* a = nullptr;
    Ort::Status st{
        ort_api_.CreateOpAttr(name.c_str(), &value, 1, ORT_OP_ATTR_INT, &a)};
    NEURON_RETURN_IF_NOT(st.IsOK() && a != nullptr,
                         MakeNeuronString("CreateOpAttr failed for ", name));
    attrs_.push_back(a);
    return NeuronStatus::OK();
  }

  OrtOpAttr** data() { return attrs_.data(); }
  size_t size() const { return attrs_.size(); }

  // CreateNode nulls each slot it consumed; drop them so ~OpAttrList is a no-op.
  void ReleaseOwnership() { attrs_.clear(); }

 private:
  const OrtApi& ort_api_;
  std::vector<OrtOpAttr*> attrs_;
};

}  // namespace

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
    bool context_embed_mode) {

  NEURON_RETURN_IF_NOT(ep_context_nodes != nullptr,
                       "CreateEpContextNodes: null output array");
  const size_t count = partition_names.size();
  NEURON_RETURN_IF_NOT(
      input_names.size() == count && output_names.size() == count &&
          compiled_dla_paths.size() == count,
      "CreateEpContextNodes: parallel arrays differ in length");

  const bool is_adapter_dla = IsAdapterFormattedDlaVersion(neuron_version);

  for (size_t i = 0; i < count; ++i) {
    const std::string& graph_name = partition_names[i];
    const std::vector<std::string>& in_names  = input_names[i];
    const std::vector<std::string>& out_names = output_names[i];

    std::string dla_buffer;
    NEURON_RETURN_IF_ERROR(
        ReadBinaryFileToString(compiled_dla_paths[i], dla_buffer));
    NEURON_RETURN_IF_NOT(!dla_buffer.empty(),
                         MakeNeuronString("compiled DLA is empty: ",
                                          compiled_dla_paths[i].string()));

    // embed_mode=0 stores each blob beside the context model and keeps only the
    // file name in the attribute, so the set stays relocatable together.
    std::string payload_attr;
    if (context_embed_mode) {
      payload_attr = std::move(dla_buffer);
    } else {
      NEURON_RETURN_IF_NOT(
          !context_model_path.empty(),
          "ep.context_embed_mode=0 needs a model path to place the DLA sidecar "
          "next to; set ep.context_file_path when compiling from memory.");
      std::filesystem::path bin_path = context_model_path;
      bin_path.replace_extension();
      bin_path += "_neuron_" + std::to_string(i) + ".bin";
      NEURON_RETURN_IF_ERROR(WriteBinaryStringToFile(bin_path, dla_buffer));
      payload_attr = bin_path.filename().string();
      LOGS_NEURON(INFO) << "NeuronEP: wrote DLA sidecar " << bin_path.string()
                        << " (" << dla_buffer.size() << " bytes)";
    }

    OpAttrList attrs(ort_api);
    // Every node carries its own DLA, so every node is a main context
    // (main_context defaults to 1 and is left unwritten).
    NEURON_RETURN_IF_ERROR(attrs.AddString(EP_CACHE_CONTEXT, payload_attr));
    NEURON_RETURN_IF_ERROR(attrs.AddInt(EMBED_MODE, context_embed_mode ? 1 : 0));
    NEURON_RETURN_IF_ERROR(
        attrs.AddString(EP_SDK_VER, GetNeuronVersionDescription(neuron_version)));
    NEURON_RETURN_IF_ERROR(attrs.AddString(PARTITION_NAME, graph_name));
    NEURON_RETURN_IF_ERROR(attrs.AddString(SOURCE, EP_CONTEXT_SOURCE_NAME));
    if (is_adapter_dla) {
      NEURON_RETURN_IF_ERROR(attrs.AddInt(ADAPTER_DLA, 1));
    }

    std::vector<const char*> in_cstr, out_cstr;
    in_cstr.reserve(in_names.size());
    out_cstr.reserve(out_names.size());
    for (const auto& n : in_names) in_cstr.push_back(n.c_str());
    for (const auto& n : out_names) out_cstr.push_back(n.c_str());

    Ort::Status cs{model_editor_api.CreateNode(
        EP_CONTEXT_OP.c_str(), "com.microsoft", graph_name.c_str(),
        in_cstr.data(), in_cstr.size(), out_cstr.data(), out_cstr.size(),
        attrs.data(), attrs.size(), &ep_context_nodes[i])};
    NEURON_RETURN_IF_NOT(cs.IsOK(),
                         MakeNeuronString("CreateNode failed for EPContext node ",
                                          graph_name, ": ", cs.GetErrorMessage()));
    // CreateNode consumed the attributes on success.
    attrs.ReleaseOwnership();
  }

  return NeuronStatus::OK();
}

} // namespace neuron
} // namespace onnxruntime
