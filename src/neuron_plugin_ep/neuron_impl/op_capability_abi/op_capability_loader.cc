// op_capability_loader.cc
// dlopen-based loader for libneuron_op_capability.so.

#include "op_capability_loader.h"

#include <dlfcn.h>

#include <string>

#include <onnx/onnx_pb.h>

#include "../neuron_types_internal.h"

namespace onnxruntime {
namespace neuron {
namespace op_capability {

namespace {

// Logs which symbol is missing, not just that one was.
template <typename Fn>
bool ResolveSymbol(void* handle, const char* name, Fn& out) {
  dlerror();  // clear any stale error
  void* sym = dlsym(handle, name);
  const char* err = dlerror();
  if (sym == nullptr || err != nullptr) {
    LOGS_NEURON(WARNING) << "libneuron_op_capability.so is missing entry point '" << name
                         << "'" << (err ? std::string(": ") + err : std::string());
    return false;
  }
  out = reinterpret_cast<Fn>(sym);
  return true;
}

}  // namespace

OpCapabilityLoader& OpCapabilityLoader::Instance() {
  static OpCapabilityLoader instance;
  return instance;
}

OpCapabilityLoader::OpCapabilityLoader() {
  LoadLibrary();
}

OpCapabilityLoader::~OpCapabilityLoader() {
  if (lib_handle_) {
    dlclose(lib_handle_);
    lib_handle_ = nullptr;
  }
}

void OpCapabilityLoader::LoadLibrary() {
  lib_handle_ = dlopen(NEURON_CAP_LIBRARY_NAME, RTLD_NOW | RTLD_LOCAL);
  if (!lib_handle_) {
    LOGS_NEURON(WARNING) << "Could not load " << NEURON_CAP_LIBRARY_NAME << ": " << dlerror();
    return;
  }

  const bool resolved =
      ResolveSymbol(lib_handle_, "NeuronCapGetAbiVersion", get_abi_version_) &&
      ResolveSymbol(lib_handle_, "NeuronCapCreateGraph", create_graph_) &&
      ResolveSymbol(lib_handle_, "NeuronCapReleaseGraph", release_graph_) &&
      ResolveSymbol(lib_handle_, "NeuronCapHasCapability", has_capability_) &&
      ResolveSymbol(lib_handle_, "NeuronCapIsNodeUnitTypeSupported",
                    is_node_unit_type_supported_) &&
      ResolveSymbol(lib_handle_, "NeuronCapIsNodeSupported", is_node_supported_);

  if (!resolved) {
    dlclose(lib_handle_);
    lib_handle_ = nullptr;
    return;
  }

  // dlsym resolves by name only, so a mismatched ABI would otherwise bind
  // silently and misbehave at runtime.
  const uint32_t lib_abi = get_abi_version_();
  if (lib_abi != NEURON_CAP_ABI_VERSION) {
    LOGS_NEURON(ERROR) << NEURON_CAP_LIBRARY_NAME << " reports ABI version " << lib_abi
                       << " but this EP was built for " << NEURON_CAP_ABI_VERSION
                       << ". Refusing to use it.";
    dlclose(lib_handle_);
    lib_handle_ = nullptr;
    get_abi_version_ = nullptr;
    create_graph_ = nullptr;
    release_graph_ = nullptr;
    has_capability_ = nullptr;
    is_node_unit_type_supported_ = nullptr;
    is_node_supported_ = nullptr;
    return;
  }

  loaded_ = true;
  LOGS_NEURON(VERBOSE) << "Loaded " << NEURON_CAP_LIBRARY_NAME << " (ABI v" << lib_abi << ").";
}

int32_t OpCapabilityLoader::HasCapability(const std::string& op_type) const {
  if (!loaded_) return NEURON_CAP_ERROR;
  return has_capability_(op_type.c_str());
}

NeuronCapGraph* OpCapabilityLoader::CreateGraph(const void* bytes, size_t length) const {
  if (!loaded_) return nullptr;
  return create_graph_(bytes, length);
}

void OpCapabilityLoader::ReleaseGraph(NeuronCapGraph* graph) const {
  if (!loaded_ || graph == nullptr) return;
  release_graph_(graph);
}

int32_t OpCapabilityLoader::IsNodeUnitTypeSupported(NeuronCapGraph* graph,
                                                    int32_t node_index) const {
  if (!loaded_) return NEURON_CAP_ERROR;
  return is_node_unit_type_supported_(graph, node_index);
}

int32_t OpCapabilityLoader::IsNodeSupported(NeuronCapGraph* graph,
                                            int32_t node_index,
                                            const NeuronCapParams* params) const {
  if (!loaded_) return NEURON_CAP_ERROR;
  return is_node_supported_(graph, node_index, params);
}

// ---------------------------------------------------------------------------
// CapabilityGraphHandle
// ---------------------------------------------------------------------------

CapabilityGraphHandle::CapabilityGraphHandle(const onnx::GraphProto& graph) {
  const auto& loader = OpCapabilityLoader::Instance();
  if (!loader.IsLoaded()) return;

  std::string bytes;
  if (!graph.SerializeToString(&bytes)) {
    // Realistically only happens on a graph exceeding protobuf's 2 GiB limit.
    LOGS_NEURON(ERROR) << "Failed to serialize GraphProto for the op capability library.";
    return;
  }

  handle_ = loader.CreateGraph(bytes.data(), bytes.size());
  if (handle_ == nullptr) {
    LOGS_NEURON(ERROR) << "The op capability library rejected the serialized graph ("
                       << bytes.size() << " bytes).";
  }
}

CapabilityGraphHandle::~CapabilityGraphHandle() {
  Reset();
}

CapabilityGraphHandle::CapabilityGraphHandle(CapabilityGraphHandle&& other) noexcept
    : handle_(other.handle_) {
  other.handle_ = nullptr;
}

CapabilityGraphHandle& CapabilityGraphHandle::operator=(
    CapabilityGraphHandle&& other) noexcept {
  if (this != &other) {
    Reset();
    handle_ = other.handle_;
    other.handle_ = nullptr;
  }
  return *this;
}

void CapabilityGraphHandle::Reset() {
  if (handle_ != nullptr) {
    OpCapabilityLoader::Instance().ReleaseGraph(handle_);
    handle_ = nullptr;
  }
}

}  // namespace op_capability
}  // namespace neuron
}  // namespace onnxruntime
