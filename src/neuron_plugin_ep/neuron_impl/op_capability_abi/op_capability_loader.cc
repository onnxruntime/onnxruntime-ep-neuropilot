// op_capability_loader.cc
// Plugin EP version – dlopen-based loader for libneuron_op_capability.so.
//
// Differences from local (in-tree) version:
//   - No InitOpCapabilityLogging call: the .so uses LOGS_NEURON (NeuronLogStream
//     to stderr), which does not require ORT LoggingManager initialization.
//   - Uses LOGS_NEURON instead of LOGS_DEFAULT_CATEGORY.

#include "op_capability_loader.h"

#include <dlfcn.h>

#include "../neuron_types_internal.h"

namespace onnxruntime {
namespace neuron {
namespace op_capability {

OpCapabilityLoader& OpCapabilityLoader::Instance() {
  static OpCapabilityLoader instance;
  return instance;
}

OpCapabilityLoader::OpCapabilityLoader() {
  LoadLibrary();
}

OpCapabilityLoader::~OpCapabilityLoader() {
  // Clear registrations before closing the library so the capability objects'
  // vtables (which live in the shared library) are not called after dlclose.
  registrations_.capabilities.clear();
  if (lib_handle_) {
    dlclose(lib_handle_);
    lib_handle_ = nullptr;
  }
}

void OpCapabilityLoader::LoadLibrary() {
  lib_handle_ = dlopen("libneuron_op_capability.so", RTLD_NOW);
  if (!lib_handle_) {
    LOGS_NEURON(WARNING) << "Could not load libneuron_op_capability.so: " << dlerror()
                         << ". Falling back to embedded capability logic.";
    return;
  }

  auto create_fn = reinterpret_cast<CreateOpCapabilityRegistrationsFn>(
      dlsym(lib_handle_, "CreateOpCapabilityRegistrations"));
  if (!create_fn) {
    LOGS_NEURON(WARNING) << "Could not find CreateOpCapabilityRegistrations symbol: " << dlerror()
                         << ". Falling back to embedded capability logic.";
    dlclose(lib_handle_);
    lib_handle_ = nullptr;
    return;
  }

  create_fn(registrations_);
  loaded_ = true;
  LOGS_NEURON(VERBOSE) << "Loaded libneuron_op_capability.so with "
                       << registrations_.capabilities.size() << " op capability(s).";
}

const IOpCapability* OpCapabilityLoader::GetCapability(const std::string& op_type) const {
  auto it = registrations_.capabilities.find(op_type);
  if (it != registrations_.capabilities.end()) {
    return it->second.get();
  }
  return nullptr;
}

}  // namespace op_capability
}  // namespace neuron
}  // namespace onnxruntime
