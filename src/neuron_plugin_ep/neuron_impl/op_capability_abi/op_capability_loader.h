// op_capability_loader.h
// Plugin EP version – singleton that loads libneuron_op_capability.so at runtime.
//
// No ORT LoggingManager dependency: the .so uses stateless LOGS_NEURON macros
// (NeuronLogStream to stderr), so no InitOpCapabilityLogging call is needed.

#pragma once

#include <string>

#include "op_capability_interface.h"

namespace onnxruntime {
namespace neuron {
namespace op_capability {

// Singleton that loads libneuron_op_capability.so via dlopen and provides
// access to the per-op capability checkers registered therein.
class OpCapabilityLoader {
 public:
  static OpCapabilityLoader& Instance();

  // Returns true if libneuron_op_capability.so was successfully loaded.
  bool IsLoaded() const { return loaded_; }

  // Returns the capability for the given ONNX op_type, or nullptr if
  // no capability was registered (meaning: use built-in BaseOpBuilder defaults).
  const IOpCapability* GetCapability(const std::string& op_type) const;

  // Prevent copies
  OpCapabilityLoader(const OpCapabilityLoader&) = delete;
  OpCapabilityLoader& operator=(const OpCapabilityLoader&) = delete;

 private:
  OpCapabilityLoader();
  ~OpCapabilityLoader();

  void LoadLibrary();

  void* lib_handle_{nullptr};
  OpCapabilityRegistrations registrations_;
  bool loaded_{false};
};

}  // namespace op_capability
}  // namespace neuron
}  // namespace onnxruntime
