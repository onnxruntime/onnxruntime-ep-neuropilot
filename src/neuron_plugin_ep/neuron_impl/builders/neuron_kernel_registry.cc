// neuron_kernel_registry.cc - Plugin EP version stub.
#include "neuron_kernel_registry.h"
namespace onnxruntime {
namespace neuron {
OrtStatus* CreateKernelRegistry(const char* /*ep_name*/, void* /*create_kernel_state*/,
                                  OrtKernelRegistry** out_kernel_registry) {
  // Plugin EP does not use a kernel registry; all execution is via OrtNodeComputeInfo.
  *out_kernel_registry = nullptr;
  return nullptr;
}
} // namespace neuron
} // namespace onnxruntime
