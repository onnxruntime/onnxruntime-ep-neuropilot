// neuron_kernel_registry.h - Plugin EP version.
// Kernel registry for MemcpyFromHost/MemcpyToHost.
// In the plugin EP, kernel registration goes through OrtEpApi.
#pragma once
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT
namespace onnxruntime {
namespace neuron {
// Returns nullptr (plugin EP handles kernels via OrtNodeComputeInfo, not kernel registry).
// This stub exists for build compatibility.
OrtStatus* CreateKernelRegistry(const char* ep_name, void* create_kernel_state,
                                  OrtKernelRegistry** out_kernel_registry);
} // namespace neuron
} // namespace onnxruntime
