// memcpy_kernel.cc – Plugin EP version.
//
// The MemcpyFromHost / MemcpyToHost kernels are only needed for the integrated
// (IExecutionProvider-based) EP and its kernel registry.  In the plugin EP
// architecture, memory copies are handled by the ORT runtime via the public
// data-transfer API (neuron_ep_data_transfer.cc).
//
// This file is kept as an empty translation unit so that the NEURON_IMPL_DIR
// glob picks it up without compile errors.

namespace onnxruntime {
namespace neuron {
}  // namespace neuron
}  // namespace onnxruntime
