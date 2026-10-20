// neuron_dmabuf.h
// Registry of dmabuf-backed host buffers the application wants the NPU to read
// or write directly, without the Neuron runtime staging a copy.
//
// ORT's C API has no way to pass a file descriptor with a tensor, so the
// application registers each buffer once (host pointer + fd) and then uses the
// pointer as an ordinary tensor.  When the EP binds a tensor whose data pointer
// lies inside a registered range, it calls NeuronExecution_set{Input,Output}FromMemory
// instead of the host-pointer form.  Unregistered tensors are unaffected.

#pragma once

#include <stddef.h>

struct NeuronMemory;

#define NEURON_EP_EXPORT __attribute__((visibility("default")))

extern "C" {
// Registers the mapping [host_ptr, host_ptr + size) of the dmabuf `fd`.
// `fd` must stay open, and the buffer must stay mapped, until
// NeuronEp_UnregisterDmabuf() returns.  Returns 0 on success.
NEURON_EP_EXPORT int NeuronEp_RegisterDmabuf(const void* host_ptr, size_t size, int fd);

// Drops the registration.  No Run() may be using the buffer.
NEURON_EP_EXPORT void NeuronEp_UnregisterDmabuf(const void* host_ptr);
}

namespace onnxruntime {
namespace neuron {
namespace dmabuf {

// If [ptr, ptr + length) lies inside a registered buffer, returns true and sets
// the NeuronMemory handle and the offset of ptr within it.
bool Lookup(const void* ptr, size_t length, const NeuronMemory** memory, size_t* offset);

}  // namespace dmabuf
}  // namespace neuron
}  // namespace onnxruntime
