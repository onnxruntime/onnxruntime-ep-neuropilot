// neuron_dmabuf.cc
// See neuron_dmabuf.h.

#include "neuron_dmabuf.h"

#include <sys/mman.h>

#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>

#include "neuron_implementation.h"

namespace {

struct Entry {
  size_t size;
  NeuronMemory* memory;
};

std::mutex g_mu;
std::map<uintptr_t, Entry> g_entries;  // keyed by start address
// Lets Lookup() skip the lock when nothing is registered, which is the case for
// every application that does not use dmabuf.
std::atomic<size_t> g_count{0};

}  // namespace

extern "C" int NeuronEp_RegisterDmabuf(const void* host_ptr, size_t size, int fd) {
  const NeuronApi* neuron = NeuronApiImplementation();
  if (!neuron || !neuron->NeuronMemory_createFromFd || !host_ptr || size == 0 || fd < 0) return -1;

  NeuronMemory* memory = nullptr;
  if (neuron->NeuronMemory_createFromFd(size, PROT_READ | PROT_WRITE, fd, 0, &memory) != NEURON_NO_ERROR ||
      !memory) {
    return -2;
  }
  std::lock_guard<std::mutex> lock(g_mu);
  const uintptr_t key = reinterpret_cast<uintptr_t>(host_ptr);
  auto it = g_entries.find(key);
  if (it != g_entries.end()) {  // re-registration replaces the old handle
    neuron->NeuronMemory_free(it->second.memory);
    it->second = Entry{size, memory};
  } else {
    g_entries.emplace(key, Entry{size, memory});
    g_count.fetch_add(1, std::memory_order_release);
  }
  return 0;
}

extern "C" void NeuronEp_UnregisterDmabuf(const void* host_ptr) {
  const NeuronApi* neuron = NeuronApiImplementation();
  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_entries.find(reinterpret_cast<uintptr_t>(host_ptr));
  if (it == g_entries.end()) return;
  if (neuron && neuron->NeuronMemory_free) neuron->NeuronMemory_free(it->second.memory);
  g_entries.erase(it);
  g_count.fetch_sub(1, std::memory_order_release);
}

namespace onnxruntime {
namespace neuron {
namespace dmabuf {

bool Lookup(const void* ptr, size_t length, const NeuronMemory** memory, size_t* offset) {
  if (g_count.load(std::memory_order_acquire) == 0) return false;
  const uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_entries.upper_bound(p);
  if (it == g_entries.begin()) return false;
  --it;
  const uintptr_t start = it->first;
  if (p < start || p + length > start + it->second.size) return false;
  *memory = it->second.memory;
  *offset = p - start;
  return true;
}

}  // namespace dmabuf
}  // namespace neuron
}  // namespace onnxruntime
