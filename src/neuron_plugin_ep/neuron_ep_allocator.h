// neuron_ep_allocator.h
// Simple malloc/free based OrtAllocator.
// For a production EP you would replace this with a device-specific allocator
// (e.g. cudaMalloc), optionally wrapped in an arena.

#pragma once

#include "neuron_ep_utils.h"
#include <cstdlib>
#include <memory>
#include <string>
#include <sstream>

// ---------------------------------------------------------------------------
// BaseAllocator – adds virtual destructor so derived types can be deleted
// through a base pointer.
// ---------------------------------------------------------------------------
struct BaseAllocator : OrtAllocator {
  virtual ~BaseAllocator() = default;
};

using AllocatorUniquePtr = std::unique_ptr<BaseAllocator>;

// ---------------------------------------------------------------------------
// CustomAllocator – thin wrapper around malloc / free.
// ---------------------------------------------------------------------------
struct CustomAllocator : BaseAllocator {
  CustomAllocator(const OrtMemoryInfo* mem_info, const ApiPtrs& apis)
      : memory_info_{mem_info}, apis_{apis} {
    version      = ORT_API_VERSION;
    Alloc        = AllocImpl;
    Free         = FreeImpl;
    Info         = InfoImpl;
    Reserve      = AllocImpl;   // no special reserve logic needed
    GetStats     = GetStatsImpl;
    AllocOnStream = nullptr;    // synchronous allocator – no stream needed
  }

  static void* ORT_API_CALL AllocImpl(OrtAllocator* this_, size_t size) {
    auto* self = static_cast<CustomAllocator*>(this_);
    ++self->num_allocs_;
    if (size > self->max_alloc_size_) self->max_alloc_size_ = size;
    return std::malloc(size);
  }

  static void ORT_API_CALL FreeImpl(OrtAllocator* /*this_*/, void* p) {
    std::free(p);
  }

  static const OrtMemoryInfo* ORT_API_CALL InfoImpl(const OrtAllocator* this_) {
    return static_cast<const CustomAllocator*>(this_)->memory_info_;
  }

  static OrtStatus* ORT_API_CALL GetStatsImpl(const OrtAllocator* this_,
                                               OrtKeyValuePairs** out) noexcept {
    const auto* self = static_cast<const CustomAllocator*>(this_);
    OrtKeyValuePairs* kvps = nullptr;
    self->apis_.ort_api.CreateKeyValuePairs(&kvps);
    self->apis_.ort_api.AddKeyValuePair(kvps, "NumAllocs",
                                        std::to_string(self->num_allocs_).c_str());
    self->apis_.ort_api.AddKeyValuePair(kvps, "MaxAllocSize",
                                        std::to_string(self->max_alloc_size_).c_str());
    *out = kvps;
    return nullptr;
  }

 private:
  const OrtMemoryInfo* memory_info_;
  const ApiPtrs        apis_;
  int64_t              num_allocs_    = 0;
  int64_t              max_alloc_size_ = 0;
};
