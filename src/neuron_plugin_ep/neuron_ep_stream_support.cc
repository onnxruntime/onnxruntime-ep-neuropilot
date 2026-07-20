// neuron_ep_stream_support.cc
// No-op stream/notification implementations for the CPU-based dummy EP.

#include "neuron_ep_stream_support.h"
#include "neuron_ep_factory.h"

// ---------------------------------------------------------------------------
// NeuronStreamImpl
// ---------------------------------------------------------------------------

/*static*/
OrtStatus* ORT_API_CALL NeuronStreamImpl::CreateNotificationImpl(
    OrtSyncStreamImpl* this_ptr,
    OrtSyncNotificationImpl** notification) noexcept {
  auto& impl = *static_cast<NeuronStreamImpl*>(this_ptr);
  *notification = std::make_unique<NeuronNotificationImpl>(
      static_cast<const ApiPtrs&>(*impl.factory_)).release();
  return nullptr;
}

/*static*/
void* ORT_API_CALL NeuronStreamImpl::GetHandleImpl(OrtSyncStreamImpl* this_ptr) noexcept {
  return static_cast<NeuronStreamImpl*>(this_ptr)->handle_;  // nullptr for CPU
}

/*static*/
OrtStatus* ORT_API_CALL NeuronStreamImpl::FlushImpl(OrtSyncStreamImpl* /*this_ptr*/) noexcept {
  // CPU EP: nothing to flush.
  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL NeuronStreamImpl::OnSessionRunEndImpl(
    OrtSyncStreamImpl* /*this_ptr*/) noexcept {
  // If we had an arena we would call arena->ResetChunksUsingStream() here.
  // We use a simple malloc allocator so nothing to do.
  return nullptr;
}

/*static*/
void ORT_API_CALL NeuronStreamImpl::ReleaseImpl(OrtSyncStreamImpl* this_ptr) noexcept {
  delete static_cast<NeuronStreamImpl*>(this_ptr);
}

// ---------------------------------------------------------------------------
// NeuronNotificationImpl
// ---------------------------------------------------------------------------

/*static*/
OrtStatus* ORT_API_CALL NeuronNotificationImpl::ActivateImpl(
    OrtSyncNotificationImpl* /*this_ptr*/) noexcept {
  // GPU EP: cudaEventRecord / aclrtRecordEvent
  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL NeuronNotificationImpl::WaitOnDeviceImpl(
    OrtSyncNotificationImpl* /*this_ptr*/,
    OrtSyncStream* /*stream*/) noexcept {
  // GPU EP: cudaStreamWaitEvent / aclrtStreamWaitEvent
  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL NeuronNotificationImpl::WaitOnHostImpl(
    OrtSyncNotificationImpl* /*this_ptr*/) noexcept {
  // GPU EP: cudaEventSynchronize / aclrtSynchronizeEvent
  return nullptr;
}

/*static*/
void ORT_API_CALL NeuronNotificationImpl::ReleaseImpl(
    OrtSyncNotificationImpl* this_ptr) noexcept {
  delete static_cast<NeuronNotificationImpl*>(this_ptr);
}
