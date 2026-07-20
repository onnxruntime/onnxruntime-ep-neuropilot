// neuron_ep_stream_support.h
// OrtSyncStreamImpl and OrtSyncNotificationImpl for NeuronDummyPluginEP.
// These are stubs suitable for a CPU-based EP.
// A real GPU EP would hold a cudaStream_t / aclrtStream here.

#pragma once

#include "neuron_ep_utils.h"

#include "neuron_ep_factory.h"

// ---------------------------------------------------------------------------
// NeuronStreamImpl
// ---------------------------------------------------------------------------
class NeuronStreamImpl : public OrtSyncStreamImpl, public ApiPtrs {
 public:
  NeuronStreamImpl(NeuronEpFactory& factory, const OrtEp* /*ep*/)
      : ApiPtrs(factory), factory_{&factory} {
    ort_version_supported = ORT_API_VERSION;
    CreateNotification    = CreateNotificationImpl;
    GetHandle             = GetHandleImpl;
    Flush                 = FlushImpl;
    OnSessionRunEnd       = OnSessionRunEndImpl;
    Release               = ReleaseImpl;
  }

 private:
  static OrtStatus* ORT_API_CALL CreateNotificationImpl(OrtSyncStreamImpl* this_ptr,
                                                         OrtSyncNotificationImpl** notification) noexcept;
  static void*      ORT_API_CALL GetHandleImpl(OrtSyncStreamImpl* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL FlushImpl(OrtSyncStreamImpl* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL OnSessionRunEndImpl(OrtSyncStreamImpl* this_ptr) noexcept;
  static void       ORT_API_CALL ReleaseImpl(OrtSyncStreamImpl* this_ptr) noexcept;

  void*              handle_{nullptr};  // would be cudaStream_t for a GPU EP
  NeuronEpFactory*   factory_{nullptr};
};

// ---------------------------------------------------------------------------
// NeuronNotificationImpl
// ---------------------------------------------------------------------------
class NeuronNotificationImpl : public OrtSyncNotificationImpl, public ApiPtrs {
 public:
  explicit NeuronNotificationImpl(const ApiPtrs& apis) : ApiPtrs(apis) {
    ort_version_supported = ORT_API_VERSION;
    Activate              = ActivateImpl;
    WaitOnDevice          = WaitOnDeviceImpl;
    WaitOnHost            = WaitOnHostImpl;
    Release               = ReleaseImpl;
  }

 private:
  static OrtStatus* ORT_API_CALL ActivateImpl(OrtSyncNotificationImpl* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL WaitOnDeviceImpl(OrtSyncNotificationImpl* this_ptr,
                                                   OrtSyncStream* stream) noexcept;
  static OrtStatus* ORT_API_CALL WaitOnHostImpl(OrtSyncNotificationImpl* this_ptr) noexcept;
  static void       ORT_API_CALL ReleaseImpl(OrtSyncNotificationImpl* this_ptr) noexcept;

  void* event_{nullptr};  // would be cudaEvent_t for a GPU EP
};
