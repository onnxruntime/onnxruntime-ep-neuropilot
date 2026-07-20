// neuron_ep_data_transfer.h
// OrtDataTransferImpl: tells ORT how to copy tensors to/from this EP's device.

#pragma once

#include "neuron_ep_utils.h"

struct NeuronDataTransfer : OrtDataTransferImpl, ApiPtrs {
  NeuronDataTransfer(ApiPtrs api_ptrs, const OrtMemoryDevice* device)
      : ApiPtrs(api_ptrs), device_mem_{device} {
    CanCopy     = CanCopyImpl;
    CopyTensors = CopyTensorsImpl;
    Release     = ReleaseImpl;
  }

  // Returns true if this object can handle a copy between src and dst devices.
  static bool ORT_API_CALL CanCopyImpl(const OrtDataTransferImpl* this_ptr,
                                       const OrtMemoryDevice* src,
                                       const OrtMemoryDevice* dst) noexcept;

  // Copies one or more tensors, optionally using a stream for async copies.
  static OrtStatus* ORT_API_CALL CopyTensorsImpl(OrtDataTransferImpl* this_ptr,
                                                  const OrtValue** src_tensors,
                                                  OrtValue** dst_tensors,
                                                  OrtSyncStream** streams,
                                                  size_t num_tensors) noexcept;

  // Called by ORT when it is done with this object.
  // We use a factory-owned singleton, so this is a no-op.
  static void ORT_API_CALL ReleaseImpl(OrtDataTransferImpl* /*this_ptr*/) noexcept {}

 private:
  const OrtMemoryDevice* device_mem_;  // our EP's device descriptor
};
