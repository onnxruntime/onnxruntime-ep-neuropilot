// neuron_ep_data_transfer.cc

#include "neuron_ep_data_transfer.h"
#include <cstring>

/*static*/
bool ORT_API_CALL NeuronDataTransfer::CanCopyImpl(const OrtDataTransferImpl* this_ptr,
                                                   const OrtMemoryDevice* src,
                                                   const OrtMemoryDevice* dst) noexcept {
  const auto& impl = *static_cast<const NeuronDataTransfer*>(this_ptr);

  bool src_ours = impl.ep_api.MemoryDevice_AreEqual(src, impl.device_mem_);
  bool dst_ours = impl.ep_api.MemoryDevice_AreEqual(dst, impl.device_mem_);

  // Our device ↔ our device: always ok.
  if (src_ours && dst_ours) return true;

  auto src_type = impl.ep_api.MemoryDevice_GetDeviceType(src);
  auto dst_type = impl.ep_api.MemoryDevice_GetDeviceType(dst);
  auto src_mem  = impl.ep_api.MemoryDevice_GetMemoryType(src);
  auto dst_mem  = impl.ep_api.MemoryDevice_GetMemoryType(dst);

  // Our device → CPU (or host-accessible memory).
  if (src_ours)
    return (dst_type == OrtMemoryInfoDeviceType_CPU ||
            dst_mem  == OrtDeviceMemoryType_HOST_ACCESSIBLE);

  // CPU (or host-accessible) → our device.
  if (dst_ours)
    return (src_type == OrtMemoryInfoDeviceType_CPU ||
            src_mem  == OrtDeviceMemoryType_HOST_ACCESSIBLE);

  return false;
}

/*static*/
OrtStatus* ORT_API_CALL NeuronDataTransfer::CopyTensorsImpl(OrtDataTransferImpl* this_ptr,
                                                              const OrtValue** src_tensors,
                                                              OrtValue** dst_tensors,
                                                              OrtSyncStream** streams,
                                                              size_t num_tensors) noexcept {
  auto& impl = *static_cast<NeuronDataTransfer*>(this_ptr);

  for (size_t i = 0; i < num_tensors; ++i) {
    const void* src_data = nullptr;
    void*       dst_data = nullptr;
    size_t      bytes    = 0;

    RETURN_IF_ERROR(impl.ort_api.GetTensorData(src_tensors[i], &src_data));
    RETURN_IF_ERROR(impl.ort_api.GetTensorMutableData(dst_tensors[i], &dst_data));
    RETURN_IF_ERROR(impl.ort_api.GetTensorSizeInBytes(src_tensors[i], &bytes));

    // In a real (GPU) EP you would dispatch to cudaMemcpyAsync here.
    // Since this is a CPU-based dummy EP we just memcpy.
    (void)streams;  // unused: synchronous copy
    std::memcpy(dst_data, src_data, bytes);
  }
  return nullptr;
}
