// neuron_ep_factory.h
// NeuronEpFactory: process-wide factory, creates per-session NeuronEp objects.
// One instance exists for the lifetime of the ORT environment after registration.

#pragma once

#include <mutex>
#include <memory>
#include <string>

#include "neuron_ep_utils.h"
#include "neuron_ep_allocator.h"
#include "neuron_ep_data_transfer.h"

// Forward declarations
class NeuronEp;

// ---------------------------------------------------------------------------
// NeuronEpFactory
// Inherits OrtEpFactory (C vtable struct) and ApiPtrs (three API pointers).
// ---------------------------------------------------------------------------
class NeuronEpFactory : public OrtEpFactory, public ApiPtrs {
 public:
  // canonical EP name; independent of the name used to register the library
  static constexpr const char* kEpName = "NeuroPilotExecutionProvider";

  // ort_api_version is the negotiated ORT API version (see CreateEpFactories);
  // it may be lower than the ORT_API_VERSION this library was compiled against.
  NeuronEpFactory(ApiPtrs apis, const OrtLogger& default_logger, uint32_t ort_api_version);
  ~NeuronEpFactory();

  // Accessors used by NeuronEp / stream support.
  NeuronDataTransfer* GetDataTransfer() const { return data_transfer_.get(); }

  const OrtLogger& default_logger_;

 private:
  // ---- OrtEpFactory vtable implementations (all static) -------------------
  static const char* ORT_API_CALL GetNameImpl(const OrtEpFactory* f) noexcept;
  static const char* ORT_API_CALL GetVendorImpl(const OrtEpFactory* f) noexcept;
  static uint32_t    ORT_API_CALL GetVendorIdImpl(const OrtEpFactory* f) noexcept;
  static const char* ORT_API_CALL GetVersionImpl(const OrtEpFactory* f) noexcept;

  static OrtStatus* ORT_API_CALL GetSupportedDevicesImpl(OrtEpFactory* f,
                                                          const OrtHardwareDevice* const* devices,
                                                          size_t num_devices,
                                                          OrtEpDevice** ep_devices,
                                                          size_t max_ep_devices,
                                                          size_t* num_ep_devices) noexcept;

  static OrtStatus* ORT_API_CALL CreateEpImpl(OrtEpFactory* f,
                                               const OrtHardwareDevice* const* devices,
                                               const OrtKeyValuePairs* const* ep_metadata,
                                               size_t num_devices,
                                               const OrtSessionOptions* session_options,
                                               const OrtLogger* logger,
                                               OrtEp** ep) noexcept;

  static void ORT_API_CALL ReleaseEpImpl(OrtEpFactory* f, OrtEp* ep) noexcept;

  static OrtStatus* ORT_API_CALL CreateAllocatorImpl(OrtEpFactory* f,
                                                      const OrtMemoryInfo* memory_info,
                                                      const OrtKeyValuePairs* options,
                                                      OrtAllocator** allocator) noexcept;

  static void ORT_API_CALL ReleaseAllocatorImpl(OrtEpFactory* f,
                                                 OrtAllocator* allocator) noexcept;

  static OrtStatus* ORT_API_CALL CreateDataTransferImpl(OrtEpFactory* f,
                                                         OrtDataTransferImpl** dt) noexcept;

  static bool       ORT_API_CALL IsStreamAwareImpl(const OrtEpFactory* f) noexcept;

  static OrtStatus* ORT_API_CALL CreateSyncStreamForDeviceImpl(OrtEpFactory* f,
                                                                const OrtMemoryDevice* device,
                                                                const OrtKeyValuePairs* options,
                                                                OrtSyncStreamImpl** stream) noexcept;

  static OrtStatus* ORT_API_CALL GetHardwareDeviceIncompatibilityDetailsImpl(
      OrtEpFactory* f,
      const OrtHardwareDevice* hw,
      OrtDeviceEpIncompatibilityDetails* details) noexcept;

  // ---- Factory state -------------------------------------------------------
  const std::string ep_name_{kEpName};

  // MediaTek Neuron EP identity.
  // Vendor ID 0x0E8D is MediaTek's PCI vendor ID.
  const std::string vendor_{"MediaTek"};
  const uint32_t    vendor_id_{0x0E8D};
  // Release version of this EP (NOT the ORT version). Defined by CMake from
  // project(ORTNeuronEp VERSION ...)
  const std::string ep_version_{ORT_NEURON_EP_VERSION};

  // CPU memory info: Neuron EP uses CPU-accessible memory.
  // The NPU acceleration is internal to the Neuron SDK.
  Ort::MemoryInfo default_memory_info_;   // default device memory
  Ort::MemoryInfo readonly_memory_info_;  // for initializers / weights

  // Shared allocator (reference-counted across sessions).
  std::unique_ptr<CustomAllocator> shared_allocator_;
  uint32_t   num_allocator_users_{0};
  std::mutex mutex_;

  // Shared data-transfer object.
  std::unique_ptr<NeuronDataTransfer> data_transfer_;

  // Provider-level options forwarded to each NeuronEp instance.
  // Owned by this factory.
  OrtKeyValuePairs* provider_options_{nullptr};
};
