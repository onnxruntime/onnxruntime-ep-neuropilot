#include <gsl/span>

#include "neuron_wrapper_device.h"

#include "../builders/helper.h"
#include "neuron_implementation.h"

namespace onnxruntime {
namespace neuron {
namespace wrapper {

/**
 * Get all target devices specified by target_device_option
 * CPU device Will be put last
 *
 * @param neuron_handle nnapi-lib handle.
 * @param target_device_option target devices users want to use.
 *
 * @return NeuronStatus of the operation.
 *
 */
NeuronStatus GetTargetDevices(const NeuronApi& neuron_handle, TargetDeviceOption target_device_option,
                        DeviceVector& devices) {
  uint32_t num_devices = 0;
  NEURON_RETURN_STATUS_ON_ERROR_WITH_NOTE(
      neuron_handle.Neuron_getDeviceCount(&num_devices), "Getting count of available devices");

  for (uint32_t i = 0; i < num_devices; i++) {
    NeuronDevice* device = nullptr;
    const char* device_name = nullptr;
    NEURON_RETURN_STATUS_ON_ERROR_WITH_NOTE(
        neuron_handle.Neuron_getDevice(i, &device), "Getting " + std::to_string(i) + "th device");

    NEURON_RETURN_STATUS_ON_ERROR_WITH_NOTE(neuron_handle.NeuronDevice_getName(device, &device_name),
                                            "Getting " + std::to_string(i) + "th device's name");
    devices.push_back({device, std::string(device_name)});
  }
  return NeuronStatus::OK();
}

std::string GetDevicesDescription(gsl::span<const Device> devices) {
  std::string neuron_target_devices_detail;
  for (const auto& device : devices) {
    if (!neuron_target_devices_detail.empty()) {
      neuron_target_devices_detail += ", ";
    }
    neuron_target_devices_detail += device.GetDescription();
  }
  return neuron_target_devices_detail;
}

std::string Device::GetDescription() const {
  return MakeNeuronString("[Name: [", name, "]]");
}

}  // namespace wrapper
}  // namespace neuron
}  // namespace onnxruntime
