// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <gsl/span>

#include "../neuron_types_internal.h"
#include "neuron_wrapper_operand.h"

#ifdef __ANDROID__
#include <android/api-level.h>
#endif

struct NeuronApi;
namespace onnxruntime {
namespace neuron {

class ModelBuilder;

namespace wrapper {

struct Device {
  NeuronDevice* device;
  std::string name;
  std::string GetDescription() const;
};

enum class TargetDeviceOption : int8_t {
  ALL_DEVICES,  // use all available target devices

  /* TODO support these options
  PREFERRED_DEVICES,  // Use one or more preferred devices (must be given)
  EXCLUDED_DEVICES,   // Exclude one or more devices (must be given)
   */

  CPU_DISABLED,  // use all available target devices except CPU
  CPU_ONLY,      // use CPU only
};

constexpr const char* const kNeuronCpuDeviceName = "neuron-reference";

using DeviceVector = std::vector<Device>;

/**
 * Get all target devices specified by target_device_option.
 */
NeuronStatus GetTargetDevices(const NeuronApi& neuron_handle, TargetDeviceOption target_device_option,
                        DeviceVector& neuron_target_devices);

std::string GetDevicesDescription(gsl::span<const wrapper::Device> devices);

/**
 * Check if a given device name is contained in a list of devices.
 */
inline bool ContainsDevice(gsl::span<const Device> devices, const std::string& target_name) {
  for (const auto& device : devices) {
    if (device.name == target_name) return true;
  }
  return false;
}
}  // namespace wrapper
}  // namespace neuron
}  // namespace onnxruntime
