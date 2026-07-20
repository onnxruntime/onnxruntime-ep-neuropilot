/*
 * Copyright (C) 2024 MediaTek Inc.
 * (Plugin EP refactoring: removed ORT internal headers, replaced InlinedVector
 * with std::vector, replaced onnxruntime::Status with NeuronStatus.)
 */
#pragma once

#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include "../neuron_types_internal.h"
#include "neuron_types.h"

namespace onnxruntime {
namespace neuron {

// Shape: vector of uint32_t dimensions.
// (Previously InlinedVector<uint32_t> from core/common/basic_types.h.)
using Shape = std::vector<uint32_t>;

namespace wrapper {

enum class Type {
  FLOAT32 = NEURON_FLOAT32,
  INT32 = NEURON_INT32,
  UINT32 = NEURON_UINT32,
  TENSOR_FLOAT32 = NEURON_TENSOR_FLOAT32,
  TENSOR_INT32 = NEURON_TENSOR_INT32,
  TENSOR_QUANT8_ASYMM = NEURON_TENSOR_QUANT8_ASYMM,
  TENSOR_QUANT8_SYMM = NEURON_TENSOR_QUANT8_SYMM,
  BOOL = NEURON_BOOL,
  FLOAT16 = NEURON_FLOAT16,
  TENSOR_QUANT16_SYMM = NEURON_TENSOR_QUANT16_SYMM,
  TENSOR_QUANT16_ASYMM = NEURON_TENSOR_QUANT16_ASYMM,
  TENSOR_FLOAT16 = NEURON_TENSOR_FLOAT16,
  TENSOR_BOOL8 = NEURON_TENSOR_BOOL8,
  TENSOR_QUANT8_SYMM_PER_CHANNEL = NEURON_TENSOR_QUANT8_SYMM_PER_CHANNEL,
  TENSOR_UINT32 = NEURON_EXT_TENSOR_UINT32,
  TENSOR_QUANT4_SYMM = NEURON_EXT_TENSOR_QUANT4_SYMM,
  TENSOR_QUANT4_ASYMM = NEURON_EXT_TENSOR_QUANT4_ASYMM,
  TENSOR_QUANT4_ASYMM_SIGNED = NEURON_EXT_TENSOR_QUANT4_ASYMM_SIGNED,
  TENSOR_QUANT8_ASYMM_PER_CHANNEL = NEURON_EXT_TENSOR_QUANT8_ASYMM_PER_CHANNEL,
  TENSOR_QUANT16_ASYMM_SIGNED = NEURON_EXT_TENSOR_QUANT16_ASYMM_SIGNED,
  TENSOR_QUANT8_ASYMM_SIGNED_PER_CHANNEL = NEURON_EXT_TENSOR_QUANT8_ASYMM_SIGNED_PER_CHANNEL,
  TENSOR_QUANT8_ASYMM_SIGNED = NEURON_TENSOR_QUANT8_ASYMM_SIGNED,
  TENSOR_INT64 = NEURON_EXT_TENSOR_INT64,
  TENSOR_UINT64 = NEURON_EXT_TENSOR_UINT64,
  EXTENSION = 999999,
};

enum class ExecutePreference {
  PREFER_LOW_POWER = NEURON_PREFER_LOW_POWER,
  PREFER_FAST_SINGLE_ANSWER = NEURON_PREFER_FAST_SINGLE_ANSWER,
  PREFER_SUSTAINED_SPEED = NEURON_PREFER_SUSTAINED_SPEED,
  PREFER_TURBO_BOOST = NEURON_PREFER_TURBO_BOOST
};

#ifdef _MSC_VER
#undef NO_ERROR
#endif

enum class Result {
  NO_ERROR = NEURON_NO_ERROR,
  OUT_OF_MEMORY = NEURON_OUT_OF_MEMORY,
  INCOMPLETE = NEURON_INCOMPLETE,
  UNEXPECTED_NULL = NEURON_UNEXPECTED_NULL,
  BAD_DATA = NEURON_BAD_DATA,
  OP_FAILED = NEURON_OP_FAILED,
  UNMAPPABLE = NEURON_UNMAPPABLE,
  BAD_STATE = NEURON_BAD_STATE,
  OUTPUT_INSUFFICIENT_SIZE = NEURON_OUTPUT_INSUFFICIENT_SIZE,
  UNAVAILABLE_DEVICE = NEURON_UNAVAILABLE_DEVICE,
};

std::string TypeToStr(const Type& type);

struct SymmPerChannelQuantParams {
  NeuronSymmPerChannelQuantParams params;
  std::vector<float> scales;
  SymmPerChannelQuantParams(std::vector<float> scalesVec, uint32_t channelDim)
      : scales(std::move(scalesVec)) {
    params = {
        /*.channelDim = */ channelDim,
        /*.scaleCount =*/static_cast<uint32_t>(scales.size()),
        /*.scales = */ scales.size() > 0 ? scales.data() : nullptr,
    };
  }
  SymmPerChannelQuantParams(const SymmPerChannelQuantParams& other)
      : params(other.params), scales(other.scales) {
    params.scales = scales.size() > 0 ? scales.data() : nullptr;
  }
  SymmPerChannelQuantParams& operator=(const SymmPerChannelQuantParams& other) {
    if (this != &other) {
      params = other.params;
      scales = other.scales;
      params.scales = scales.size() > 0 ? scales.data() : nullptr;
    }
    return *this;
  }
};

struct PerChannelQuantParams {
  NeuronPerChannelQuantParams params;
  std::vector<float> scales;
  std::vector<int32_t> zero_points;
  PerChannelQuantParams(const std::vector<float>& scales_vec,
                        const std::vector<int32_t>& zp, uint32_t channelDim)
      : scales(scales_vec), zero_points(zp) {
    params = {
        channelDim,
        static_cast<uint32_t>(scales.size()),
        scales.size() > 0 ? scales.data() : nullptr,
        static_cast<uint32_t>(zero_points.size()),
        zero_points.size() > 0 ? zero_points.data() : nullptr,
    };
  }
  PerChannelQuantParams(const PerChannelQuantParams& other)
      : params(other.params), scales(other.scales), zero_points(other.zero_points) {
    params.scales = scales.size() > 0 ? scales.data() : nullptr;
    params.zeroPoints = zero_points.size() > 0 ? zero_points.data() : nullptr;
  }
  PerChannelQuantParams& operator=(const PerChannelQuantParams& other) {
    if (this != &other) {
      params = other.params;
      scales = other.scales;
      zero_points = other.zero_points;
      params.scales = scales.size() > 0 ? scales.data() : nullptr;
      params.zeroPoints = zero_points.size() > 0 ? zero_points.data() : nullptr;
    }
    return *this;
  }
};

// Convenience wrapper for ONNX type info used to construct an OperandType.
struct OnnxDataTypeInfo {
  int32_t onnx_data_type = 0;
  std::vector<float> scales{};
  std::vector<int32_t> zero_points{};
  std::optional<int64_t> channel_dim{};
};

struct OperandType {
  NeuronOperandType operandType;
  Type type;
  Shape dimensions;
  std::optional<SymmPerChannelQuantParams> channelQuant;
  std::optional<PerChannelQuantParams> asymmChannelQuant;

  explicit OperandType(const NeuronOperandType& operandType);
  explicit OperandType(Type type, const Shape& d, float scale = 0.0f, int32_t zeroPoint = 0);
  explicit OperandType(Type type, const Shape& d, SymmPerChannelQuantParams&& channelQuant);
  explicit OperandType(Type type, const Shape& d, PerChannelQuantParams&& asymmChannelQuant);
  explicit OperandType(const OnnxDataTypeInfo& onnx_io_info, const Shape& d);

  OperandType(const OperandType& other);
  OperandType& operator=(const OperandType& other);

  size_t GetElementByteSize() const;
  size_t GetOperandBlobByteSize() const;

  void SetDimensions(const Shape& d);
  void SetNeuronOperandType(const NeuronOperandType& operandType,
                            const std::optional<SymmPerChannelQuantParams>& symm_param = {},
                            const std::optional<PerChannelQuantParams>& asymm_param = {});

  operator NeuronOperandType() const { return operandType; }
};

}  // namespace wrapper
}  // namespace neuron
}  // namespace onnxruntime
