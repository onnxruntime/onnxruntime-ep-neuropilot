/*
 * Copyright (C) 2024 MediaTek Inc., this file is modified on 06/21/2024
 * by MediaTek Inc. based on MIT License .
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the ""Software""), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED ""AS IS"", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "neuron_wrapper_operand.h"

#include "../neuron_types_internal.h"
#include "../builders/helper.h"
#include "neuron_types.h"
#include <onnx/onnx_pb.h>
#include <numeric>

namespace onnxruntime {
namespace neuron {
namespace wrapper {
std::string TypeToStr(const Type& type) {
  if (type == Type::FLOAT32) {
    return "FLOAT32";
  } else if (type == Type::INT32) {
    return "INT32";
  } else if (type == Type::UINT32) {
    return "UINT32";
  } else if (type == Type::TENSOR_FLOAT32) {
    return "TENSOR_FLOAT32";
  } else if (type == Type::TENSOR_INT32) {
    return "TENSOR_INT32";
  } else if (type == Type::TENSOR_UINT32) {
    return "TENSOR_UINT32";
  } else if (type == Type::TENSOR_QUANT4_ASYMM_SIGNED) {
    return "TENSOR_QUANT4_ASYMM_SIGNED";
  } else if (type == Type::TENSOR_QUANT4_ASYMM) {
    return "TENSOR_QUANT4_ASYMM";
  } else if (type == Type::TENSOR_QUANT4_SYMM) {
    return "TENSOR_QUANT4_SYMM";
  } else if (type == Type::TENSOR_QUANT8_ASYMM) {
    return "TENSOR_QUANT8_ASYMM";
  } else if (type == Type::TENSOR_QUANT8_SYMM) {
    return "TENSOR_QUANT8_SYMM";
  } else if (type == Type::TENSOR_QUANT16_SYMM) {
    return "TENSOR_QUANT16_SYMM";
  } else if (type == Type::TENSOR_QUANT16_ASYMM) {
    return "TENSOR_QUANT16_ASYMM";
  } else if (type == Type::TENSOR_QUANT16_ASYMM_SIGNED) {
    return "TENSOR_QUANT16_ASYMM_SIGNED";
  } else if (type == Type::BOOL) {
    return "BOOL";
  } else if (type == Type::TENSOR_FLOAT16) {
    return "TENSOR_FLOAT16";
  } else if (type == Type::FLOAT16) {
    return "FLOAT16";
  } else if (type == Type::TENSOR_BOOL8) {
    return "TENSOR_BOOL8";
  } else if (type == Type::TENSOR_QUANT8_SYMM_PER_CHANNEL) {
    return "TENSOR_QUANT8_SYMM_PER_CHANNEL";
  } else if (type == Type::TENSOR_QUANT8_ASYMM_PER_CHANNEL) {
    return "TENSOR_QUANT8_ASYMM_PER_CHANNEL";
  } else if (type == Type::TENSOR_QUANT8_ASYMM_SIGNED_PER_CHANNEL) {
    return "TENSOR_QUANT8_ASYMM_SIGNED_PER_CHANNEL";
  } else if (type == Type::TENSOR_QUANT8_ASYMM_SIGNED) {
    return "TENSOR_QUANT8_ASYMM_SIGNED";
  } else if (type == Type::TENSOR_INT64) {
    return "TENSOR_INT64";
  } else if (type == Type::TENSOR_UINT64) {
    return "TENSOR_UINT64";
  } else if (type == Type::EXTENSION) {
    return "EXTENSION";
  }
  return "Unknown type(" + std::to_string(static_cast<int>(type)) + ")";
}

Type NeuronTypeToType(int32_t type) {
  switch (type) {
    case NEURON_FLOAT32:
      return Type::FLOAT32;
    case NEURON_INT32:
      return Type::INT32;
    case NEURON_UINT32:
      return Type::UINT32;
    case NEURON_TENSOR_FLOAT32:
      return Type::TENSOR_FLOAT32;
    case NEURON_TENSOR_INT32:
      return Type::TENSOR_INT32;
    case NEURON_TENSOR_QUANT8_ASYMM:
      return Type::TENSOR_QUANT8_ASYMM;
    case NEURON_TENSOR_QUANT8_SYMM:
      return Type::TENSOR_QUANT8_SYMM;
    case NEURON_BOOL:
      return Type::BOOL;
    case NEURON_FLOAT16:
      return Type::FLOAT16;
    case NEURON_TENSOR_QUANT16_SYMM:
      return Type::TENSOR_QUANT16_SYMM;
    case NEURON_TENSOR_QUANT16_ASYMM:
      return Type::TENSOR_QUANT16_ASYMM;
    case NEURON_TENSOR_FLOAT16:
      return Type::TENSOR_FLOAT16;
    case NEURON_TENSOR_BOOL8:
      return Type::TENSOR_BOOL8;
    case NEURON_TENSOR_QUANT8_SYMM_PER_CHANNEL:
      return Type::TENSOR_QUANT8_SYMM_PER_CHANNEL;
    case NEURON_EXT_TENSOR_UINT32:
      return Type::TENSOR_UINT32;
    case NEURON_EXT_TENSOR_QUANT4_SYMM:
      return Type::TENSOR_QUANT4_SYMM;
    case NEURON_EXT_TENSOR_QUANT4_ASYMM:
      return Type::TENSOR_QUANT4_ASYMM;
    case NEURON_EXT_TENSOR_QUANT4_ASYMM_SIGNED:
      return Type::TENSOR_QUANT4_ASYMM_SIGNED;
    case NEURON_EXT_TENSOR_QUANT8_ASYMM_PER_CHANNEL:
      return Type::TENSOR_QUANT8_ASYMM_PER_CHANNEL;
    case NEURON_EXT_TENSOR_QUANT16_ASYMM_SIGNED:
      return Type::TENSOR_QUANT16_ASYMM_SIGNED;
    case NEURON_EXT_TENSOR_QUANT8_ASYMM_SIGNED_PER_CHANNEL:
      return Type::TENSOR_QUANT8_ASYMM_SIGNED_PER_CHANNEL;
    case NEURON_TENSOR_QUANT8_ASYMM_SIGNED:
      return Type::TENSOR_QUANT8_ASYMM_SIGNED;
    case NEURON_EXT_TENSOR_INT64:
      return Type::TENSOR_INT64;
    case NEURON_EXT_TENSOR_UINT64:
      return Type::TENSOR_UINT64;
    default:
      return Type::EXTENSION;
  }
}

Type OnnxDataTypeToType(const OnnxDataTypeInfo& onnx_type_info,
                        const Shape& shape) {
  int32_t type = onnx_type_info.onnx_data_type;
  const std::vector<float>& scales = onnx_type_info.scales;
  const std::vector<int32_t>& zero_points = onnx_type_info.zero_points;
  switch (type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return shape.size() > 0 ? Type::TENSOR_FLOAT32 : Type::FLOAT32;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      return scales.size() > 1 ? Type::TENSOR_QUANT8_ASYMM_PER_CHANNEL : Type::TENSOR_QUANT8_ASYMM;
    case ONNX_NAMESPACE::TensorProto_DataType_INT8: {
      if (scales.size() == 0) {
        return Type::TENSOR_QUANT8_SYMM;
      } else if (scales.size() == 1) {
        // TENSOR_QUANT8_SYMM is only valid for weight tensors in old-style symmetric
        // per-tensor quantization.  Activations in QDQ models must always use
        // TENSOR_QUANT8_ASYMM_SIGNED — even when zero_point happens to be 0 —
        // because the Neuron SDK rejects SYMM when used as a Conv/binary-op input.
        // Using ASYMM_SIGNED with zero_point=0 is equally correct for symmetric
        // per-tensor weights and is compatible with ASYMM_SIGNED activations.
        return Type::TENSOR_QUANT8_ASYMM_SIGNED;
      } else {
        const bool no_zero_point = std::all_of(zero_points.begin(), zero_points.end(), [](auto v) { return v == 0; });
        return no_zero_point ? Type::TENSOR_QUANT8_SYMM_PER_CHANNEL : Type::TENSOR_QUANT8_ASYMM_SIGNED_PER_CHANNEL;
      }
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16: {
      return Type::TENSOR_QUANT16_ASYMM;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT16:
      return Type::TENSOR_QUANT16_SYMM;
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      return shape.size() > 0 ? Type::TENSOR_INT32 : Type::INT32;
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
      return shape.size() > 0 ? Type::TENSOR_BOOL8 : Type::BOOL;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      return shape.size() > 0 ? Type::TENSOR_FLOAT16 : Type::FLOAT16;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
      return shape.size() > 0 ? Type::TENSOR_UINT32 : Type::UINT32;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
      if (shape.size() > 0) {
        return Type::TENSOR_UINT64;
      }
      throw std::runtime_error(MakeNeuronString("Can't convert onnx type: ", GetTensorProtoDataTypeStr(type), " (scalar) to Neuron EP type"));
    case ONNX_NAMESPACE::TensorProto_DataType_UINT4:
      if (scales.size() <= 1) {
        return Type::TENSOR_QUANT4_ASYMM;
      }
      throw std::runtime_error(MakeNeuronString("Can't convert onnx type: ", GetTensorProtoDataTypeStr(type), " (per-channel) to Neuron EP type"));
    case ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED:
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2FNUZ:  // not supported yet
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2:      // not supported yet
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FNUZ:  // not supported yet
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN:    // not supported yet
    case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:        // not supported yet
    case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128:      // not supported yet
    case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64:       // not supported yet
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:          // not supported yet
    case ONNX_NAMESPACE::TensorProto_DataType_STRING:          // not supported yet
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:           // not supported — nodes with INT64 primary inputs are rejected at GetCapability
      throw std::runtime_error(MakeNeuronString("Can't convert onnx type: ", GetTensorProtoDataTypeStr(type), " to Neuron EP type"));
    default:
      return Type::EXTENSION;
  }
}

OperandType::OperandType(const NeuronOperandType& operandType) {
  SetNeuronOperandType(operandType);
}

OperandType::OperandType(Type type, const Shape& d, float scale, int32_t zeroPoint)
    : type(type), dimensions(d) {
  operandType = {
      /*.type = */ static_cast<int32_t>(type),
      /*.dimensionCount = */ static_cast<uint32_t>(dimensions.size()),
      /*.dimensions = */ dimensions.size() > 0 ? dimensions.data() : nullptr,
      /*.scale = */ scale,
      /*.zeroPoint = */ zeroPoint,
  };
}

OperandType::OperandType(Type type, const Shape& d, SymmPerChannelQuantParams&& channelQuant)
    : type(type), dimensions(d), channelQuant(std::move(channelQuant)) {
  operandType = {
      /*.type = */ static_cast<int32_t>(type),
      /*.dimensionCount = */ static_cast<uint32_t>(dimensions.size()),
      /*.dimensions = */ dimensions.size() > 0 ? dimensions.data() : nullptr,
      /*.scale = */ 0.0f,
      /*.zeroPoint = */ 0,
  };
  if (channelQuant.params.channelDim >= dimensions.size()) {
    throw std::runtime_error(MakeNeuronString("Channel quant channelDim is greater then dimensions size"));
  }
}

OperandType::OperandType(Type type, const Shape& d, PerChannelQuantParams&& channelQuant)
    : type(type), dimensions(d), asymmChannelQuant(std::move(channelQuant)) {
  operandType = {
      /*.type = */ static_cast<int32_t>(type),
      /*.dimensionCount = */ static_cast<uint32_t>(dimensions.size()),
      /*.dimensions = */ dimensions.size() > 0 ? dimensions.data() : nullptr,
      /*.scale = */ 0.0f,
      /*.zeroPoint = */ 0,
  };
  if (asymmChannelQuant && asymmChannelQuant->params.channelDim >= dimensions.size()) {
    throw std::runtime_error(MakeNeuronString("Asyym per-channel quant channelDim is greater then dimensions size"));
  }
}

OperandType::OperandType(const OnnxDataTypeInfo& onnx_type_info, const Shape& d)
    : type(OnnxDataTypeToType(onnx_type_info, d)), dimensions(d) {
  const auto& scales = onnx_type_info.scales;
  const auto& zero_points = onnx_type_info.zero_points;
  operandType = {
      /*.type = */ static_cast<int32_t>(type),
      /*.dimensionCount = */ static_cast<uint32_t>(dimensions.size()),
      /*.dimensions = */ dimensions.size() > 0 ? dimensions.data() : nullptr,
      /*.scale = */ scales.size() == 1 ? scales[0] : 0.0f,
      /*.zeroPoint = */ zero_points.size() == 1 ? zero_points[0] : 0,
  };
  if (scales.size() > 1) {
    auto axis = onnx_type_info.channel_dim.value_or(1);
    auto rank = d.size();
    if (rank == 1) {
      axis = 0;
    }
    const bool no_zero_point = std::all_of(zero_points.begin(), zero_points.end(), [](auto v) { return v == 0; });
    if (no_zero_point) {
      channelQuant = SymmPerChannelQuantParams(scales,
                                               static_cast<uint32_t>(axis));
    } else {
      asymmChannelQuant = PerChannelQuantParams(scales,
                                                zero_points,
                                                static_cast<uint32_t>(axis));
    }
  }
}

// OperandType::OperandType(const GraphViewer& graph_viewer, const NodeUnit& node_unit, ArgType arg_type, size_t index){

// }

OperandType::OperandType(const OperandType& other)
    : operandType(other.operandType),
      type(other.type),
      dimensions(other.dimensions),
      channelQuant(other.channelQuant),
      asymmChannelQuant(other.asymmChannelQuant) {
  operandType.dimensions = dimensions.size() > 0 ? dimensions.data() : nullptr;
}

OperandType& OperandType::operator=(const OperandType& other) {
  if (this != &other) {
    type = other.type;
    dimensions = other.dimensions;
    operandType = other.operandType;
    channelQuant = other.channelQuant;
    asymmChannelQuant = other.asymmChannelQuant;
    operandType.dimensions = dimensions.size() > 0 ? dimensions.data() : nullptr;
  }

  return *this;
}

size_t OperandType::GetElementByteSize() const {
  switch (type) {
    case Type::TENSOR_QUANT8_ASYMM:
    case Type::TENSOR_QUANT8_SYMM:
    case Type::TENSOR_QUANT8_ASYMM_SIGNED:
    case Type::TENSOR_QUANT8_SYMM_PER_CHANNEL:
    case Type::TENSOR_QUANT8_ASYMM_PER_CHANNEL:
    case Type::TENSOR_QUANT8_ASYMM_SIGNED_PER_CHANNEL:
    case Type::BOOL:
    case Type::TENSOR_BOOL8:
      return 1;
    case Type::FLOAT16:
    case Type::TENSOR_FLOAT16:
    case Type::TENSOR_QUANT16_SYMM:
    case Type::TENSOR_QUANT16_ASYMM:
      return 2;
    case Type::TENSOR_FLOAT32:
    case Type::FLOAT32:
    case Type::UINT32:
    case Type::TENSOR_INT32:
      return 4;
    case Type::TENSOR_INT64:
    case Type::TENSOR_UINT64:
      return 8;
    default:
      throw std::runtime_error(MakeNeuronString("Wrong type: " + TypeToStr(type)));
  }
}

size_t OperandType::GetOperandBlobByteSize() const {
  size_t result = GetElementByteSize();
  for (auto d : dimensions) result *= d;
  return result;
}

void OperandType::SetDimensions(const Shape& d) {
  dimensions = d;
  operandType.dimensionCount = static_cast<uint32_t>(dimensions.size());
  operandType.dimensions = dimensions.size() > 0 ? dimensions.data() : nullptr;
  if (channelQuant.has_value() && channelQuant.value().params.channelDim > dimensions.size()) {
    throw std::runtime_error(MakeNeuronString("Channel quant channelDim is greater then dimensions size"));
  }
}

void OperandType::SetNeuronOperandType(const NeuronOperandType& neuronOperandType,
                                       const std::optional<SymmPerChannelQuantParams>& symm_param,
                                       const std::optional<PerChannelQuantParams>& asymm_param) {
  // operandType
  type = NeuronTypeToType(neuronOperandType.type);
  dimensions.clear();
  for (uint32_t i = 0; i < neuronOperandType.dimensionCount; i++) {
    dimensions.push_back(neuronOperandType.dimensions[i]);
  }

  operandType = {
      /*.type = */ neuronOperandType.type,
      /*.dimensionCount = */ static_cast<uint32_t>(dimensions.size()),
      /*.dimensions = */ dimensions.size() > 0 ? dimensions.data() : nullptr,
      /*.scale = */ neuronOperandType.scale,
      /*.zeroPoint = */ neuronOperandType.zeroPoint,
  };

  if (symm_param) {
    channelQuant = symm_param;
  } else if (asymm_param) {
    asymmChannelQuant = asymm_param;
  }
}

}  // namespace wrapper
}  // namespace neuron
}  // namespace onnxruntime
