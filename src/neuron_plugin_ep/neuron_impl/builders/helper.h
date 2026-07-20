// helper.h
// Plugin EP version – all ORT internal headers removed.
// Uses NeuronGraphContext instead of GraphViewer/NodeUnit.

#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include <onnx/onnx_pb.h>
#include <gsl/span>

#include "../neuron_graph_context.h"
#include "../neuron_lib/neuron_wrapper_device.h"
#include "../neuron_lib/neuron_wrapper_operand.h"

#ifndef ORT_NEURON_MAX_SUPPORTED_API_LEVEL
#define ORT_NEURON_MAX_SUPPORTED_API_LEVEL 31
#endif

// ---------------------------------------------------------------------------
// Neuron error-check macros (use NeuronStatus; replaces ORT_RETURN_IF_NOT)
// ---------------------------------------------------------------------------
#define NEURON_THROW_ON_ERROR(val)           \
  {                                          \
    const auto ret = (val);                  \
    NEURON_ENFORCE(                          \
        ret == NEURON_NO_ERROR,              \
        "ResultCode: ", GetErrorCause(ret)); \
  }

#define NEURON_THROW_ON_ERROR_WITH_NOTE(val, note)           \
  {                                                          \
    const auto ret = (val);                                  \
    NEURON_ENFORCE(                                          \
        ret == NEURON_NO_ERROR,                              \
        "ResultCode: ", GetErrorCause(ret), ", ", (note));   \
  }

#define NEURON_RETURN_STATUS_ON_ERROR(val)   \
  {                                          \
    const auto ret = (val);                  \
    NEURON_RETURN_IF_NOT(                    \
        ret == NEURON_NO_ERROR,              \
        "ResultCode: ", GetErrorCause(ret)); \
  }

#define NEURON_RETURN_STATUS_ON_ERROR_WITH_NOTE(val, note) \
  {                                                        \
    const auto ret = (val);                                \
    NEURON_RETURN_IF_NOT(                                  \
        ret == NEURON_NO_ERROR,                            \
        "ResultCode: ", GetErrorCause(ret), ", ", (note)); \
  }

// ---- Utility functions (header-only) ----------------------------------------

// Normalize a negative axis index: axis in [-rank, rank).
inline int64_t HandleNegativeAxis(int64_t axis, int64_t rank) {
  return axis < 0 ? axis + rank : axis;
}

// Create a gsl::span from an initializer-list (temporary storage, valid for
// the duration of the enclosing full-expression).
template <typename T>
inline gsl::span<const T> AsSpan(std::initializer_list<T> il) {
  return gsl::span<const T>(il.begin(), static_cast<typename gsl::span<const T>::size_type>(il.size()));
}

namespace onnxruntime {
namespace neuron {

// Forward declarations
class ModelBuilder;
struct OpSupportCheckParams;

// ---- String helpers --------------------------------------------------------
std::string NeuronOperandTypeToStr(int type);
std::string NeuronOperationTypeToStr(int type);
std::string GetErrorCause(int error_code);
std::string GetTensorProtoDataTypeStr(int tensor_element_type);

// ---- Quantized op type classification --------------------------------------
// In the plugin EP, QDQ nodes arrive as individual QuantizeLinear /
// DequantizeLinear ONNX nodes plus the core op.  QLinearConv etc. arrive as
// single nodes.  The enum covers both styles.
enum class QuantizedOpType : uint8_t {
  Unknown,
  DequantizeLinear,
  QuantizeLinear,
  QLinearConv,
  QLinearMatMul,
  QLinearAdd,
  QLinearSigmoid,
  QLinearAveragePool,
  QLinearMul,
  // QDQ fused variants (node op_type is the core op, with Q/DQ siblings)
  QDQConv,
  QDQResize,
  QDQAveragePool,
  QDQAdd,
  QDQMul,
  QDQSub,
  QDQTranspose,
  QDQReshape,
  QDQSoftmax,
  QDQConcat,
  QDQGemm,
  QDQMatMul,
  QDQGelu,
};

enum class ConvType : uint8_t {
  Regular,
  Depthwise,
  Grouped,
  ConvTranspose,
};

// Determine quantized op type from node op_type string.
QuantizedOpType GetQuantizedOpType(const onnx::NodeProto& node,
                                    const NeuronGraphContext& ctx);

// Determine convolution subtype from node + initializer info.
ConvType GetConvType(const onnx::NodeProto& node, const NeuronGraphContext& ctx);

bool IsQuantizedConv(QuantizedOpType quant_op_type);
bool IsQuantizedPool(QuantizedOpType quant_op_type);
bool IsQuantizedMatMul(QuantizedOpType quant_op_type);
bool IsQuantizedGemm(QuantizedOpType quant_op_type);
bool IsQuantizedBinaryOp(QuantizedOpType quant_op_type);

// ---- Type/shape from graph context -----------------------------------------
bool GetShape(const NeuronGraphContext& ctx, const std::string& name, Shape& shape);
bool GetType(const NeuronGraphContext& ctx, const std::string& name, int32_t& type);

// ---- Quantization scale / zero-point helpers --------------------------------
// For plugin EP, scale and zero-point are constant initializers referenced by
// a node's inputs (e.g., input 1 is scale, input 2 is zero-point).

// Get scale and zero-point from the named initializer tensors.
NeuronStatus GetQuantizationScaleAndZeroPoint(
    const NeuronGraphContext& ctx,
    const std::string& scale_name,
    const std::string& zero_point_name,
    float& scale,
    int32_t& zero_point);

// ---- Op support checking ---------------------------------------------------
bool IsNodeSupported(const onnx::NodeProto& node,
                     const NeuronGraphContext& ctx,
                     const OpSupportCheckParams& params);

bool IsNodeSupportedInGroup(const NeuronGraphContext& ctx,
                             const OpSupportCheckParams& check_params,
                             const onnx::NodeProto& node,
                             std::unordered_set<std::string>& node_outputs_in_current_group);

bool IsValidSupportedNodeGroup(const std::vector<const onnx::NodeProto*>& group,
                                uint32_t min_group_size);

// ---- Shape utilities -------------------------------------------------------
std::string Shape2String(const Shape& shape);
uint32_t ShapeSize(const Shape& shape, size_t begin_idx, size_t end_idx);
inline uint32_t ShapeSize(const Shape& shape) {
  return ShapeSize(shape, 0, shape.size());
}

// ---- Misc helpers ----------------------------------------------------------
template <typename T>
std::string Iterable2String(const T& iter) {
  std::ostringstream os;
  os << "[ ";
  for (const auto& item : iter) {
    if constexpr (std::is_same_v<std::decay_t<decltype(item)>, int8_t>) {
      os << static_cast<int32_t>(item) << " ";
    } else if constexpr (std::is_same_v<std::decay_t<decltype(item)>, uint8_t>) {
      os << static_cast<uint32_t>(item) << " ";
    } else {
      os << item << " ";
    }
  }
  os << "]";
  return os.str();
}

// Convert ONNX int64 axes to Neuron int32, optionally handling negative axis.
std::vector<int32_t> OnnxAxesToNeuron(gsl::span<const int64_t> onnx_axes,
                                      std::optional<size_t> input_rank = std::nullopt);

std::string GetNeuronVersionDescription(const NeuronRuntimeVersion& cur_version);
bool GreaterEqualVersion(const NeuronRuntimeVersion& cur_version,
                         const NeuronRuntimeVersion& target_version);

bool EndsWith(const std::string& str, const std::string& suffix);
NeuronStatus WriteBinaryStringToFile(const std::filesystem::path& bin_file,
                                     const std::string& buffer);
NeuronStatus ReadBinaryFileToString(const std::filesystem::path& bin_file,
                                    std::string& buffer);
bool IsFileExist(const std::filesystem::path& filepath);

// Check that an initializer is constant (always true for proto initializers).
bool CheckIsConstantInitializer(const NeuronGraphContext& ctx,
                                 const onnx::NodeProto& node,
                                 const std::string& input_name,
                                 const char* input_description);

// DLA model helpers.
bool GraphHasDlaNode(const NeuronGraphContext& ctx);
bool GraphHasAdapterFormattedDlaNode(const NeuronGraphContext& ctx);
NeuronStatus PartitionDlaModel(const NeuronGraphContext& ctx,
                                int num_nodes,
                                const std::function<std::string()>& gen_metadef_name,
                                std::vector<std::vector<int>>& partitions_node_indices);

}  // namespace neuron
}  // namespace onnxruntime
