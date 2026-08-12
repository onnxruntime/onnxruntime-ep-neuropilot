#include "helper.h"
#ifndef NEURON_OP_CAPABILITY_LIB
#include "op_builder_factory.h"
#endif

#include <fstream>
#include <string>
#include <filesystem>

#include "../neuron_types_internal.h"
#include "../neuron_lib/neuron_wrapper_operand.h"
#include <onnx/onnx_pb.h>

namespace onnxruntime {
namespace neuron {

#define RETURN_STR_BY_CASE(type_value) \
  case (type_value):                   \
    return #type_value
std::string NeuronOperandTypeToStr(int type) {
  switch (type) {
    RETURN_STR_BY_CASE(NEURON_FLOAT32);
    RETURN_STR_BY_CASE(NEURON_INT32);
    RETURN_STR_BY_CASE(NEURON_UINT32);
    RETURN_STR_BY_CASE(NEURON_TENSOR_FLOAT32);
    RETURN_STR_BY_CASE(NEURON_TENSOR_INT32);
    RETURN_STR_BY_CASE(NEURON_TENSOR_QUANT8_ASYMM);
    RETURN_STR_BY_CASE(NEURON_BOOL);
    RETURN_STR_BY_CASE(NEURON_TENSOR_QUANT16_SYMM);
    RETURN_STR_BY_CASE(NEURON_TENSOR_FLOAT16);
    RETURN_STR_BY_CASE(NEURON_TENSOR_BOOL8);
    RETURN_STR_BY_CASE(NEURON_FLOAT16);
    RETURN_STR_BY_CASE(NEURON_TENSOR_QUANT8_SYMM_PER_CHANNEL);
    RETURN_STR_BY_CASE(NEURON_TENSOR_QUANT16_ASYMM);
    RETURN_STR_BY_CASE(NEURON_TENSOR_QUANT8_SYMM);
    RETURN_STR_BY_CASE(NEURON_TENSOR_QUANT8_ASYMM_SIGNED);
    RETURN_STR_BY_CASE(NEURON_MODEL);
    RETURN_STR_BY_CASE(NEURON_EXT_TENSOR_UINT32);
    RETURN_STR_BY_CASE(NEURON_EXT_TENSOR_QUANT8_ASYMM_PER_CHANNEL);
    RETURN_STR_BY_CASE(NEURON_EXT_TENSOR_QUANT4_ASYMM);
    RETURN_STR_BY_CASE(NEURON_EXT_TENSOR_QUANT4_ASYMM_SIGNED);
    RETURN_STR_BY_CASE(NEURON_EXT_TENSOR_QUANT4_SYMM);
    RETURN_STR_BY_CASE(NEURON_EXT_TENSOR_QUANT16_ASYMM_SIGNED);
    RETURN_STR_BY_CASE(NEURON_EXT_TENSOR_QUANT8_ASYMM_SIGNED_PER_CHANNEL);
    RETURN_STR_BY_CASE(NEURON_EXT_TENSOR_INT64);
    RETURN_STR_BY_CASE(NEURON_EXT_TENSOR_UINT64);
  }
  return "UNDEFINED_TYPE(" + std::to_string(type) + ")";
}

std::string NeuronOperationTypeToStr(int type) {
  switch (type) {
    RETURN_STR_BY_CASE(NEURON_ADD);
    RETURN_STR_BY_CASE(NEURON_AVERAGE_POOL_2D);
    RETURN_STR_BY_CASE(NEURON_CONCATENATION);
    RETURN_STR_BY_CASE(NEURON_CONV_2D);
    RETURN_STR_BY_CASE(NEURON_DEPTHWISE_CONV_2D);
    RETURN_STR_BY_CASE(NEURON_DEPTH_TO_SPACE);
    RETURN_STR_BY_CASE(NEURON_DEQUANTIZE);
    RETURN_STR_BY_CASE(NEURON_EMBEDDING_LOOKUP);
    RETURN_STR_BY_CASE(NEURON_FLOOR);
    RETURN_STR_BY_CASE(NEURON_FULLY_CONNECTED);
    RETURN_STR_BY_CASE(NEURON_HASHTABLE_LOOKUP);
    RETURN_STR_BY_CASE(NEURON_L2_NORMALIZATION);
    RETURN_STR_BY_CASE(NEURON_L2_POOL_2D);
    RETURN_STR_BY_CASE(NEURON_LOCAL_RESPONSE_NORMALIZATION);
    RETURN_STR_BY_CASE(NEURON_LOGISTIC);
    RETURN_STR_BY_CASE(NEURON_LSH_PROJECTION);
    RETURN_STR_BY_CASE(NEURON_LSTM);
    RETURN_STR_BY_CASE(NEURON_MAX_POOL_2D);
    RETURN_STR_BY_CASE(NEURON_MUL);
    RETURN_STR_BY_CASE(NEURON_RELU);
    RETURN_STR_BY_CASE(NEURON_RELU1);
    RETURN_STR_BY_CASE(NEURON_RELU6);
    RETURN_STR_BY_CASE(NEURON_RESHAPE);
    RETURN_STR_BY_CASE(NEURON_RESIZE_BILINEAR);
    RETURN_STR_BY_CASE(NEURON_RNN);
    RETURN_STR_BY_CASE(NEURON_SOFTMAX);
    RETURN_STR_BY_CASE(NEURON_SPACE_TO_DEPTH);
    RETURN_STR_BY_CASE(NEURON_SVDF);
    RETURN_STR_BY_CASE(NEURON_TANH);
    RETURN_STR_BY_CASE(NEURON_BATCH_TO_SPACE_ND);
    RETURN_STR_BY_CASE(NEURON_DIV);
    RETURN_STR_BY_CASE(NEURON_MEAN);
    RETURN_STR_BY_CASE(NEURON_PAD);
    RETURN_STR_BY_CASE(NEURON_SPACE_TO_BATCH_ND);
    RETURN_STR_BY_CASE(NEURON_SQUEEZE);
    RETURN_STR_BY_CASE(NEURON_STRIDED_SLICE);
    RETURN_STR_BY_CASE(NEURON_SUB);
    RETURN_STR_BY_CASE(NEURON_TRANSPOSE);
    RETURN_STR_BY_CASE(NEURON_ABS);
    RETURN_STR_BY_CASE(NEURON_ARGMAX);
    RETURN_STR_BY_CASE(NEURON_ARGMIN);
    RETURN_STR_BY_CASE(NEURON_AXIS_ALIGNED_BBOX_TRANSFORM);
    RETURN_STR_BY_CASE(NEURON_BIDIRECTIONAL_SEQUENCE_LSTM);
    RETURN_STR_BY_CASE(NEURON_BIDIRECTIONAL_SEQUENCE_RNN);
    RETURN_STR_BY_CASE(NEURON_BOX_WITH_NMS_LIMIT);
    RETURN_STR_BY_CASE(NEURON_CAST);
    RETURN_STR_BY_CASE(NEURON_CHANNEL_SHUFFLE);
    RETURN_STR_BY_CASE(NEURON_DETECTION_POSTPROCESSING);
    RETURN_STR_BY_CASE(NEURON_EQUAL);
    RETURN_STR_BY_CASE(NEURON_EXP);
    RETURN_STR_BY_CASE(NEURON_EXPAND_DIMS);
    RETURN_STR_BY_CASE(NEURON_GATHER);
    RETURN_STR_BY_CASE(NEURON_GENERATE_PROPOSALS);
    RETURN_STR_BY_CASE(NEURON_GREATER);
    RETURN_STR_BY_CASE(NEURON_GREATER_EQUAL);
    RETURN_STR_BY_CASE(NEURON_GROUPED_CONV_2D);
    RETURN_STR_BY_CASE(NEURON_HEATMAP_MAX_KEYPOINT);
    RETURN_STR_BY_CASE(NEURON_INSTANCE_NORMALIZATION);
    RETURN_STR_BY_CASE(NEURON_LESS);
    RETURN_STR_BY_CASE(NEURON_LESS_EQUAL);
    RETURN_STR_BY_CASE(NEURON_LOG);
    RETURN_STR_BY_CASE(NEURON_LOGICAL_AND);
    RETURN_STR_BY_CASE(NEURON_LOGICAL_NOT);
    RETURN_STR_BY_CASE(NEURON_LOGICAL_OR);
    RETURN_STR_BY_CASE(NEURON_LOG_SOFTMAX);
    RETURN_STR_BY_CASE(NEURON_MAXIMUM);
    RETURN_STR_BY_CASE(NEURON_MINIMUM);
    RETURN_STR_BY_CASE(NEURON_NEG);
    RETURN_STR_BY_CASE(NEURON_NOT_EQUAL);
    RETURN_STR_BY_CASE(NEURON_PAD_V2);
    RETURN_STR_BY_CASE(NEURON_POW);
    RETURN_STR_BY_CASE(NEURON_PRELU);
    RETURN_STR_BY_CASE(NEURON_QUANTIZE);
    RETURN_STR_BY_CASE(NEURON_QUANTIZED_16BIT_LSTM);
    RETURN_STR_BY_CASE(NEURON_RANDOM_MULTINOMIAL);
    RETURN_STR_BY_CASE(NEURON_REDUCE_ALL);
    RETURN_STR_BY_CASE(NEURON_REDUCE_ANY);
    RETURN_STR_BY_CASE(NEURON_REDUCE_MAX);
    RETURN_STR_BY_CASE(NEURON_REDUCE_MIN);
    RETURN_STR_BY_CASE(NEURON_REDUCE_PROD);
    RETURN_STR_BY_CASE(NEURON_REDUCE_SUM);
    RETURN_STR_BY_CASE(NEURON_ROI_ALIGN);
    RETURN_STR_BY_CASE(NEURON_ROI_POOLING);
    RETURN_STR_BY_CASE(NEURON_RSQRT);
    RETURN_STR_BY_CASE(NEURON_SELECT);
    RETURN_STR_BY_CASE(NEURON_SIN);
    RETURN_STR_BY_CASE(NEURON_SLICE);
    RETURN_STR_BY_CASE(NEURON_SPLIT);
    RETURN_STR_BY_CASE(NEURON_SQRT);
    RETURN_STR_BY_CASE(NEURON_TILE);
    RETURN_STR_BY_CASE(NEURON_TOPK_V2);
    RETURN_STR_BY_CASE(NEURON_TRANSPOSE_CONV_2D);
    RETURN_STR_BY_CASE(NEURON_UNIDIRECTIONAL_SEQUENCE_LSTM);
    RETURN_STR_BY_CASE(NEURON_UNIDIRECTIONAL_SEQUENCE_RNN);
    RETURN_STR_BY_CASE(NEURON_RESIZE_NEAREST_NEIGHBOR);
    RETURN_STR_BY_CASE(NEURON_IF);
    RETURN_STR_BY_CASE(NEURON_WHILE);
    RETURN_STR_BY_CASE(NEURON_ELU);
    RETURN_STR_BY_CASE(NEURON_HARD_SWISH);
    RETURN_STR_BY_CASE(NEURON_FILL);
    RETURN_STR_BY_CASE(NEURON_RANK);
    RETURN_STR_BY_CASE(NEURON_BATCH_MATMUL);
    RETURN_STR_BY_CASE(NEURON_PACK);
    RETURN_STR_BY_CASE(NEURON_MIRROR_PAD);
    RETURN_STR_BY_CASE(NEURON_MIRROR_REVERSE);
    RETURN_STR_BY_CASE(NEURON_GELU);
    RETURN_STR_BY_CASE(NEURON_GELU_V2);
    RETURN_STR_BY_CASE(NEURON_NUMBER_OF_OPERATIONS);
    RETURN_STR_BY_CASE(NEURON_OEM_OPERATION);
  }
  return "UNDEFINED_OPERATION_CODE(" + std::to_string(type) + ")";
}

std::string GetErrorCause(int error_code) {
  switch (error_code) {
    RETURN_STR_BY_CASE(NEURON_NO_ERROR);
    RETURN_STR_BY_CASE(NEURON_OUT_OF_MEMORY);
    RETURN_STR_BY_CASE(NEURON_INCOMPLETE);
    RETURN_STR_BY_CASE(NEURON_UNEXPECTED_NULL);
    RETURN_STR_BY_CASE(NEURON_BAD_DATA);
    RETURN_STR_BY_CASE(NEURON_OP_FAILED);
    RETURN_STR_BY_CASE(NEURON_BAD_STATE);
    RETURN_STR_BY_CASE(NEURON_UNMAPPABLE);
    RETURN_STR_BY_CASE(NEURON_OUTPUT_INSUFFICIENT_SIZE);
    RETURN_STR_BY_CASE(NEURON_UNAVAILABLE_DEVICE);
    RETURN_STR_BY_CASE(NEURON_MISSED_DEADLINE_TRANSIENT);
    RETURN_STR_BY_CASE(NEURON_MISSED_DEADLINE_PERSISTENT);
    RETURN_STR_BY_CASE(NEURON_RESOURCE_EXHAUSTED_TRANSIENT);
    RETURN_STR_BY_CASE(NEURON_RESOURCE_EXHAUSTED_PERSISTENT);
    RETURN_STR_BY_CASE(NEURON_DEAD_OBJECT);
  }
  return "Unknown error code: " + std::to_string(error_code);
}

std::string GetTensorProtoDataTypeStr(int tensor_element_type) {
  using namespace ONNX_NAMESPACE;
  switch (tensor_element_type) {
    RETURN_STR_BY_CASE(TensorProto_DataType_UNDEFINED);
    RETURN_STR_BY_CASE(TensorProto_DataType_FLOAT);
    RETURN_STR_BY_CASE(TensorProto_DataType_UINT8);
    RETURN_STR_BY_CASE(TensorProto_DataType_INT8);
    RETURN_STR_BY_CASE(TensorProto_DataType_UINT16);
    RETURN_STR_BY_CASE(TensorProto_DataType_INT16);
    RETURN_STR_BY_CASE(TensorProto_DataType_INT32);
    RETURN_STR_BY_CASE(TensorProto_DataType_INT64);
    RETURN_STR_BY_CASE(TensorProto_DataType_STRING);
    RETURN_STR_BY_CASE(TensorProto_DataType_BOOL);
    RETURN_STR_BY_CASE(TensorProto_DataType_FLOAT16);
    RETURN_STR_BY_CASE(TensorProto_DataType_DOUBLE);
    RETURN_STR_BY_CASE(TensorProto_DataType_UINT32);
    RETURN_STR_BY_CASE(TensorProto_DataType_UINT64);
    RETURN_STR_BY_CASE(TensorProto_DataType_COMPLEX64);
    RETURN_STR_BY_CASE(TensorProto_DataType_COMPLEX128);
    RETURN_STR_BY_CASE(TensorProto_DataType_BFLOAT16);
    RETURN_STR_BY_CASE(TensorProto_DataType_FLOAT8E4M3FN);
    RETURN_STR_BY_CASE(TensorProto_DataType_FLOAT8E4M3FNUZ);
    RETURN_STR_BY_CASE(TensorProto_DataType_FLOAT8E5M2);
    RETURN_STR_BY_CASE(TensorProto_DataType_FLOAT8E5M2FNUZ);
    RETURN_STR_BY_CASE(TensorProto_DataType_UINT4);
    RETURN_STR_BY_CASE(TensorProto_DataType_INT4);
  }
  return "UNKNOWN_DATATYPE(" + std::to_string(tensor_element_type) + ")";
}
#undef RETURN_STR_BY_CASE

QuantizedOpType GetQuantizedOpType(const onnx::NodeProto& node, const NeuronGraphContext& ctx) {
  const auto& op = node.op_type();
  if (op == "DequantizeLinear")   return QuantizedOpType::DequantizeLinear;
  if (op == "QuantizeLinear")     return QuantizedOpType::QuantizeLinear;
  if (op == "QLinearConv")        return QuantizedOpType::QLinearConv;
  if (op == "QLinearMatMul")      return QuantizedOpType::QLinearMatMul;
  if (op == "QLinearAdd")         return QuantizedOpType::QLinearAdd;
  if (op == "QLinearSigmoid")     return QuantizedOpType::QLinearSigmoid;
  if (op == "QLinearAveragePool") return QuantizedOpType::QLinearAveragePool;
  if (op == "QLinearMul")         return QuantizedOpType::QLinearMul;

  // For QDQ-wrapped standard ONNX ops, DetectQDQGroups stores the group keyed
  // by the target node's first output.  Check that map to classify correctly.
  if (node.output_size() > 0 && !node.output(0).empty() &&
      ctx.GetQDQGroup(node.output(0)) != nullptr) {
    if (op == "Conv" || op == "ConvTranspose") return QuantizedOpType::QDQConv;
    if (op == "Gemm")        return QuantizedOpType::QDQGemm;
    if (op == "MatMul")      return QuantizedOpType::QDQMatMul;
    if (op == "Add")         return QuantizedOpType::QDQAdd;
    if (op == "Mul")         return QuantizedOpType::QDQMul;
    if (op == "Sub")         return QuantizedOpType::QDQSub;
    if (op == "Transpose")   return QuantizedOpType::QDQTranspose;
    if (op == "Reshape")     return QuantizedOpType::QDQReshape;
    if (op == "Softmax")     return QuantizedOpType::QDQSoftmax;
    if (op == "Concat")      return QuantizedOpType::QDQConcat;
    if (op == "Resize")      return QuantizedOpType::QDQResize;
    if (op == "AveragePool") return QuantizedOpType::QDQAveragePool;
    if (op == "Gelu")        return QuantizedOpType::QDQGelu;
  }
  return QuantizedOpType::Unknown;
}

ConvType GetConvType(const onnx::NodeProto& node, const NeuronGraphContext& ctx) {
  if (node.op_type() == "ConvTranspose") {
    return ConvType::ConvTranspose;
  }

  int group = 1;
  for (const auto& attr : node.attribute()) {
    if (attr.name() == "group") {
      group = static_cast<int>(attr.i());
      break;
    }
  }

  // For ONNX we only have 1 conv ops
  // For NNAPI we have 3
  // Input is (N, C, H, W)
  // group == 1,                                   --> regular conv
  // group != 1 && weight is (M, 1, kH, kW),       --> depthwise conv
  // group != 1 && weight is (M, C/group, kH, kW), --> grouped conv
  if (group == 1) {
    return ConvType::Regular;
  }

  // Check weight shape from context
  if (node.input_size() >= 2) {
    const auto& weight_name = node.input(1);
    const auto* weight_tensor = ctx.GetInitializer(weight_name);
    if (weight_tensor && weight_tensor->dims_size() >= 2) {
      if (weight_tensor->dims(1) == 1) {
        return ConvType::Depthwise;
      }
    }
  }

  return ConvType::Grouped;
}

bool IsQuantizedConv(QuantizedOpType quant_op_type) {
  return (quant_op_type == QuantizedOpType::QLinearConv) ||
         (quant_op_type == QuantizedOpType::QDQConv);
}

bool IsQuantizedPool(QuantizedOpType quant_op_type) {
  return (quant_op_type == QuantizedOpType::QLinearAveragePool) ||
         (quant_op_type == QuantizedOpType::QDQAveragePool);
}

bool IsQuantizedMatMul(QuantizedOpType quant_op_type) {
  return quant_op_type == QuantizedOpType::QLinearMatMul ||
         quant_op_type == QuantizedOpType::QDQMatMul;
}

bool IsQuantizedGemm(QuantizedOpType quant_op_type) {
  return quant_op_type == QuantizedOpType::QDQGemm;
}

bool IsQuantizedBinaryOp(QuantizedOpType quant_op_type) {
  return quant_op_type == QuantizedOpType::QLinearMatMul ||
         quant_op_type == QuantizedOpType::QLinearAdd ||
         quant_op_type == QuantizedOpType::QLinearMul ||
         quant_op_type == QuantizedOpType::QDQAdd ||
         quant_op_type == QuantizedOpType::QDQMul ||
         quant_op_type == QuantizedOpType::QDQSub ||
         quant_op_type == QuantizedOpType::QDQGemm ||
         quant_op_type == QuantizedOpType::QDQMatMul ||
         IsQuantizedConv(quant_op_type);
}

bool GetShape(const NeuronGraphContext& ctx, const std::string& name, Shape& shape) {
  shape.clear();
  const auto* tensor_info = ctx.GetValueInfo(name);
  if (!tensor_info) {
    LOGS_NEURON(WARNING) << "Name [" << name << "] has no shape info";
    return false;
  }

  if (!tensor_info->has_type() || !tensor_info->type().has_tensor_type() ||
      !tensor_info->type().tensor_type().has_shape()) {
    LOGS_NEURON(WARNING) << "Name [" << name << "] has no tensor shape";
    return false;
  }

  const auto& shape_proto = tensor_info->type().tensor_type().shape();
  // Neuron uses 0 for dynamic dimension, which is the default value for dim.dim_value()
  for (const auto& dim : shape_proto.dim()) {
    shape.push_back(static_cast<uint32_t>(dim.dim_value()));
  }

  return true;
}

bool GetType(const NeuronGraphContext& ctx, const std::string& name, int32_t& type) {
  type = ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
  const auto* tensor_info = ctx.GetValueInfo(name);
  if (!tensor_info || !tensor_info->has_type() ||
      !tensor_info->type().has_tensor_type() ||
      !tensor_info->type().tensor_type().has_elem_type()) {
    LOGS_NEURON(WARNING) << "Name [" << name << "] has no input type";
    return false;
  }
  type = tensor_info->type().tensor_type().elem_type();
  return true;
}

bool CheckIsConstantInitializer(const NeuronGraphContext& ctx, const std::string& input_name,
                                const std::string& node_name, const char* input_description) {
  if (!ctx.GetInitializer(input_name)) {
    LOGS_NEURON(VERBOSE) << input_description << " of " << node_name
                          << " must be a constant initializer";
    return false;
  }
  return true;
}

bool IsInternalQuantizationSupported(const onnx::NodeProto& node,
                                     const std::unordered_set<std::string>& node_outputs_in_group) {
  const auto& op_type = node.op_type();

  // The node's input(s) have to be an output of node(s) within the group
  // If not, then this node is using graph/partition input(s) as input(s)
  // We only need to check input0 for all operators except "Concat"
  bool check_all_inputs = op_type == "Concat";
  size_t check_input_num = check_all_inputs ? static_cast<size_t>(node.input_size()) : 1;
  for (size_t i = 0; i < check_input_num; i++) {
    const auto& input_name = node.input(static_cast<int>(i));
    if (node_outputs_in_group.find(input_name) == node_outputs_in_group.end()) {
      LOGS_NEURON(VERBOSE) << "Node [" << node.name() << "] type: [" << op_type
                            << "] has input [" << input_name
                            << "] does not support using graph input(quantized) as node input";
      return false;
    }
  }

  return true;
}

#ifndef NEURON_OP_CAPABILITY_LIB
bool IsNodeSupportedInGroup(const NeuronGraphContext& ctx,
                            const OpSupportCheckParams& check_params,
                            const onnx::NodeProto& node,
                            std::unordered_set<std::string>& node_outputs_in_current_group) {
  const auto& op_builders = GetOpBuilders();
  const auto op_builder_it = op_builders.find(node.op_type());
  if (op_builder_it == op_builders.end()) {
    LOGS_NEURON(VERBOSE) << "No existing OpBuilders support this op: " << node.op_type();
    return false;
  }

  const auto* op_builder = op_builder_it->second;
  if (!op_builder->IsOpSupported(node, ctx, check_params)) {
    return false;
  }

  // Check internal quantization support for ops that pass uint8 through
  static const std::unordered_set<std::string> internal_quantized_op_types = {
    "Transpose", "Resize", "Concat", "MaxPool",
  };
  if (internal_quantized_op_types.count(node.op_type())) {
    if (node.input_size() > 0) {
      int32_t input_type = ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
      GetType(ctx, node.input(0), input_type);
      if (input_type == ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
        return IsInternalQuantizationSupported(node, node_outputs_in_current_group);
      }
    }
  }

  return true;
}
#endif  // NEURON_OP_CAPABILITY_LIB

bool IsValidSupportedNodeGroup(const std::vector<const onnx::NodeProto*>& supported_node_group,
                                uint32_t min_group_size) {
  if (supported_node_group.size() == 1) {
    const std::string& op = supported_node_group[0]->op_type();
    if (op == "Reshape" || op == "Flatten" || op == "Identity")
      return false;
  }

  if (supported_node_group.size() <= min_group_size) {
    return false;
  }

  return true;
}

std::vector<int32_t> OnnxAxesToNeuron(gsl::span<const int64_t> onnx_axes, std::optional<size_t> input_rank) {
  std::vector<int32_t> result;
  result.reserve(onnx_axes.size());
  for (auto dim : onnx_axes) {
    if (input_rank.has_value()) {
      dim = HandleNegativeAxis(dim, *input_rank);
    }
    result.push_back(static_cast<int32_t>(dim));
  }
  return result;
}

std::string GetNeuronVersionDescription(const NeuronRuntimeVersion& cur_version) {
  return MakeNeuronString(std::to_string(cur_version.major), ".",
                          std::to_string(cur_version.minor), ".",
                          std::to_string(cur_version.patch));
}

bool GreaterEqualVersion(const NeuronRuntimeVersion& cur_version, const NeuronRuntimeVersion& target_version) {
  if (cur_version.major > target_version.major) return true;
  if (cur_version.major < target_version.major) return false;
  if (cur_version.minor > target_version.minor) return true;
  if (cur_version.minor < target_version.minor) return false;
  return cur_version.patch >= target_version.patch;
}

bool EndsWith(const std::string& str, const std::string& suffix) {
  if (str.size() < suffix.size()) return false;
  return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string Shape2String(const Shape& shape) {
  return Iterable2String(shape);
}

uint32_t ShapeSize(const Shape& shape, size_t begin_idx, size_t end_idx) {
  ORT_ENFORCE(begin_idx <= end_idx && begin_idx <= shape.size(),
              "Invalid indices: begin [", begin_idx, "], end [", end_idx, "], shape size [", shape.size(), "]");
  uint32_t result = 1;
  for (size_t i = begin_idx; i < end_idx; ++i) {
    result *= shape[i];
  }
  return result;
}

NeuronStatus WriteBinaryStringToFile(const std::filesystem::path& bin_file, const std::string& buffer) {
  std::ofstream output(bin_file, std::fstream::binary);
  if (!output) {
    LOGS_NEURON(ERROR) << "Failed to open created binary file: " << bin_file << ".";
    return NeuronStatus(NeuronStatusCode::FAIL, "Failed to open dla file.");
  }

  LOGS_NEURON(ERROR) << "WriteBinaryStringToFile: " << bin_file << ".";
  output.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
  output.close();
  return NeuronStatus::OK();
}

NeuronStatus ReadBinaryFileToString(const std::filesystem::path& bin_file, std::string& buffer) {
  std::ifstream input(bin_file, std::fstream::binary);
  if (!input) {
    LOGS_NEURON(ERROR) << "Failed to open created binary file: " << bin_file << ".";
    return NeuronStatus(NeuronStatusCode::FAIL, "Failed to open dla file.");
  }

  input.seekg(0, input.end);
  int64_t length = input.tellg();
  input.seekg(0, input.beg);
  buffer = std::string(length, '\0');
  input.read(buffer.data(), buffer.size());
  input.close();
  return NeuronStatus::OK();
}

bool IsFileExist(const std::filesystem::path& filepath) {
  return std::filesystem::is_regular_file(filepath) && std::filesystem::exists(filepath);
}

NeuronStatus GetQuantizationScaleAndZeroPoint(
    const NeuronGraphContext& ctx,
    const std::string& scale_name,
    const std::string& zero_point_name,
    float& scale,
    int32_t& zero_point) {
  scale = 0.0f;
  zero_point = 0;

  const auto* s = ctx.GetInitializer(scale_name);
  NEURON_RETURN_IF_NOT(s, "Scale initializer not found: ", scale_name);
  std::vector<float> scale_data;
  NEURON_RETURN_IF_ERROR(NeuronGraphContext::GetFloatData(*s, scale_data));
  NEURON_RETURN_IF_NOT(!scale_data.empty(), "Empty scale data: ", scale_name);
  scale = scale_data[0];

  if (!zero_point_name.empty()) {
    const auto* zp = ctx.GetInitializer(zero_point_name);
    NEURON_RETURN_IF_NOT(zp, "Zero-point initializer not found: ", zero_point_name);
    std::vector<int32_t> zp_data;
    NEURON_RETURN_IF_ERROR(NeuronGraphContext::GetInt32Data(*zp, zp_data));
    if (!zp_data.empty()) {
      const auto dt = zp->data_type();
      if (dt == onnx::TensorProto::INT8)
        zero_point = static_cast<int8_t>(zp_data[0]);
      else if (dt == onnx::TensorProto::UINT8)
        zero_point = static_cast<uint8_t>(zp_data[0]);
      else
        zero_point = zp_data[0];
    }
  }
  return NeuronStatus::OK();
}

}  // namespace neuron
}  // namespace onnxruntime
