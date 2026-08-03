// neuron_model_builder.cc
// Plugin EP version – uses NeuronGraphContext instead of GraphViewer/NodeUnit.
//
// Key changes from original:
//  - Constructor: NeuronGraphContext& ctx instead of GraphViewer& graph_viewer
//  - No QDQ::GetAllNodeUnits, no NodeUnit
//  - PreprocessActivations: iterates ctx_.graph.node() directly
//  - RegisterInitializers: reads onnx::TensorProto raw bytes
//  - RegisterModelInputs / RegisterModelOutputs: use ctx_.graph.input/output()
//  - AddOperations: iterates ctx_.topo_order
//  - FindActivation: uses activation_output_map_ keyed on output name
//  - InlinedVector<T> → std::vector<T>
//  - Status → NeuronStatus
//  - ORT_MAKE_STATUS → NeuronStatus{false, MakeNeuronString(...)}

#include "neuron_model_builder.h"

#include <algorithm>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <onnx/onnx_pb.h>
#include <gsl/span>

#include "../neuron_types_internal.h"
#include "../neuron_graph_context.h"
#include "helper.h"
#include "onnx_ctx_model_helper.h"
#include "op_builder_factory.h"
#include "op_builder_helpers.h"
#include "shaper.h"
#include "../neuron_lib/neuron_implementation.h"
#include "../neuron_lib/neuron_wrapper_device.h"
#include "../neuron_lib/neuron_wrapper_model.h"
#include "../neuron_lib/neuron_wrapper_operand.h"

using namespace onnxruntime::neuron::wrapper;

#define MAX_OEM_OP_STRING_LEN 100

namespace onnxruntime {
namespace neuron {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
ModelBuilder::ModelBuilder(const NeuronGraphContext& ctx,
                           const NeuronApi& neuron_handle,
                           NeuronRuntimeVersion neuron_version,
                           gsl::span<const wrapper::Device> neuron_target_devices,
                           wrapper::TargetDeviceOption target_device_option,
                           const std::string& neuron_optimization_string)
    : neuron_(neuron_handle),
      neuron_version_(neuron_version),
      ctx_(ctx),
      neuron_model_{std::make_unique<wrapper::Model>(neuron_handle)},
      shaper_{ctx},
      neuron_target_devices_(neuron_target_devices),
      target_device_option_(target_device_option),
      neuron_optimization_string_(neuron_optimization_string) {
}

// ---------------------------------------------------------------------------
// Scalar operand helpers
// ---------------------------------------------------------------------------
#define DEFINE_ADD_OPERAND_FROM_SCALAR(scalar_type, op_type)                           \
  NeuronStatus ModelBuilder::AddOperandFromScalar(scalar_type value, uint32_t& index) { \
    OperandType operandType(Type::op_type, std::vector<uint32_t>{});                   \
    NEURON_RETURN_IF_ERROR(AddNewNeuronOperand(operandType, index));                    \
    NEURON_RETURN_STATUS_ON_ERROR_WITH_NOTE(                                           \
        neuron_.NeuronModel_setOperandValue(                                           \
            neuron_model_->model_, index, &value, sizeof(value)),                      \
        "value: " + std::to_string(value));                                            \
    LOGS_NEURON(VERBOSE) << "Set Scalar Operand index: " << index << " value: " << value; \
    return NeuronStatus::OK();                                                         \
  }

DEFINE_ADD_OPERAND_FROM_SCALAR(bool, BOOL)
DEFINE_ADD_OPERAND_FROM_SCALAR(int32_t, INT32)
DEFINE_ADD_OPERAND_FROM_SCALAR(float, FLOAT32)
#undef DEFINE_ADD_OPERAND_FROM_SCALAR

// ---------------------------------------------------------------------------
// AddInitializerToSkip
// ---------------------------------------------------------------------------
void ModelBuilder::AddInitializerToSkip(const std::string& tensor_name) {
  skipped_initializers_.insert(tensor_name);
}

// ---------------------------------------------------------------------------
// ComposeNeuronModel – the main build pipeline
// ---------------------------------------------------------------------------
NeuronStatus ModelBuilder::ComposeNeuronModel() {
  NEURON_RETURN_STATUS_ON_ERROR(neuron_.NeuronModel_create(&neuron_model_->model_));
  PreprocessInitializers();
  PreprocessActivations();
  NEURON_RETURN_IF_ERROR(RegisterInitializers());
  NEURON_RETURN_IF_ERROR(RegisterModelInputs(/*add_new_operand*/ true));
  NEURON_RETURN_IF_ERROR(AddOperations());
  NEURON_RETURN_IF_ERROR(RegisterModelOutputs());
  return NeuronStatus::OK();
}

// ---------------------------------------------------------------------------
// Alignment helper
// ---------------------------------------------------------------------------
constexpr size_t kDefaultByteAlignmentForNeuron = 16;
static size_t GetPaddedByteSize(size_t size) {
  return (size + kDefaultByteAlignmentForNeuron - 1) & ~(kDefaultByteAlignmentForNeuron - 1);
}

// ---------------------------------------------------------------------------
// PreprocessInitializers
// Calls each op builder's AddInitializersToSkip so that Q/DQ params that are
// folded into quantized operand types are not registered as standalone inputs.
// ---------------------------------------------------------------------------
void ModelBuilder::PreprocessInitializers() {
  const auto& op_builders = GetOpBuilders();
  for (const auto& node : ctx_.graph.node()) {
    const auto it = op_builders.find(node.op_type());
    if (it != op_builders.end()) {
      it->second->AddInitializersToSkip(*this, node, ctx_);
    }
  }
}

// ---------------------------------------------------------------------------
// GetClipMinMax – read min/max from a Clip node (opset 6 attrs or opset 11+
// scalar initializer inputs).
// ---------------------------------------------------------------------------
static bool GetClipMinMax(const NeuronGraphContext& ctx,
                          const onnx::NodeProto& node,
                          float& min_val, float& max_val) {
  min_val = -std::numeric_limits<float>::max();
  max_val =  std::numeric_limits<float>::max();

  // Opset 11+: inputs[1] = min, inputs[2] = max (optional scalar initializers)
  if (node.input_size() >= 2 && !node.input(1).empty()) {
    const auto* t = ctx.GetInitializer(node.input(1));
    if (!t) return false;
    std::vector<float> v;
    if (!NeuronGraphContext::GetFloatData(*t, v).IsOK() || v.empty()) return false;
    min_val = v[0];
  } else {
    // Opset 6: attribute
    min_val = static_cast<float>(NeuronGraphContext::GetAttrFloat(node, "min", min_val));
  }

  if (node.input_size() >= 3 && !node.input(2).empty()) {
    const auto* t = ctx.GetInitializer(node.input(2));
    if (!t) return false;
    std::vector<float> v;
    if (!NeuronGraphContext::GetFloatData(*t, v).IsOK() || v.empty()) return false;
    max_val = v[0];
  } else {
    max_val = static_cast<float>(NeuronGraphContext::GetAttrFloat(node, "max", max_val));
  }
  return true;
}

// ---------------------------------------------------------------------------
// PreprocessActivations
// Detect Relu / Clip (Relu1, Relu6) nodes that can be fused with a preceding
// op.  Store the mapping from their *input* name to the fuse code.
// ---------------------------------------------------------------------------
void ModelBuilder::PreprocessActivations() {
  for (const auto& node : ctx_.graph.node()) {
    const auto& op = node.op_type();
    if (op == "Relu") {
      // Map the input name so FindActivation can look it up
      if (node.input_size() >= 1)
        activation_output_map_[node.input(0)] = NEURON_FUSED_RELU;
    } else if (op == "Clip") {
      float min_val, max_val;
      if (!GetClipMinMax(ctx_, node, min_val, max_val)) continue;
      int32_t fuse_code = NEURON_FUSED_NONE;
      if (min_val == -1.0f && max_val == 1.0f)
        fuse_code = NEURON_FUSED_RELU1;
      else if (min_val == 0.0f && max_val == 6.0f)
        fuse_code = NEURON_FUSED_RELU6;
      else
        continue;
      if (node.input_size() >= 1)
        activation_output_map_[node.input(0)] = fuse_code;
    }
  }
}

// ---------------------------------------------------------------------------
// Helper: get raw bytes from a TensorProto
// ---------------------------------------------------------------------------
static std::vector<uint8_t> GetTensorRawBytes(const onnx::TensorProto& tensor) {
  if (!tensor.raw_data().empty()) {
    const auto& rd = tensor.raw_data();
    return std::vector<uint8_t>(
        reinterpret_cast<const uint8_t*>(rd.data()),
        reinterpret_cast<const uint8_t*>(rd.data()) + rd.size());
  }
  // Typed fields — convert to bytes
  switch (tensor.data_type()) {
    case onnx::TensorProto::FLOAT: {
      std::vector<uint8_t> buf(tensor.float_data_size() * sizeof(float));
      memcpy(buf.data(), tensor.float_data().data(), buf.size());
      return buf;
    }
    case onnx::TensorProto::INT32:
    case onnx::TensorProto::BOOL: {
      std::vector<uint8_t> buf(tensor.int32_data_size() * sizeof(int32_t));
      memcpy(buf.data(), tensor.int32_data().data(), buf.size());
      return buf;
    }
    case onnx::TensorProto::INT64: {
      std::vector<uint8_t> buf(tensor.int64_data_size() * sizeof(int64_t));
      memcpy(buf.data(), tensor.int64_data().data(), buf.size());
      return buf;
    }
    case onnx::TensorProto::DOUBLE: {
      std::vector<uint8_t> buf(tensor.double_data_size() * sizeof(double));
      memcpy(buf.data(), tensor.double_data().data(), buf.size());
      return buf;
    }
    case onnx::TensorProto::UINT64: {
      std::vector<uint8_t> buf(tensor.uint64_data_size() * sizeof(uint64_t));
      memcpy(buf.data(), tensor.uint64_data().data(), buf.size());
      return buf;
    }
    default:
      return {};
  }
}

// ---------------------------------------------------------------------------
// RegisterInitializers
// ---------------------------------------------------------------------------
NeuronStatus ModelBuilder::RegisterInitializers() {
  const auto& graph = ctx_.graph;

  // Pre-build a map: quantized-initializer-name → first DQ node consuming it.
  // Used below so that INT8/UINT8 initializers absorbed via DQ nodes get the
  // correct scale/ZP registered with the Neuron SDK instead of scale=0.
  std::unordered_map<std::string, const onnx::NodeProto*> init_dq_consumer;
  for (const auto& node : graph.node()) {
    if (node.op_type() == "DequantizeLinear" &&
        node.input_size() >= 2 && !node.input(0).empty() &&
        ctx_.IsInitializer(node.input(0))) {
      init_dq_consumer.emplace(node.input(0), &node);
    }
  }

  // First pass: collect index / size / padded_size for each initializer to add.
  struct InitEntry { uint32_t index; size_t size; size_t padded_size; };
  std::vector<InitEntry> entries;
  size_t sizeAll = 0;

  for (const auto& tensor : graph.initializer()) {
    const auto& name = tensor.name();
    if (Contains(skipped_initializers_, name)) continue;

    Shape shape;
    for (auto dim : tensor.dims()) shape.push_back(static_cast<uint32_t>(dim));
    // Neuron treats empty-shape as dynamic-rank; treat scalar as {1}
    if (shape.empty()) shape.push_back(1);

    // Build OnnxDataTypeInfo from element type.
    // For INT8/UINT8/INT16 initializers consumed by an absorbed DQ node,
    // look up the DQ's scale/ZP so the Neuron operand gets a valid
    // quantized type (not TENSOR_QUANT8_SYMM with scale=0 which Neuron
    // rejects when the operand is later used in elementwise or conv ops).
    wrapper::OnnxDataTypeInfo type_info;
    type_info.onnx_data_type = tensor.data_type();
    const bool is_quantized_init =
        (tensor.data_type() == onnx::TensorProto::INT8  ||
         tensor.data_type() == onnx::TensorProto::UINT8 ||
         tensor.data_type() == onnx::TensorProto::INT16);
    if (is_quantized_init) {
      const auto dq_it = init_dq_consumer.find(name);
      if (dq_it != init_dq_consumer.end()) {
        const auto* dq = dq_it->second;
        float scale = 0.0f;
        if (ctx_.GetScalarFloat(dq->input(1), scale).IsOK()) {
          type_info.scales = {scale};
          int32_t zp = 0;
          if (dq->input_size() > 2 && !dq->input(2).empty())
            ctx_.GetScalarInt32(dq->input(2), zp);  // best-effort; stays 0 on fail
          type_info.zero_points = {zp};
        }
      }
    }
    OperandType operand_type = [&]() -> OperandType {
      try {
        return OperandType(type_info, shape);
      } catch (const std::exception& e) {
        throw std::runtime_error(
            std::string("RegisterInitializers: initializer '") + name +
            "' (dtype=" + std::to_string(type_info.onnx_data_type) + "): " + e.what());
      }
    }();
    shaper_.AddShape(name, shape);

    uint32_t index = 0;
    NEURON_RETURN_IF_ERROR(AddNewOperand(name, operand_type, index));
    const size_t size         = operand_type.GetOperandBlobByteSize();
    const size_t padded_size  = GetPaddedByteSize(size);
    sizeAll += padded_size;
    entries.push_back({index, size, padded_size});
  }

  if (entries.empty()) return NeuronStatus::OK();

  // Allocate shared memory for all initializers in one block.
  neuron_model_->mem_initializers_ =
      std::make_unique<wrapper::Model::NNMemory>(neuron_, "mem_initializers_", sizeAll);

  // Second pass: copy data into shared memory.
  size_t entry_idx = 0;
  size_t offset = 0;
  for (const auto& tensor : graph.initializer()) {
    const auto& name = tensor.name();
    if (Contains(skipped_initializers_, name)) continue;

    auto& e = entries[entry_idx++];
    std::vector<uint8_t> raw = GetTensorRawBytes(tensor);
    NEURON_RETURN_IF_NOT(!raw.empty() || e.size == 0,
                         "Empty raw bytes for initializer: ", name);

    uint8_t* dest = neuron_model_->mem_initializers_->GetDataPtr() + offset;
    if (!raw.empty()) {
      size_t copy_size = std::min(raw.size(), e.size);
      memcpy(dest, raw.data(), copy_size);
    }
    NEURON_RETURN_IF_ERROR(
        SetOperandValue(e.index, neuron_model_->mem_initializers_.get(), e.size, offset));
    offset += e.padded_size;
  }

  return NeuronStatus::OK();
}

// ---------------------------------------------------------------------------
// RegisterModelInputs
// ---------------------------------------------------------------------------
NeuronStatus ModelBuilder::RegisterModelInputs(bool add_new_operand) {
  // Build a map: quantized-tensor-name → DequantizeLinear node that consumes
  // it at input(0).  Used below to recover scale/ZP for quantized graph inputs
  // whose ONNX type proto carries no quantization annotation.
  std::unordered_map<std::string, const onnx::NodeProto*> dq_consumer;
  for (const auto& node : ctx_.graph.node()) {
    if (node.op_type() == "DequantizeLinear" &&
        node.input_size() >= 2 && !node.input(0).empty()) {
      dq_consumer.emplace(node.input(0), &node);
    }
  }

  for (const auto& input : ctx_.graph.input()) {
    const auto& name = input.name();

    // Skip initializers (they were handled in RegisterInitializers)
    if (ctx_.IsInitializer(name)) continue;

    // Skip if already registered
    if (Contains(operands_, name)) continue;

    Shape shape = shaper_[name];

    const auto& type_proto = input.type();
    if (!type_proto.has_tensor_type() || !type_proto.tensor_type().has_elem_type()) {
      return NeuronStatus{false, MakeNeuronString(
          "Graph input '", name, "' has no element type")};
    }
    int32_t elem_type = type_proto.tensor_type().elem_type();

    wrapper::OnnxDataTypeInfo type_info;
    type_info.onnx_data_type = elem_type;

    // ONNX type protos for quantized tensors carry no scale/ZP.  If a
    // DequantizeLinear node consumes this input we can recover the correct
    // values from its constant initializer inputs so Neuron gets a valid
    // (non-zero) scale rather than the default 0, which causes SDK hangs.
    const bool is_quantized = (elem_type == onnx::TensorProto::UINT8  ||
                               elem_type == onnx::TensorProto::INT8   ||
                               elem_type == onnx::TensorProto::INT16);
    if (is_quantized) {
      const auto dq_it = dq_consumer.find(name);
      if (dq_it != dq_consumer.end()) {
        const onnx::NodeProto* dq = dq_it->second;
        float scale = 0.0f;
        if (ctx_.GetScalarFloat(dq->input(1), scale).IsOK()) {
          type_info.scales = {scale};
          int32_t zp = 0;
          if (dq->input_size() > 2 && !dq->input(2).empty())
            ctx_.GetScalarInt32(dq->input(2), zp);  // best-effort; zp stays 0 on failure
          type_info.zero_points = {zp};
        }
      }
    }

    OperandType operand_type = [&]() -> OperandType {
      try {
        return OperandType(type_info, shape);
      } catch (const std::exception& e) {
        throw std::runtime_error(
            std::string("RegisterModelInputs: graph input '") + name +
            "' (dtype=" + std::to_string(type_info.onnx_data_type) + "): " + e.what());
      }
    }();

    if (add_new_operand) {
      uint32_t index = 0;
      NEURON_RETURN_IF_ERROR(AddNewOperand(name, operand_type, index));
      input_index_vec_.push_back(index);
    }
    neuron_model_->AddInput(name, operand_type);
  }
  return NeuronStatus::OK();
}

// ---------------------------------------------------------------------------
// RegisterModelOutputs
// ---------------------------------------------------------------------------
NeuronStatus ModelBuilder::RegisterModelOutputs() {
  for (const auto& output : ctx_.graph.output()) {
    const auto& output_name = output.name();

    if (!Contains(operands_, output_name)) {
      return NeuronStatus{false, MakeNeuronString(
          "Output '", output_name, "' is not registered as Neuron Operand")};
    }

    // Scalar outputs have {1} shape in Neuron but {} in ONNX
    const auto& output_shape = shaper_[output_name];
    if (output_shape.empty() || (output_shape.size() == 1 && output_shape[0] == 1)) {
      // Check if ONNX type is truly scalar (no dims)
      const auto& type_proto = output.type();
      if (type_proto.has_tensor_type() && type_proto.tensor_type().has_shape()) {
        if (type_proto.tensor_type().shape().dim_size() == 0) {
          neuron_model_->AddScalarOutput(output_name);
        }
      }
    }

    output_index_vec_.push_back(operand_indices_[output_name]);
    neuron_model_->AddOutput(output_name, output_name,
                             operand_types_.at(output_name));
  }
  return NeuronStatus::OK();
}

// ---------------------------------------------------------------------------
// AddQDQOperandAlias
//
// Called instead of adding a NEURON_DEQUANTIZE / NEURON_QUANTIZE operation
// when a DQ or Q node is absorbed into a QDQ group.
//
// For an absorbed DequantizeLinear:
//   dq.input(0)  → quantized activation (already registered as QUANT8 operand)
//   dq.output(0) → must alias to the same Neuron operand index, so the target
//                  Op sees a QUANT8 input instead of a FLOAT32 dequantized one.
//
// For an absorbed QuantizeLinear:
//   The target Op's output was already registered as QUANT8 by the target Op's
//   own AddToModelBuilder (which reads QDQ params from ctx_.qdq_groups).
//   q.output(0) must alias to that same operand index.
// ---------------------------------------------------------------------------
NeuronStatus ModelBuilder::AddQDQOperandAlias(const onnx::NodeProto& dq_or_q_node) {
  const bool is_dq = (dq_or_q_node.op_type() == "DequantizeLinear");

  // Source name: the operand that already exists in operand_indices_.
  //   DQ: input(0) is the quantized tensor registered as a graph input.
  //   Q:  input(0) is the target Op's output, registered by that Op's builder.
  const std::string& src_name = is_dq ? dq_or_q_node.input(0)
                                       : dq_or_q_node.input(0);
  // Destination name: the name we need to map to the same operand index.
  //   DQ: output(0) is what the target Op's input(j) refers to.
  //   Q:  output(0) is what ORT reports as the fused node's output.
  if (dq_or_q_node.output_size() == 0 || dq_or_q_node.output(0).empty())
    return NeuronStatus::OK();  // nothing to alias
  const std::string& dst_name = dq_or_q_node.output(0);

  // Look up the existing operand index for the source.
  const auto src_it = operand_indices_.find(src_name);
  if (src_it == operand_indices_.end()) {
    // Soft-fail: the source is a weight initializer that was intentionally
    // skipped by the downstream op builder (e.g., Conv's AddInitializersToSkip).
    // The op builder will register the weight operand directly with the correct
    // quantized type and layout.  No alias is needed here.
    LOGS_NEURON(VERBOSE) << "QDQ alias: skipping alias for '" << dst_name
                         << "' — source '" << src_name
                         << "' not registered (handled by downstream op builder)";
    return NeuronStatus::OK();
  }

  // Register the alias — must mirror all three tracking structures that
  // RegisterOperand() populates, so that RegisterModelOutputs()'s
  // Contains(operands_, name) check succeeds when a Q output is a graph output.
  operand_indices_[dst_name] = src_it->second;
  operands_.insert(dst_name);

  // Copy the OperandType entry so downstream builders that look up
  // operand_types_.at(dst_name) get the correct (quantized) type.
  const auto type_it = operand_types_.find(src_name);
  if (type_it != operand_types_.end()) {
    operand_types_.insert_or_assign(dst_name, type_it->second);
  }

  LOGS_NEURON(VERBOSE) << "QDQ alias: '" << dst_name << "' → operand #"
                       << src_it->second << " ('" << src_name << "')";
  return NeuronStatus::OK();
}

// ---------------------------------------------------------------------------
// AddOperations
// ---------------------------------------------------------------------------
NeuronStatus ModelBuilder::AddOperations() {
  const auto& op_builders = GetOpBuilders();
  for (int node_idx : ctx_.topo_order) {
    const auto& node = ctx_.graph.node(node_idx);

    // ── Skip DQ/Q nodes absorbed into QDQ groups ──────────────────────────
    // Instead of emitting a NEURON_DEQUANTIZE / NEURON_QUANTIZE operation,
    // register an operand alias so downstream ops see the quantized tensor
    // directly.  The Neuron SDK will then operate on quantized data natively.
    if (ctx_.qdq_absorbed_node_indices.count(node_idx)) {
      LOGS_NEURON(VERBOSE) << "QDQ: absorbing [" << node.op_type()
                           << "] node " << node_idx << " into adjacent QDQ group";
      NEURON_RETURN_IF_ERROR(AddQDQOperandAlias(node));
      continue;
    }

    LOGS_NEURON(VERBOSE) << "Adding node [" << node_idx << "] op_type: ["
                         << node.op_type() << "]";

    const auto it = op_builders.find(node.op_type());
    if (it != op_builders.end()) {
      NEURON_RETURN_IF_ERROR(it->second->AddToModelBuilder(*this, node, ctx_));
    } else if (node.op_type() == EP_CONTEXT_OP) {
      NEURON_RETURN_IF_ERROR(AddDlaOperation(node, ctx_));
    } else {
      return NeuronStatus{false, MakeNeuronString(
          "Node op_type [", node.op_type(), "] is not supported")};
    }
  }
  return NeuronStatus::OK();
}

// ---------------------------------------------------------------------------
// AddOperation
// ---------------------------------------------------------------------------
NeuronStatus ModelBuilder::AddOperation(int op,
                                         const std::vector<uint32_t>& input_indices,
                                         const std::vector<std::string>& output_names,
                                         const std::vector<OperandType>& output_types) {
  std::vector<uint32_t> output_indices;
  output_indices.reserve(output_types.size());
  for (size_t i = 0; i < output_types.size(); i++) {
    uint32_t index = 0;
    NEURON_RETURN_IF_ERROR(AddNewOperand(output_names[i], output_types[i], index));
    output_indices.push_back(index);
  }

  NEURON_RETURN_STATUS_ON_ERROR_WITH_NOTE(
      neuron_.NeuronModel_addOperation(
          neuron_model_->model_,
          static_cast<NeuronOperationType>(op),
          static_cast<uint32_t>(input_indices.size()),
          input_indices.data(),
          static_cast<uint32_t>(output_indices.size()),
          output_indices.data()),
      "op = " + std::to_string(op));

  num_neuron_ops_++;
  LOGS_NEURON(VERBOSE) << "Added Neuron Operation [" << NeuronOperationTypeToStr(op) << "]"
                       << " inputs: " << Shape2String(input_indices)
                       << " outputs: " << Shape2String(output_indices);
  return NeuronStatus::OK();
}

// ---------------------------------------------------------------------------
// AddNewOperand + AddNewNeuronOperand + RegisterOperand
// ---------------------------------------------------------------------------
NeuronStatus ModelBuilder::AddNewOperand(const std::string& name,
                                          const OperandType& operand_type,
                                          uint32_t& index) {
  LOGS_NEURON(VERBOSE) << "operand name: " << name;
  NEURON_RETURN_IF_ERROR(AddNewNeuronOperand(operand_type, index));
  RegisterOperand(name, index, operand_type);
  return NeuronStatus::OK();
}

NeuronStatus ModelBuilder::AddNewNeuronOperand(const OperandType& operand_type, uint32_t& index) {
  NEURON_RETURN_STATUS_ON_ERROR(
      neuron_.NeuronModel_addOperand(neuron_model_->model_, &operand_type.operandType));
  index = next_index_++;

  if (operand_type.channelQuant) {
    if (!GreaterEqualVersion(neuron_version_, {5, 1, 0})) {
      return NeuronStatus{false, MakeNeuronString(
          "Per-channel quant requires Neuron >= 5.1.0, current: ",
          GetNeuronVersionDescription(neuron_version_))};
    }
    NEURON_RETURN_STATUS_ON_ERROR(neuron_.NeuronModel_setOperandSymmPerChannelQuantParams(
        neuron_model_->model_, index, &operand_type.channelQuant->params));
  } else if (operand_type.asymmChannelQuant) {
    if (!GreaterEqualVersion(neuron_version_, {6, 0, 0})) {
      return NeuronStatus{false, MakeNeuronString(
          "Asymm per-channel quant requires Neuron >= 6.0.0, current: ",
          GetNeuronVersionDescription(neuron_version_))};
    }
    NEURON_RETURN_STATUS_ON_ERROR(neuron_.NeuronModel_setOperandPerChannelQuantParams(
        neuron_model_->model_, index, &operand_type.asymmChannelQuant->params));
  }
  const std::string type_str = NeuronOperandTypeToStr(static_cast<int>(operand_type.type));
  LOGS_NEURON(VERBOSE) << "Added Operand type: " << type_str
                       << " shape: " << Shape2String(operand_type.dimensions)
                       << " scale: " << operand_type.operandType.scale
                       << " zp: " << operand_type.operandType.zeroPoint
                       << " index: " << index;
  // A quantized operand with scale 0 is invalid — the hardware compiler rejects
  // it (MDLA "shift exceeds range").  Surface it loudly with the operand index
  // so the offending tensor can be traced back to its registration path.
  if (operand_type.operandType.scale == 0.0f &&
      type_str.find("QUANT") != std::string::npos && !operand_type.channelQuant &&
      !operand_type.asymmChannelQuant) {
    LOGS_NEURON(WARNING) << "Quantized operand index " << index << " (" << type_str
                         << ") registered with scale 0 — invalid; the hardware "
                            "compiler will reject it";
  }
  return NeuronStatus::OK();
}

void ModelBuilder::RegisterOperand(const std::string& name, uint32_t index,
                                    const OperandType& operand_type) {
  operand_indices_[name] = index;
  operand_types_.emplace(name, operand_type);
  operands_.insert(name);
}

// ---------------------------------------------------------------------------
// SetOperandValue
// ---------------------------------------------------------------------------
NeuronStatus ModelBuilder::SetOperandValue(uint32_t index,
                                            wrapper::Model::NNMemory* memory,
                                            size_t size, size_t offset) {
  NEURON_RETURN_STATUS_ON_ERROR(
      neuron_.NeuronModel_setOperandValue(
          neuron_model_->model_, index,
          memory->GetDataPtr() + offset,
          size));
  return NeuronStatus::OK();
}

// ---------------------------------------------------------------------------
// AddOperandFromPersistMemoryBuffer
// ---------------------------------------------------------------------------
NeuronStatus ModelBuilder::AddOperandFromPersistMemoryBuffer(
    const std::string& name, const void* buffer,
    const wrapper::OperandType& operand_type) {
  uint32_t index = 0;
  NEURON_RETURN_IF_ERROR(AddNewOperand(name, operand_type, index));
  const size_t size = operand_type.GetOperandBlobByteSize();

  if (size < NEURON_MAX_SIZE_OF_IMMEDIATELY_COPIED_VALUES) {
    NEURON_RETURN_STATUS_ON_ERROR(
        neuron_.NeuronModel_setOperandValue(neuron_model_->model_, index, buffer, size));
  } else {
    const size_t padded_size = GetPaddedByteSize(size);
    auto persist_buffer = std::make_unique<wrapper::Model::NNMemory>(neuron_, name.c_str(), padded_size);
    memcpy(persist_buffer->GetDataPtr(), buffer, size);
    NEURON_RETURN_IF_ERROR(SetOperandValue(index, persist_buffer.get(), size, 0));
    neuron_model_->mem_persist_buffers_.push_back(std::move(persist_buffer));
  }
  return NeuronStatus::OK();
}

// ---------------------------------------------------------------------------
// AddOperandFromPersistMemoryBufferWithNewLayout
// ---------------------------------------------------------------------------
NeuronStatus ModelBuilder::AddOperandFromPersistMemoryBufferWithNewLayout(
    const std::string& name, const void* buffer,
    const wrapper::OperandType& operand_type,
    ModelBuilder::DataLayout new_layout) {
  if (new_layout == ModelBuilder::DataLayout::NO_CHANGE)
    return AddOperandFromPersistMemoryBuffer(name, buffer, operand_type);

  if (operand_type.dimensions.size() != 4)
    return NeuronStatus{false, MakeNeuronString(
        "Can't transpose initializer of shape [",
        Shape2String(operand_type.dimensions), "] to new layout")};

  Shape permutation;
  if (new_layout == ModelBuilder::DataLayout::L_0231)
    permutation = {0, 2, 3, 1};
  else if (new_layout == ModelBuilder::DataLayout::L_1230)
    permutation = {1, 2, 3, 0};
  else
    return NeuronStatus{false, MakeNeuronString("Unknown layout: ", static_cast<int>(new_layout))};

  auto do_transpose = [&](auto type_tag) {
    using T = decltype(type_tag);
    Shape new_shape;
    std::vector<T> new_buffer =
        op_builder_helpers::TransposeToNewBuffer<T>(
            buffer,
            operand_type.GetOperandBlobByteSize(),
            operand_type.dimensions,
            permutation,
            new_shape);
    wrapper::OperandType new_op_type = operand_type;
    new_op_type.SetDimensions(new_shape);

    auto channel_quant = new_op_type.channelQuant;
    auto asymm_quant   = new_op_type.asymmChannelQuant;
    if (channel_quant.has_value()) {
      auto qv = channel_quant.value();
      auto it = std::find(permutation.begin(), permutation.end(), qv.params.channelDim);
      qv.params.channelDim = static_cast<uint32_t>(std::distance(permutation.begin(), it));
      channel_quant = qv;
    }
    if (asymm_quant.has_value()) {
      auto qv = asymm_quant.value();
      auto it = std::find(permutation.begin(), permutation.end(), qv.params.channelDim);
      qv.params.channelDim = static_cast<uint32_t>(std::distance(permutation.begin(), it));
      asymm_quant = qv;
    }
    new_op_type.SetNeuronOperandType(new_op_type.operandType, channel_quant, asymm_quant);
    return AddOperandFromPersistMemoryBuffer(name, new_buffer.data(), new_op_type);
  };

  switch (operand_type.GetElementByteSize()) {
    case 8: return do_transpose(int64_t{});
    case 4: return do_transpose(int32_t{});
    case 2: return do_transpose(int16_t{});
    case 1: return do_transpose(int8_t{});
    default:
      return NeuronStatus{false, MakeNeuronString(
          "Can't transpose operand of type [",
          NeuronOperandTypeToStr(operand_type.operandType.type), "]")};
  }
}

// ---------------------------------------------------------------------------
// AddOperandFromConstantInitializer (plugin EP version)
// Takes the initializer directly from ctx instead of going through NodeUnit.
// ---------------------------------------------------------------------------
NeuronStatus ModelBuilder::AddOperandFromConstantInitializer(
    const onnx::NodeProto& node,
    const NeuronGraphContext& ctx,
    int input_index,
    bool dequantize_to_float32,
    ModelBuilder::DataLayout new_layout,
    const std::string& new_operand_name) {

  NEURON_RETURN_IF_NOT(input_index < node.input_size(),
                       "AddOperandFromConstantInitializer: input index out of range");
  const auto& initializer_name = node.input(input_index);
  const auto* initializer = ctx.GetInitializer(initializer_name);

  // For QDQ ops (e.g., Conv with a DQ-wrapped weight), the input at input_index
  // may be a DQ output, not a direct initializer.  Trace through the DQ producer
  // to find the actual quantized weight initializer and read its scale/ZP.
  const onnx::NodeProto* dq_producer = nullptr;
  if (!initializer) {
    dq_producer = ctx.GetProducer(initializer_name);
    if (dq_producer && dq_producer->op_type() == "DequantizeLinear" &&
        dq_producer->input_size() > 0) {
      initializer = ctx.GetInitializer(dq_producer->input(0));
    }
  }

  NEURON_RETURN_IF_NOT(initializer != nullptr,
                       "AddOperandFromConstantInitializer: '", initializer_name,
                       "' is not an initializer");

  Shape shape;
  for (auto dim : initializer->dims()) shape.push_back(static_cast<uint32_t>(dim));
  if (shape.empty()) shape.push_back(1);
  shaper_.AddShape(initializer_name, shape);

  wrapper::OnnxDataTypeInfo type_info;
  type_info.onnx_data_type = initializer->data_type();

  // If we traced through a DQ node, populate scale/ZP so OperandType gets the
  // correct quantized type (not scale=0 QUANT8_SYMM).
  if (dq_producer) {
    QuantParamTensor qp;
    if (ReadQuantParam(*dq_producer, ctx, qp)) {
      if (qp.IsPerChannel()) {
        type_info.scales      = qp.per_channel_scales;
        type_info.zero_points = qp.per_channel_zero_points;
        type_info.channel_dim = qp.axis;
      } else {
        type_info.scales      = {qp.scale};
        type_info.zero_points = {qp.zero_point};
      }
    }
  }

  if (!dequantize_to_float32 ||
      type_info.onnx_data_type == onnx::TensorProto::FLOAT) {
    std::vector<uint8_t> raw = GetTensorRawBytes(*initializer);
    const OperandType op_type(type_info, shape);
    return AddOperandFromPersistMemoryBufferWithNewLayout(
        new_operand_name, raw.data(), op_type, new_layout);
  }

  // Dequantize to float32
  OperandType float_op_type(Type::TENSOR_FLOAT32, shape);
  std::vector<float> float_buffer;
  using op_builder_helpers::DequantizeInitializerToBuffer;
  std::optional<int64_t> axis = std::nullopt;  // no per-channel axis info in plugin EP
  switch (type_info.onnx_data_type) {
    case onnx::TensorProto::INT8:
      NEURON_RETURN_IF_ERROR(DequantizeInitializerToBuffer<int8_t>(
          *initializer, type_info.scales, type_info.zero_points, axis, float_buffer));
      break;
    case onnx::TensorProto::UINT8:
      NEURON_RETURN_IF_ERROR(DequantizeInitializerToBuffer<uint8_t>(
          *initializer, type_info.scales, type_info.zero_points, axis, float_buffer));
      break;
    case onnx::TensorProto::INT16:
      NEURON_RETURN_IF_ERROR(DequantizeInitializerToBuffer<int16_t>(
          *initializer, type_info.scales, type_info.zero_points, axis, float_buffer));
      break;
    case onnx::TensorProto::INT32:
      NEURON_RETURN_IF_ERROR(DequantizeInitializerToBuffer<int32_t>(
          *initializer, type_info.scales, type_info.zero_points, axis, float_buffer));
      break;
    default:
      return NeuronStatus{false, MakeNeuronString(
          "Cannot dequantize initializer of type: ",
          GetTensorProtoDataTypeStr(type_info.onnx_data_type))};
  }
  return AddOperandFromPersistMemoryBufferWithNewLayout(
      new_operand_name, float_buffer.data(), float_op_type, new_layout);
}

// ---------------------------------------------------------------------------
// FindActivation
// Returns a fuse code (NEURON_FUSED_RELU etc.) if the single output of `node`
// feeds directly into a fusable activation.
// ---------------------------------------------------------------------------
int32_t ModelBuilder::FindActivation(const onnx::NodeProto& node,
                                      const NeuronGraphContext& /*ctx*/) {
  if (node.output_size() != 1) return NEURON_FUSED_NONE;

  const auto& output_name = node.output(0);

  // If this output is a graph output, can't fuse
  for (const auto& gout : ctx_.graph.output()) {
    if (gout.name() == output_name) return NEURON_FUSED_NONE;
  }

  auto it = activation_output_map_.find(output_name);
  if (it == activation_output_map_.end()) return NEURON_FUSED_NONE;

  // Mark the output as fused so the activation op builder can skip adding the op
  const int32_t fuse_code = it->second;
  fused_activations_.insert(output_name);
  LOGS_NEURON(VERBOSE) << "Node [" << node.name() << "] op [" << node.op_type()
                       << "] fused activation on output [" << output_name << "]";
  return fuse_code;
}

// ---------------------------------------------------------------------------
// Composite helpers: Reshape, ExpandDims, Cast, Mul, Quantize, Dequantize
// ---------------------------------------------------------------------------
NeuronStatus ModelBuilder::AddReshape(const std::string& input,
                                       const Shape& new_shape,
                                       const std::string& output) {
  LOGS_NEURON(VERBOSE) << "AddReshape " << input << " → " << Shape2String(new_shape);
  const std::string shape_name = GetUniqueName(input + "/new_shape");
  std::vector<int32_t> shape_i32;
  shape_i32.reserve(new_shape.size());
  std::transform(new_shape.begin(), new_shape.end(), std::back_inserter(shape_i32),
                 [](uint32_t d) { return narrow<int32_t>(d); });
  return op_builder_helpers::AddNeuronReshape(*this, input, shape_name, shape_i32, output);
}

NeuronStatus ModelBuilder::AddExpandDims(const std::string& input,
                                          int32_t index_to_insert,
                                          const std::string& output) {
  auto output_shape = GetShaper()[input];
  output_shape.insert(output_shape.begin() + index_to_insert, 1);
  auto output_operand = GetOperandTypes().at(input);
  output_operand.SetDimensions(output_shape);
  std::vector<uint32_t> input_indices = {GetOperandIndices().at(input)};
  NEURON_RETURN_IF_ERROR(op_builder_helpers::AddScalarOperand(*this, input_indices, index_to_insert));
  return AddOperation(NEURON_EXPAND_DIMS, input_indices, {output}, {output_operand});
}

NeuronStatus ModelBuilder::AddCast(const std::string& input,
                                    wrapper::OperandType target_type,
                                    const std::string& output) {
  std::vector<uint32_t> input_indices = {GetOperandIndices().at(input)};
  return AddOperation(NEURON_CAST, input_indices, {output}, {target_type});
}

NeuronStatus ModelBuilder::AddMul(const std::string& input, float value,
                                   const std::string& output) {
  int32_t fuse_code = NEURON_FUSED_NONE;
  const std::string mul_value_name = GetUniqueName(input + "/mul_value");
  const OperandType& input_type = GetOperandTypes().at(input);
  OperandType value_type(input_type.type, {1});
  NEURON_RETURN_IF_ERROR(AddOperandFromPersistMemoryBuffer(mul_value_name, &value, value_type));
  std::vector<uint32_t> input_indices = {
      GetOperandIndices().at(input), GetOperandIndices().at(mul_value_name)};
  NEURON_RETURN_IF_ERROR(op_builder_helpers::AddScalarOperand(*this, input_indices, fuse_code));
  return AddOperation(NEURON_MUL, input_indices, {output}, {input_type});
}

NeuronStatus ModelBuilder::AddQuantizeLinear(const std::string& input,
                                              const wrapper::OnnxDataTypeInfo& output_type_info,
                                              const std::string& output) {
  NEURON_RETURN_IF_NOT(output_type_info.scales.size() == 1,
                       "QuantizeLinear per-channel quant not supported");
  const OperandType output_op_type(output_type_info, shaper_[output]);
  std::vector<uint32_t> input_indices = {GetOperandIndices().at(input)};
  return AddOperation(NEURON_QUANTIZE, input_indices, {output}, {output_op_type});
}

NeuronStatus ModelBuilder::AddDequantizeLinear(const std::string& input,
                                                const wrapper::OnnxDataTypeInfo& input_type_info,
                                                const std::string& output) {
  NEURON_RETURN_IF_NOT(input_type_info.scales.size() == 1,
                       "DequantizeLinear per-channel quant not supported");
  NEURON_RETURN_IF_ERROR(op_builder_helpers::IsValidInputQuantizedType(
      *this, input, input_type_info.scales[0], input_type_info.zero_points[0]));
  OperandType output_op_type(Type::TENSOR_FLOAT32, shaper_[output]);
  if (input_type_info.onnx_data_type != onnx::TensorProto::INT32) {
    std::vector<uint32_t> input_indices = {GetOperandIndices().at(input)};
    return AddOperation(NEURON_DEQUANTIZE, input_indices, {output}, {output_op_type});
  } else {
    const std::string cast_out = GetUniqueName(input + "/cast");
    NEURON_RETURN_IF_ERROR(AddCast(input, output_op_type, cast_out));
    return AddMul(cast_out, input_type_info.scales[0], output);
  }
}

// ---------------------------------------------------------------------------
// Extension / OEM / DLA helpers
// ---------------------------------------------------------------------------
NeuronStatus ModelBuilder::GetExtensionGeneralOperationType(NeuronOperationType* type) {
  NeuronOperationType operation_type = NEURON_OEM_OPERATION;
  if (neuron_.android_sdk_version > kMinSdkVersionForNeuron13) {
    NEURON_RETURN_STATUS_ON_ERROR_WITH_NOTE(
        neuron_.NeuronModel_getExtensionOperationType(
            neuron_model_->model_,
            kExtensionGeneralOpration,
            ADAPTER_EXTENSION_GENERAL_OPERATION_TYPE,
            &operation_type),
        "getting extension operation type");
  }
  *type = static_cast<NeuronOperationType>(operation_type);
  return NeuronStatus::OK();
}

NeuronStatus ModelBuilder::AddScalarExtensionOperand(const std::string& operand_name,
                                                      const uint8_t* value,
                                                      uint32_t length,
                                                      uint32_t& operand_index) {
  int32_t operand_type;
  NEURON_RETURN_STATUS_ON_ERROR_WITH_NOTE(
      neuron_.NeuronModel_getExtensionOperandType(
          neuron_model_->model_, kExtensionGeneralOpration,
          ADAPTER_EXTENSION_GENERAL_OPERAND_ARGSTRING, &operand_type),
      "getting extension operand type");

  NeuronOperandType operand_ext_type;
  operand_ext_type.type = operand_type;
  operand_ext_type.scale = 0.0f;
  operand_ext_type.zeroPoint = 0;
  operand_ext_type.dimensionCount = 1;
  uint32_t dims[1] = {length};
  operand_ext_type.dimensions = dims;

  OperandType ext_op_type(operand_ext_type);
  NEURON_RETURN_IF_ERROR(AddNewOperand(operand_name, ext_op_type, operand_index));
  NEURON_RETURN_STATUS_ON_ERROR_WITH_NOTE(
      neuron_.NeuronModel_setOperandValue(
          neuron_model_->model_, operand_index, value, length),
      "setting extension operand value");
  return NeuronStatus::OK();
}

/*
 * Pack a C-string into the OEM scalar operand encoding:
 *  [1B typeLen | N bytes type | 4B dataLen | N bytes data]
 */
// OemOperandValue is defined in neuron_model_builder.h
static int EncodeOperandValue(OemOperandValue* op, uint8_t* output) {
  if (!output) return -1;
  size_t pos = 0;
  output[pos++] = op->typeLen;
  memcpy(output + pos, op->type, op->typeLen); pos += op->typeLen;
  uint32_t* len = reinterpret_cast<uint32_t*>(&output[pos]); *len = op->dataLen; pos += 4;
  memcpy(output + pos, op->data, op->dataLen);
  return 0;
}
static size_t PackOemScalarString(const char* str, std::unique_ptr<uint8_t[]>& out) {
  if (!str) return 0;
  uint8_t type[] = {'s','t','r','i','n','g'};
  OemOperandValue ov;
  ov.typeLen = sizeof(type); ov.type = type;
  ov.dataLen = static_cast<uint32_t>(strlen(str));
  if (ov.dataLen > MAX_OEM_OP_STRING_LEN) return 0;
  std::vector<uint8_t> buf(ov.dataLen);
  ov.data = buf.data();
  memcpy(ov.data, str, ov.dataLen);
  size_t out_len = ov.typeLen + ov.dataLen + sizeof(size_t) * 2;
  out = std::make_unique<uint8_t[]>(out_len);
  if (EncodeOperandValue(&ov, out.get()) != 0) return 0;
  return out_len;
}

NeuronStatus ModelBuilder::AddCustomNameStringOperand(const std::string& custom_op_name,
                                                       const std::string& operand_name,
                                                       uint32_t& index) {
  std::unique_ptr<uint8_t[]> oem_scalar;
  uint32_t oem_size = static_cast<uint32_t>(
      PackOemScalarString(custom_op_name.c_str(), oem_scalar));
  if (oem_scalar && oem_size > 0)
    NEURON_RETURN_IF_ERROR(AddScalarExtensionOperand(operand_name, oem_scalar.get(), oem_size, index));
  return NeuronStatus::OK();
}

NeuronStatus ModelBuilder::AddDLaOperand(const onnx::NodeProto& node,
                                          const NeuronGraphContext& /*ctx*/,
                                          uint32_t& index) {
  int32_t dla_operand_data_type;
  NEURON_RETURN_STATUS_ON_ERROR_WITH_NOTE(
      neuron_.NeuronModel_getExtensionOperandType(
          neuron_model_->model_,
          RESTORE_DLA_EXTENSION_NAME,
          ADAPTER_EXTENSION_HINT_OPERAND_RESTORE_DATA_NO_COPY,
          &dla_operand_data_type),
      "Getting restore dla operand type");

  NeuronOperandType dla_op_type{
      .type = dla_operand_data_type,
      .dimensionCount = 0,
      .dimensions = nullptr,
      .scale = 0.0f,
      .zeroPoint = 0,
  };

  OperandType op_type(dla_op_type);
  NEURON_RETURN_IF_ERROR(AddNewOperand("NeuronDLAContent", op_type, index));

  const std::string context_binary =
      NeuronGraphContext::GetAttrString(node, EP_CACHE_CONTEXT, "");
  NEURON_RETURN_STATUS_ON_ERROR_WITH_NOTE(
      neuron_.NeuronModel_setOperandValue(
          neuron_model_->model_, index,
          const_cast<char*>(context_binary.c_str()),
          static_cast<uint64_t>(context_binary.length())),
      "Setting DLA operand value");
  return NeuronStatus::OK();
}

NeuronStatus ModelBuilder::AddDlaOperation(const onnx::NodeProto& node,
                                            const NeuronGraphContext& ctx) {
  if (node.input_size() == 0)
    return NeuronStatus{false, "AddDlaOperation: no inputs"};
  if (node.output_size() == 0)
    return NeuronStatus{false, "AddDlaOperation: no outputs"};

  const auto& operand_indices = GetOperandIndices();
  std::vector<uint32_t> input_indices;
  for (const auto& inp : node.input())
    input_indices.push_back(operand_indices.at(inp));

  uint32_t dla_index;
  NEURON_RETURN_IF_ERROR(AddDLaOperand(node, ctx, dla_index));
  input_indices.push_back(dla_index);

  std::vector<std::string> output_names;
  std::vector<OperandType> output_types;
  for (const auto& out_name : node.output()) {
    Shape output_shape = shaper_[out_name];
    int32_t elem_type = ctx.GetElementType(out_name);
    wrapper::OnnxDataTypeInfo type_info;
    type_info.onnx_data_type = (elem_type != 0) ? elem_type : onnx::TensorProto::FLOAT;
    OperandType op_type(type_info, output_shape);
    output_names.push_back(out_name);
    output_types.push_back(op_type);
  }

  NeuronOperationType restore_op;
  NEURON_RETURN_STATUS_ON_ERROR_WITH_NOTE(
      neuron_.NeuronModel_getExtensionOperationType(
          neuron_model_->model_,
          RESTORE_DLA_EXTENSION_NAME,
          ADAPTER_EXTENSION_HINT_OPERATION_TYPE_RESTORE,
          &restore_op),
      "Getting restore DLA operation code");

  return AddOperation(restore_op, input_indices, output_names, output_types);
}

// ---------------------------------------------------------------------------
// ValidateDeviceSupport
// ---------------------------------------------------------------------------
NeuronStatus ModelBuilder::ValidateDeviceSupport() {
  NEURON_RETURN_IF_ERROR(ComposeNeuronModel());

  LOGS_NEURON(VERBOSE) << "Identifying inputs: " << Shape2String(input_index_vec_)
                       << " outputs: " << Shape2String(output_index_vec_);
  NEURON_RETURN_STATUS_ON_ERROR_WITH_NOTE(
      neuron_.NeuronModel_identifyInputsAndOutputs(
          neuron_model_->model_,
          static_cast<uint32_t>(input_index_vec_.size()),
          input_index_vec_.data(),
          static_cast<uint32_t>(output_index_vec_.size()),
          output_index_vec_.data()),
      "identifyInputsAndOutputs");

  if (use_fp16_ && GreaterEqualVersion(neuron_version_, {4, 1, 0})) {
    NEURON_RETURN_STATUS_ON_ERROR_WITH_NOTE(
        neuron_.NeuronModel_relaxComputationFloat32toFloat16(neuron_model_->model_, true),
        "Set fp16");
  }

  NEURON_RETURN_STATUS_ON_ERROR_WITH_NOTE(
      neuron_.NeuronModel_finish(neuron_model_->model_),
      "NeuronModel_finish");

  if (neuron_target_devices_.empty()) return NeuronStatus::OK();

  std::vector<NeuronDevice*> device_handles;
  for (const auto& dev : neuron_target_devices_) {
    LOGS_NEURON(VERBOSE) << "Using device: " << dev.name;
    device_handles.push_back(dev.device);
  }

  std::unique_ptr<bool[]> supported_ops_holder =
      std::make_unique<bool[]>(num_neuron_ops_);
  NEURON_RETURN_STATUS_ON_ERROR_WITH_NOTE(
      neuron_.NeuronModel_getSupportedOperationsForDevices(
          neuron_model_->model_, device_handles.data(),
          static_cast<uint32_t>(device_handles.size()),
          supported_ops_holder.get()),
      "getSupportedOperationsForDevices");

  bool all_ok = std::all_of(supported_ops_holder.get(),
                            supported_ops_holder.get() + num_neuron_ops_,
                            [](bool s) { return s; });
  if (!all_ok) {
    LOGS_NEURON(VERBOSE) << "Some ops not supported by target devices: "
                         << GetDevicesDescription(neuron_target_devices_);
    return NeuronStatus{false, MakeNeuronString(
        "Some ops unsupported by devices: ",
        GetDevicesDescription(neuron_target_devices_))};
  }

  device_support_validated_ = true;
  return NeuronStatus::OK();
}

// ---------------------------------------------------------------------------
// Compile
// ---------------------------------------------------------------------------
NeuronStatus ModelBuilder::Compile(std::unique_ptr<wrapper::Model>& model) {
  NEURON_RETURN_IF_ERROR(ValidateDeviceSupport());

  std::vector<NeuronDevice*> device_handles;
  for (const auto& dev : neuron_target_devices_)
    device_handles.push_back(dev.device);

  const bool use_create_for_devices =
      device_support_validated_ &&
      target_device_option_ != wrapper::TargetDeviceOption::ALL_DEVICES;

  if (use_create_for_devices) {
    NEURON_RETURN_STATUS_ON_ERROR_WITH_NOTE(
        neuron_.NeuronCompilation_createForDevices(
            neuron_model_->model_, device_handles.data(),
            static_cast<uint32_t>(device_handles.size()),
            &neuron_model_->compilation_),
        "createForDevices");
  } else {
    NEURON_RETURN_STATUS_ON_ERROR_WITH_NOTE(
        neuron_.NeuronCompilation_create(neuron_model_->model_, &neuron_model_->compilation_),
        "NeuronCompilation_create");
  }

  NEURON_RETURN_STATUS_ON_ERROR_WITH_NOTE(
      neuron_.NeuronCompilation_setPreference(
          neuron_model_->compilation_, static_cast<int32_t>(exe_pref_)),
      "setPreference");

  bool is_adapter_dla = GreaterEqualVersion(neuron_version_, {4, 3, 0}) &&
                        !GreaterEqualVersion(neuron_version_, {7, 0, 0});
  if (store_dla_ && !is_adapter_dla) {
    std::string opt = "--dla-file " + dla_path_.string();
    NEURON_RETURN_STATUS_ON_ERROR_WITH_NOTE(
        neuron_.NeuronCompilation_setOptimizationString(
            neuron_model_->compilation_, opt.c_str()),
        "setting dla-file opt string");
  }

  if (!neuron_optimization_string_.empty()) {
    NEURON_RETURN_STATUS_ON_ERROR_WITH_NOTE(
        neuron_.NeuronCompilation_setOptimizationString(
            neuron_model_->compilation_, neuron_optimization_string_.c_str()),
        "setting optimization string");
  }

  NEURON_RETURN_STATUS_ON_ERROR_WITH_NOTE(
      neuron_.NeuronCompilation_finish(neuron_model_->compilation_),
      "NeuronCompilation_finish");

  LOGS_NEURON(VERBOSE) << "Compilation finished.";

  if (store_dla_ && is_adapter_dla) {
    auto* comp = neuron_model_->compilation_;
    size_t buffer_size = 0;
    NEURON_RETURN_STATUS_ON_ERROR_WITH_NOTE(
        neuron_.NeuronCompilation_getCompiledNetworkSize(comp, &buffer_size),
        "getCompiledNetworkSize");
    std::string dla_buffer(buffer_size, '\0');
    NEURON_RETURN_STATUS_ON_ERROR_WITH_NOTE(
        neuron_.NeuronCompilation_storeCompiledNetwork(comp, dla_buffer.data(), dla_buffer.size()),
        "storeCompiledNetwork");
    NEURON_RETURN_IF_ERROR(WriteBinaryStringToFile(dla_path_, dla_buffer));
  }

  model.reset(neuron_model_.release());
  return NeuronStatus::OK();
}

// ---------------------------------------------------------------------------
// RestoredFromAdapterFormattedDla
// ---------------------------------------------------------------------------
NeuronStatus ModelBuilder::RestoredFromAdapterFormattedDla(
    std::unique_ptr<wrapper::Model>& model) {
  NEURON_RETURN_IF_NOT(ctx_.graph.node_size() == 1,
                       "RestoredFromAdapterFormattedDla: expected exactly one node");
  const auto& node = ctx_.graph.node(0);
  NEURON_RETURN_IF_NOT(IsAdapterFormattedDlaNode(node),
                       "RestoredFromAdapterFormattedDla: node is not adapter DLA");

  const std::string& dla_buffer = NeuronGraphContext::GetAttrString(node, EP_CACHE_CONTEXT, "");
  auto& neuron_model = neuron_model_->model_;
  auto& neuron_comp  = neuron_model_->compilation_;
  NEURON_RETURN_IF_NOT(!neuron_model && !neuron_comp,
                       "model/compilation should not exist before restore");
  NEURON_RETURN_STATUS_ON_ERROR_WITH_NOTE(
      neuron_.NeuronModel_restoreFromCompiledNetwork(
          &neuron_model, &neuron_comp,
          dla_buffer.data(), dla_buffer.size()),
      "restoring adapter DLA");

  // Register inputs / outputs for the restored model
  PreprocessActivations();
  NEURON_RETURN_IF_ERROR(RegisterModelInputs(/*add_new_operand*/ false));

  // Register outputs from node outputs
  for (const auto& out_name : node.output()) {
    int32_t elem_type = ctx_.GetElementType(out_name);
    if (elem_type == 0) elem_type = onnx::TensorProto::FLOAT;
    wrapper::OnnxDataTypeInfo type_info;
    type_info.onnx_data_type = elem_type;
    Shape shape = shaper_[out_name];
    OperandType op_type(type_info, shape);
    neuron_model_->AddOutput(out_name, out_name, op_type);
  }

  model.reset(neuron_model_.release());
  return NeuronStatus::OK();
}

// ---------------------------------------------------------------------------
// Misc accessors
// ---------------------------------------------------------------------------
NeuronRuntimeVersion ModelBuilder::GetNeuronVersion() const {
  return neuron_version_;
}

const onnx::TensorProto* ModelBuilder::GetConstantInitializer(const std::string& name) const {
  return ctx_.GetInitializer(name);
}

bool ModelBuilder::HasDevice(const std::string& device_name) const {
  return ContainsDevice(neuron_target_devices_, device_name);
}

std::string ModelBuilder::GetUniqueName(const std::string& base_name) {
  std::string unique_name;
  do {
    std::ostringstream os;
    os << base_name << "_token_" << name_token_++;
    unique_name = os.str();
  } while (Contains(unique_names_, unique_name));
  unique_names_.insert(unique_name);
  return unique_name;
}

}  // namespace neuron
}  // namespace onnxruntime
