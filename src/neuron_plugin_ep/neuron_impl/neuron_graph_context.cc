// neuron_graph_context.cc
//
// NeuronGraphContext implementation.

#include "neuron_graph_context.h"

#include <cstring>  // memcpy
#include <numeric>  // iota

#include "builders/neuron_qdq_support.h"
#include "op_capability_abi/graph_payload_elision.h"

namespace onnxruntime {
namespace neuron {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
NeuronGraphContext::NeuronGraphContext(const onnx::GraphProto& g)
    : graph(g) {
  // Build initializer map.
  for (const auto& init : graph.initializer()) {
    initializers[init.name()] = &init;
  }

  // Build value_info map from inputs, outputs, and intermediate value_info.
  for (const auto& vi : graph.input()) {
    value_infos[vi.name()] = &vi;
  }
  for (const auto& vi : graph.output()) {
    // outputs may override – use the last definition
    value_infos[vi.name()] = &vi;
  }
  for (const auto& vi : graph.value_info()) {
    value_infos[vi.name()] = &vi;
  }

  // Build output_producers and node_indices maps.
  for (int i = 0; i < graph.node_size(); ++i) {
    const auto& node = graph.node(i);
    node_indices[&node] = i;
    for (const auto& out : node.output()) {
      if (!out.empty()) {
        output_producers[out] = &node;
      }
    }
  }

  BuildTopoOrder();

  // Detect DQ→Op→Q patterns and populate qdq_groups / qdq_absorbed_node_indices.
  DetectQDQGroups(graph, *this, qdq_groups, qdq_absorbed_node_indices);
}

// ---------------------------------------------------------------------------
// BuildTopoOrder
// An ONNX GraphProto stores nodes in topological order by the spec, so we
// just record them 0..N-1 for now.  A proper topological sort is added
// defensively.
// ---------------------------------------------------------------------------
void NeuronGraphContext::BuildTopoOrder() {
  topo_order.resize(graph.node_size());
  std::iota(topo_order.begin(), topo_order.end(), 0);
  // ONNX spec §4.2: nodes MUST be in topological order, so no sort needed.
}

// ---------------------------------------------------------------------------
// GetShape
// ---------------------------------------------------------------------------
Shape NeuronGraphContext::GetShape(const std::string& name) const {
  const auto* vi = GetValueInfo(name);
  if (!vi) return {};

  if (!vi->has_type()) return {};
  const auto& t = vi->type();
  if (!t.has_tensor_type()) return {};
  const auto& tt = t.tensor_type();
  if (!tt.has_shape()) return {};

  Shape s;
  for (const auto& d : tt.shape().dim()) {
    if (d.has_dim_value()) {
      s.push_back(static_cast<uint32_t>(d.dim_value()));
    } else {
      // Dynamic dim – return 0 as sentinel.
      s.push_back(0);
    }
  }
  return s;
}

// ---------------------------------------------------------------------------
// GetElementType
// ---------------------------------------------------------------------------
int32_t NeuronGraphContext::GetElementType(const std::string& name) const {
  const auto* vi = GetValueInfo(name);
  if (!vi) return 0;
  if (!vi->has_type()) return 0;
  const auto& t = vi->type();
  if (!t.has_tensor_type()) return 0;
  return t.tensor_type().elem_type();
}

// ---------------------------------------------------------------------------
// HasStaticShape
// ---------------------------------------------------------------------------
bool NeuronGraphContext::HasStaticShape(const std::string& name) const {
  const Shape s = GetShape(name);
  if (s.empty()) {
    // GetShape returns {} both for a rank-0 scalar and for a tensor whose shape
    // is unknown.  A declared scalar is static; only the unknown case is not.
    const auto* vi = GetValueInfo(name);
    return vi && vi->has_type() && vi->type().has_tensor_type() &&
           vi->type().tensor_type().has_shape() &&
           vi->type().tensor_type().shape().dim_size() == 0;
  }
  for (auto d : s) {
    if (d == 0) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// GetAttr
// ---------------------------------------------------------------------------
/*static*/
const onnx::AttributeProto* NeuronGraphContext::GetAttr(
    const onnx::NodeProto& node, const std::string& name) {
  for (const auto& attr : node.attribute()) {
    if (attr.name() == name) return &attr;
  }
  return nullptr;
}

/*static*/
int64_t NeuronGraphContext::GetAttrInt(const onnx::NodeProto& node,
                                       const std::string& name,
                                       int64_t default_val) {
  const auto* a = GetAttr(node, name);
  if (!a || a->type() != onnx::AttributeProto::INT) return default_val;
  return a->i();
}

/*static*/
float NeuronGraphContext::GetAttrFloat(const onnx::NodeProto& node,
                                        const std::string& name,
                                        float default_val) {
  const auto* a = GetAttr(node, name);
  if (!a || a->type() != onnx::AttributeProto::FLOAT) return default_val;
  return a->f();
}

/*static*/
std::string NeuronGraphContext::GetAttrString(const onnx::NodeProto& node,
                                               const std::string& name,
                                               const std::string& default_val) {
  const auto* a = GetAttr(node, name);
  if (!a || a->type() != onnx::AttributeProto::STRING) return default_val;
  return a->s();
}

/*static*/
std::vector<int64_t> NeuronGraphContext::GetAttrInts(
    const onnx::NodeProto& node, const std::string& name) {
  const auto* a = GetAttr(node, name);
  if (!a || a->type() != onnx::AttributeProto::INTS) return {};
  return {a->ints().begin(), a->ints().end()};
}

/*static*/
std::vector<float> NeuronGraphContext::GetAttrFloats(
    const onnx::NodeProto& node, const std::string& name) {
  const auto* a = GetAttr(node, name);
  if (!a || a->type() != onnx::AttributeProto::FLOATS) return {};
  return {a->floats().begin(), a->floats().end()};
}

// ---------------------------------------------------------------------------
// IsElidedInitializer
// ---------------------------------------------------------------------------
/*static*/
bool NeuronGraphContext::IsElidedInitializer(const onnx::TensorProto& tensor) {
  return op_capability::IsElidedPayload(tensor);
}

// ---------------------------------------------------------------------------
// GetFloatData
// ---------------------------------------------------------------------------
/*static*/
NeuronStatus NeuronGraphContext::GetFloatData(const onnx::TensorProto& tensor,
                                               std::vector<float>& out) {
  if (IsElidedInitializer(tensor)) {
    return NeuronStatus{false,
                        "GetFloatData: tensor '" + tensor.name() +
                            "' was elided at the op capability ABI boundary"};
  }
  if (tensor.data_type() != onnx::TensorProto::FLOAT) {
    return NeuronStatus{false, "GetFloatData: tensor is not FLOAT type"};
  }
  if (!tensor.raw_data().empty()) {
    const auto& raw = tensor.raw_data();
    size_t count = raw.size() / sizeof(float);
    out.resize(count);
    std::memcpy(out.data(), raw.data(), raw.size());
  } else {
    out.assign(tensor.float_data().begin(), tensor.float_data().end());
  }
  return NeuronStatus::OK();
}

// ---------------------------------------------------------------------------
// GetInt32Data
// ---------------------------------------------------------------------------
/*static*/
NeuronStatus NeuronGraphContext::GetInt32Data(const onnx::TensorProto& tensor,
                                               std::vector<int32_t>& out) {
  if (IsElidedInitializer(tensor)) {
    return NeuronStatus{false,
                        "GetInt32Data: tensor '" + tensor.name() +
                            "' was elided at the op capability ABI boundary"};
  }
  if (tensor.data_type() != onnx::TensorProto::INT32) {
    return NeuronStatus{false, "GetInt32Data: tensor is not INT32 type"};
  }
  if (!tensor.raw_data().empty()) {
    const auto& raw = tensor.raw_data();
    size_t count = raw.size() / sizeof(int32_t);
    out.resize(count);
    std::memcpy(out.data(), raw.data(), raw.size());
  } else {
    out.assign(tensor.int32_data().begin(), tensor.int32_data().end());
  }
  return NeuronStatus::OK();
}

// ---------------------------------------------------------------------------
// GetInt64Data
// ---------------------------------------------------------------------------
/*static*/
NeuronStatus NeuronGraphContext::GetInt64Data(const onnx::TensorProto& tensor,
                                               std::vector<int64_t>& out) {
  if (IsElidedInitializer(tensor)) {
    return NeuronStatus{false,
                        "GetInt64Data: tensor '" + tensor.name() +
                            "' was elided at the op capability ABI boundary"};
  }
  if (tensor.data_type() != onnx::TensorProto::INT64) {
    return NeuronStatus{false, "GetInt64Data: tensor is not INT64 type"};
  }
  if (!tensor.raw_data().empty()) {
    const auto& raw = tensor.raw_data();
    size_t count = raw.size() / sizeof(int64_t);
    out.resize(count);
    std::memcpy(out.data(), raw.data(), raw.size());
  } else {
    out.assign(tensor.int64_data().begin(), tensor.int64_data().end());
  }
  return NeuronStatus::OK();
}

// ---------------------------------------------------------------------------
// GetScalarFloat / GetScalarInt32
// ---------------------------------------------------------------------------
NeuronStatus NeuronGraphContext::GetScalarFloat(const std::string& name,
                                                 float& out) const {
  const auto* t = GetInitializer(name);
  NEURON_RETURN_IF_NOT(t != nullptr, "GetScalarFloat: no initializer named '", name, "'");
  std::vector<float> data;
  NEURON_RETURN_IF_ERROR(GetFloatData(*t, data));
  NEURON_RETURN_IF_NOT(!data.empty(), "GetScalarFloat: empty tensor '", name, "'");
  out = data[0];
  return NeuronStatus::OK();
}

NeuronStatus NeuronGraphContext::GetScalarInt32(const std::string& name,
                                                 int32_t& out) const {
  const auto* t = GetInitializer(name);
  NEURON_RETURN_IF_NOT(t != nullptr, "GetScalarInt32: no initializer named '", name, "'");
  // INT32 or INT8 zero-point stored as INT32 in ONNX.
  if (t->data_type() == onnx::TensorProto::INT8) {
    if (!t->raw_data().empty()) {
      out = static_cast<int32_t>(*reinterpret_cast<const int8_t*>(t->raw_data().data()));
    } else if (!t->int32_data().empty()) {
      out = t->int32_data(0);
    } else {
      return NeuronStatus{false, "GetScalarInt32(INT8): empty tensor"};
    }
    return NeuronStatus::OK();
  }
  if (t->data_type() == onnx::TensorProto::UINT8) {
    if (!t->raw_data().empty()) {
      out = static_cast<int32_t>(*reinterpret_cast<const uint8_t*>(t->raw_data().data()));
    } else if (!t->int32_data().empty()) {
      out = t->int32_data(0);
    } else {
      return NeuronStatus{false, "GetScalarInt32(UINT8): empty tensor"};
    }
    return NeuronStatus::OK();
  }
  std::vector<int32_t> data;
  NEURON_RETURN_IF_ERROR(GetInt32Data(*t, data));
  NEURON_RETURN_IF_NOT(!data.empty(), "GetScalarInt32: empty tensor '", name, "'");
  out = data[0];
  return NeuronStatus::OK();
}

}  // namespace neuron
}  // namespace onnxruntime
