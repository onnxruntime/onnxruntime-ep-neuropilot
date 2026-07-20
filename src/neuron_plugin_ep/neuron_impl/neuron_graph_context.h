// neuron_graph_context.h
//
// NeuronGraphContext: a lightweight view over an onnx::GraphProto that
// provides the same lookup services that onnxruntime::GraphViewer used to
// provide inside neuron_impl.
//
// Key replacements:
//   GraphViewer                          → NeuronGraphContext
//   graph_viewer.GetInitializerTensors() → ctx.initializers
//   graph_viewer.IsConstantInitializer() → ctx.IsInitializer()
//   GetShapeInfoFromNodeArg(gv, name)    → ctx.GetShape(name)
//   GetNodesInTopologicalOrder()         → ctx.topo_order  (proto indices)
//
// No ORT internal headers are included here.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <onnx/onnx_pb.h>

#include "neuron_types_internal.h"
#include "builders/neuron_qdq_support.h"

namespace onnxruntime {
namespace neuron {

// Shape = vector of uint32_t dimensions (same semantics as the ORT internal
// InlinedVector<uint32_t> alias that existed in core/common/basic_types.h).
using Shape = std::vector<uint32_t>;

// ---------------------------------------------------------------------------
// NeuronGraphContext
// Constructed once per fused subgraph from its onnx::GraphProto.
// ---------------------------------------------------------------------------
class NeuronGraphContext {
 public:
  explicit NeuronGraphContext(const onnx::GraphProto& graph);

  // ---- Graph proto reference -----------------------------------------------
  const onnx::GraphProto& graph;

  // ---- Lookup tables (built in constructor) --------------------------------

  // name → TensorProto* for every initializer in the graph.
  std::unordered_map<std::string, const onnx::TensorProto*> initializers;

  // name → ValueInfoProto* covering:
  //   graph.input(), graph.output(), graph.value_info() (intermediate values)
  std::unordered_map<std::string, const onnx::ValueInfoProto*> value_infos;

  // output name → NodeProto* that produces it.
  std::unordered_map<std::string, const onnx::NodeProto*> output_producers;

  // Topologically sorted node indices into graph.node().
  // (In a valid ONNX model nodes are already in topological order, but we
  //  store the sorted list explicitly for clarity.)
  std::vector<int> topo_order;

  // ---- QDQ group data (populated by DetectQDQGroups in constructor) --------

  // Map from target-node-first-output-name → QDQNodeGroup.
  // Op builders look up:  ctx.qdq_groups[node.output(0)]
  std::unordered_map<std::string, QDQNodeGroup> qdq_groups;

  // Set of graph.node() indices that are DQ or Q nodes absorbed into a
  // QDQ group.  ModelBuilder::AddOperations() skips these indices.
  std::unordered_set<int> qdq_absorbed_node_indices;

  // Look up the QDQ group for a node by its first output name.
  // Returns nullptr if the node is not a QDQ target.
  const QDQNodeGroup* GetQDQGroup(const std::string& first_output_name) const {
    auto it = qdq_groups.find(first_output_name);
    return (it != qdq_groups.end()) ? &it->second : nullptr;
  }

  // ---- Query methods -------------------------------------------------------

  // True if `name` is a constant initializer (present in graph.initializer()).
  bool IsInitializer(const std::string& name) const {
    return initializers.count(name) != 0;
  }

  // Return the TensorProto for a constant initializer, or nullptr.
  const onnx::TensorProto* GetInitializer(const std::string& name) const {
    auto it = initializers.find(name);
    return (it != initializers.end()) ? it->second : nullptr;
  }

  // Return the ValueInfoProto for a named value (input/output/intermediate),
  // or nullptr if not found.
  const onnx::ValueInfoProto* GetValueInfo(const std::string& name) const {
    auto it = value_infos.find(name);
    return (it != value_infos.end()) ? it->second : nullptr;
  }

  // Extract the static shape of a value as a Shape (vector<uint32_t>).
  // Returns empty vector if shape is unknown or dynamic.
  Shape GetShape(const std::string& name) const;

  // Extract the ONNX element type (TensorProto::DataType enum) for a value.
  // Returns 0 (UNDEFINED) if not found.
  int32_t GetElementType(const std::string& name) const;

  // True if the value has a fully-static shape (no unknown dimensions).
  bool HasStaticShape(const std::string& name) const;

  // Return the node that produces `output_name`, or nullptr.
  const onnx::NodeProto* GetProducer(const std::string& output_name) const {
    auto it = output_producers.find(output_name);
    return (it != output_producers.end()) ? it->second : nullptr;
  }

  // ---- Attribute helpers ---------------------------------------------------

  // Get a named attribute from a node, or nullptr if not present.
  static const onnx::AttributeProto* GetAttr(const onnx::NodeProto& node,
                                              const std::string& name);

  // Get integer attribute value (default: 0).
  static int64_t GetAttrInt(const onnx::NodeProto& node,
                             const std::string& name,
                             int64_t default_val = 0);

  // Get float attribute value (default: 0.f).
  static float GetAttrFloat(const onnx::NodeProto& node,
                             const std::string& name,
                             float default_val = 0.0f);

  // Get string attribute value (default: "").
  static std::string GetAttrString(const onnx::NodeProto& node,
                                   const std::string& name,
                                   const std::string& default_val = {});

  // Get repeated-int attribute as vector (e.g., pads, strides, dilations).
  static std::vector<int64_t> GetAttrInts(const onnx::NodeProto& node,
                                           const std::string& name);

  // Get repeated-float attribute as vector.
  static std::vector<float> GetAttrFloats(const onnx::NodeProto& node,
                                           const std::string& name);

  // ---- Initializer data access helpers ------------------------------------

  // Read all float values from a TensorProto (handles raw_data and float_data).
  static NeuronStatus GetFloatData(const onnx::TensorProto& tensor,
                                   std::vector<float>& out);

  // Read all int32 values from a TensorProto.
  static NeuronStatus GetInt32Data(const onnx::TensorProto& tensor,
                                   std::vector<int32_t>& out);

  // Read all int64 values from a TensorProto.
  static NeuronStatus GetInt64Data(const onnx::TensorProto& tensor,
                                   std::vector<int64_t>& out);

  // Convenience: read single float scalar from initializer by name.
  NeuronStatus GetScalarFloat(const std::string& name, float& out) const;

  // Convenience: read single int32 scalar from initializer by name.
  NeuronStatus GetScalarInt32(const std::string& name, int32_t& out) const;

 private:
  void BuildTopoOrder();
};

}  // namespace neuron
}  // namespace onnxruntime
