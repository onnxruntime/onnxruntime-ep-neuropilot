// neuron_qdq_support.h
//
// QDQ (QuantizeLinear / DequantizeLinear) group detection for the Neuron
// standalone plugin EP.
//
// Works entirely from onnx::GraphProto — no ORT internal headers are required.
//
// Design rationale:
//   The onnxruntime-qnn out-of-tree plugin EP solves the same problem by
//   reimplementing ORT's internal QDQ::GetAllNodeUnits() over the public
//   OrtGraph* C API (see qnn_ep_utils.cc / ort_api.h in that repo).
//
//   The Neuron standalone EP takes an equivalent but simpler route: because
//   NeuronGraphContext has already converted OrtGraph* → onnx::GraphProto, all
//   graph structure (nodes, initializers, value_info) is available at the proto
//   level.  DetectQDQGroups() performs the same DQ→Op→Q pattern-matching that
//   QNN's OrtSelectorManager does, but using onnx::NodeProto / onnx::TensorProto
//   directly.
//
// Usage:
//   NeuronGraphContext calls DetectQDQGroups() in its constructor.  Results are
//   stored in ctx.qdq_groups and ctx.qdq_absorbed_node_indices.
//
//   Op builders read ctx.qdq_groups[node.output(0)] to decide whether to
//   configure quantized Neuron operand types.
//
//   ModelBuilder::AddOperations() skips nodes in qdq_absorbed_node_indices and
//   instead registers operand aliases so Conv (or other target ops) receive
//   quantized operands directly.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <onnx/onnx_pb.h>

namespace onnxruntime {
namespace neuron {

// Forward declaration — defined in neuron_graph_context.h.
class NeuronGraphContext;

// ---------------------------------------------------------------------------
// QuantParamTensor
//
// Scale and zero-point for a single quantized tensor edge extracted from a
// DequantizeLinear or QuantizeLinear node's constant initializer inputs.
// ---------------------------------------------------------------------------
struct QuantParamTensor {
  // Per-tensor fields (always populated).
  float   scale      = 1.0f;
  int32_t zero_point = 0;

  // ONNX element type of the *quantized* tensor
  // (onnx::TensorProto::INT8 = 3, UINT8 = 2, INT16 = 5, INT32 = 6).
  int32_t quant_type = 2;  // UINT8

  // Per-channel fields (populated when scale has >1 element).
  int                    axis = -1;  // -1 = per-tensor
  std::vector<float>     per_channel_scales;
  std::vector<int32_t>   per_channel_zero_points;

  bool IsPerChannel() const noexcept { return !per_channel_scales.empty(); }

  // True for INT8 / INT16 (signed) quantized tensors.
  bool IsSigned() const noexcept {
    return quant_type == 3 /*INT8*/ || quant_type == 5 /*INT16*/;
  }
};

// ---------------------------------------------------------------------------
// QDQNodeGroup
//
// Describes a single DQ* → Op → Q* pattern detected in the graph.
//
// Invariants guaranteed by DetectQDQGroups:
//   - Every DQ node in dq_node_indices has its output consumed *only* by
//     this group's target node (exclusive consumer).
//   - Every Q node in q_node_indices has its input produced *only* by the
//     target node output slot j==0 (primary output, exclusive producer).
//   - scale and zero_point come from constant initializers (not runtime values).
// ---------------------------------------------------------------------------
struct QDQNodeGroup {
  // Index into graph.node() for the core (non-DQ/Q) target node.
  int target_node_index = -1;
  const onnx::NodeProto* target_node = nullptr;

  // Indices of DQ nodes whose outputs feed into this target's inputs.
  std::vector<int> dq_node_indices;

  // Indices of Q nodes that consume this target's primary output.
  std::vector<int> q_node_indices;

  // Quant params per *input slot* of the target node.
  //   input_quant_params[j] = params from the DQ node that feeds input(j).
  //   nullopt                = input j is not DQ-wrapped (float or initializer).
  std::vector<std::optional<QuantParamTensor>> input_quant_params;

  // Quant params for the target node's primary output (from the Q node).
  // nullopt = output is not Q-wrapped (stays float).
  std::optional<QuantParamTensor> output_quant_params;

  // True if at least one input or the output carries quantization.
  bool HasAnyQuantization() const {
    for (const auto& p : input_quant_params)
      if (p.has_value()) return true;
    return output_quant_params.has_value();
  }

  // Collect all absorbed node indices (DQ + Q) for this group.
  std::vector<int> AbsorbedIndices() const {
    std::vector<int> out;
    out.insert(out.end(), dq_node_indices.begin(), dq_node_indices.end());
    out.insert(out.end(), q_node_indices.begin(), q_node_indices.end());
    return out;
  }
};

// ---------------------------------------------------------------------------
// DetectQDQGroups
//
// Scans `graph` (in topological order from `ctx`) for DQ→Op→Q patterns and
// populates:
//
//   qdq_groups       — keyed by the target node's first non-empty output name.
//                      Use ctx.qdq_groups[node.output(0)] in op builders.
//
//   absorbed_indices — set of graph.node() indices that are DQ or Q nodes
//                      absorbed into a QDQ group.  ModelBuilder::AddOperations
//                      skips these indices instead of adding explicit Neuron
//                      DEQUANTIZE / QUANTIZE operations.
//
// A node is *not* detected as a QDQ target if:
//   - It is itself a DequantizeLinear or QuantizeLinear.
//   - Its DQ inputs have multiple consumers (would corrupt other users).
//   - Any scale or zero_point is a runtime (non-initializer) value.
// ---------------------------------------------------------------------------
void DetectQDQGroups(
    const onnx::GraphProto& graph,
    const NeuronGraphContext& ctx,
    std::unordered_map<std::string, QDQNodeGroup>& qdq_groups,
    std::unordered_set<int>&                       absorbed_indices);

// ---------------------------------------------------------------------------
// ReadQuantParam
//
// Extracts QuantParamTensor from a DequantizeLinear or QuantizeLinear node.
// Returns false if scale/zero_point are not constant initializers (dynamic
// quantization) — the caller should treat such nodes as unsupported QDQ.
// ---------------------------------------------------------------------------
bool ReadQuantParam(const onnx::NodeProto&    dq_or_q_node,
                    const NeuronGraphContext& ctx,
                    QuantParamTensor&         out);

}  // namespace neuron
}  // namespace onnxruntime
