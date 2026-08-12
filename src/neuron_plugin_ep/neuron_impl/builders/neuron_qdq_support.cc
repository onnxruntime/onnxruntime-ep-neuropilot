// neuron_qdq_support.cc
//
// QDQ group detection at the ONNX GraphProto level.
// No ORT internal headers.  Reads only onnx::GraphProto, onnx::NodeProto,
// onnx::TensorProto via the public ONNX protobuf API.

#include "neuron_qdq_support.h"
#include "../neuron_graph_context.h"

#include <cstring>  // memcpy

namespace onnxruntime {
namespace neuron {

// ============================================================================
// Internal helpers — proto-level tensor data extraction
// ============================================================================
namespace {

// Read all float values from a TensorProto (float_data or raw_data).
static std::vector<float> ReadFloatArray(const onnx::TensorProto& tp) {
  std::vector<float> result;
  if (tp.float_data_size() > 0) {
    result.assign(tp.float_data().begin(), tp.float_data().end());
    return result;
  }
  if (!tp.raw_data().empty()) {
    const size_t n = tp.raw_data().size() / sizeof(float);
    result.resize(n);
    std::memcpy(result.data(), tp.raw_data().data(), n * sizeof(float));
  }
  return result;
}

// Read a scalar float from a TensorProto.
static bool ReadScalarFloat(const onnx::TensorProto& tp, float& out) {
  const auto v = ReadFloatArray(tp);
  if (v.size() == 1) { out = v[0]; return true; }
  return false;
}

// Read all integer values from a TensorProto, normalising to int32_t.
// Handles INT8 (data_type=3), UINT8 (data_type=2), INT32 (data_type=6)
// stored in raw_data or in the appropriate repeated field.
static std::vector<int32_t> ReadIntArray(const onnx::TensorProto& tp) {
  std::vector<int32_t> result;
  const int dt = tp.data_type();

  if (dt == onnx::TensorProto::INT32 && tp.int32_data_size() > 0) {
    result.assign(tp.int32_data().begin(), tp.int32_data().end());
    return result;
  }
  if (!tp.raw_data().empty()) {
    if (dt == onnx::TensorProto::INT32) {
      const size_t n = tp.raw_data().size() / 4;
      result.resize(n);
      std::memcpy(result.data(), tp.raw_data().data(), n * 4);
    } else {
      // INT8 or UINT8 — each element is 1 byte
      const size_t n = tp.raw_data().size();
      result.resize(n);
      for (size_t i = 0; i < n; ++i) {
        const auto raw = static_cast<uint8_t>(tp.raw_data()[i]);
        if (dt == onnx::TensorProto::INT8 && raw > 127)
          result[i] = static_cast<int32_t>(raw) - 256;
        else
          result[i] = static_cast<int32_t>(raw);
      }
    }
    return result;
  }
  // Per ONNX spec, INT8/UINT8 zero_points may be stored in int32_data rather
  // than raw_data (both are legal encodings).  Handle that case here so that
  // per-channel weight DQ nodes are not silently dropped when their zero_point
  // tensor uses int32_data storage.
  if ((dt == onnx::TensorProto::INT8 || dt == onnx::TensorProto::UINT8) &&
      tp.int32_data_size() > 0) {
    result.reserve(static_cast<size_t>(tp.int32_data_size()));
    for (int v : tp.int32_data()) {
      if (dt == onnx::TensorProto::INT8)
        result.push_back(static_cast<int32_t>(static_cast<int8_t>(v & 0xFF)));
      else
        result.push_back(static_cast<int32_t>(static_cast<uint8_t>(v & 0xFF)));
    }
    return result;
  }
  // If nothing matched, return empty — caller will treat as unsupported.
  return result;
}

static bool ReadScalarInt32(const onnx::TensorProto& tp, int32_t& out) {
  const auto v = ReadIntArray(tp);
  if (v.size() == 1) { out = v[0]; return true; }
  // A scalar zero_point may also be stored as a 0-dimensional INT8 tensor
  // with raw_data of 1 byte and dims().empty().
  if (v.empty() && tp.dims().empty() && !tp.raw_data().empty()) {
    const auto raw = static_cast<uint8_t>(tp.raw_data()[0]);
    out = (tp.data_type() == onnx::TensorProto::INT8 && raw > 127)
              ? static_cast<int32_t>(raw) - 256
              : static_cast<int32_t>(raw);
    return true;
  }
  return false;
}

}  // anonymous namespace

// ============================================================================
// ReadQuantParam
// ============================================================================
bool ReadQuantParam(const onnx::NodeProto&    node,
                    const NeuronGraphContext& ctx,
                    QuantParamTensor&         out) {
  // DequantizeLinear / QuantizeLinear signature:
  //   input[0] = tensor to (de)quantize
  //   input[1] = scale       (required, float scalar or 1-D)
  //   input[2] = zero_point  (optional, int8/uint8/int32 scalar or 1-D)

  if (node.input_size() < 2) return false;

  const std::string& scale_name = node.input(1);
  const onnx::TensorProto* scale_tp = ctx.GetInitializer(scale_name);
  if (!scale_tp) return false;  // dynamic scale — not supported for QDQ fusion

  const std::vector<float> scales = ReadFloatArray(*scale_tp);
  if (scales.empty()) return false;

  // Determine quantized data type from the zero_point tensor's data_type.
  // Default to UINT8 when no zero_point is provided.
  out.quant_type = onnx::TensorProto::UINT8;

  if (node.input_size() >= 3 && !node.input(2).empty()) {
    const std::string& zp_name = node.input(2);
    const onnx::TensorProto* zp_tp = ctx.GetInitializer(zp_name);
    if (!zp_tp) return false;  // dynamic zero_point

    out.quant_type = zp_tp->data_type();

    if (scales.size() > 1) {
      // Per-channel quantization.
      const auto zps = ReadIntArray(*zp_tp);
      if (zps.empty()) return false;
      out.per_channel_scales      = scales;
      out.per_channel_zero_points = zps;
      out.axis = static_cast<int>(
          NeuronGraphContext::GetAttrInt(node, "axis", 1));
    } else {
      // Per-tensor quantization.
      out.scale = scales[0];
      if (!ReadScalarInt32(*zp_tp, out.zero_point)) return false;
      out.axis  = -1;
    }
  } else {
    // No zero_point input — implicit zero (per-tensor).
    if (scales.size() != 1) return false;
    out.scale      = scales[0];
    out.zero_point = 0;
    out.axis       = -1;
    // Determine quant_type from the quantized tensor's element type.
    // For DQ: the quantized tensor is input(0); for Q: it is output(0).
    // This prevents INT8 tensors with omitted zero_point from being
    // misclassified as UINT8, which would make IsSigned() return false
    // and cause TENSOR_QUANT8_ASYMM to be used instead of ASYMM_SIGNED.
    {
      const bool is_dq = (node.op_type() == "DequantizeLinear");
      const std::string& quant_tensor_name =
          is_dq ? node.input(0) : (node.output_size() > 0 ? node.output(0) : "");
      if (!quant_tensor_name.empty()) {
        int32_t elem_type = ctx.GetElementType(quant_tensor_name);
        if (elem_type == 0) {
          // value_info missing — fall back to the initializer's data_type.
          const auto* init = ctx.GetInitializer(quant_tensor_name);
          if (init) elem_type = init->data_type();
        }
        if (elem_type == onnx::TensorProto::INT8 ||
            elem_type == onnx::TensorProto::INT16) {
          out.quant_type = elem_type;
        }
        // UINT8 and others keep the default.
      }
    }
  }

  return true;
}

// ============================================================================
// DetectQDQGroups
// ============================================================================
void DetectQDQGroups(
    const onnx::GraphProto& graph,
    const NeuronGraphContext& ctx,
    std::unordered_map<std::string, QDQNodeGroup>& qdq_groups,
    std::unordered_set<int>&                       absorbed_indices) {

  const int num_nodes = graph.node_size();
  if (num_nodes == 0) return;

  // ---- Build edge maps -------------------------------------------------------

  // output_name → index of the node that produces it.
  std::unordered_map<std::string, int> output_to_node_idx;
  output_to_node_idx.reserve(static_cast<size_t>(num_nodes));

  // output_name → list of (consumer_node_idx, input_slot) pairs.
  struct ConsumerEdge { int node_idx; int input_slot; };
  std::unordered_map<std::string, std::vector<ConsumerEdge>> consumers;

  for (int i = 0; i < num_nodes; ++i) {
    const auto& node = graph.node(i);
    for (const auto& out : node.output())
      if (!out.empty()) output_to_node_idx[out] = i;
    for (int j = 0; j < node.input_size(); ++j)
      if (!node.input(j).empty())
        consumers[node.input(j)].push_back({i, j});
  }

  // Count the distinct consumer nodes of a producer's output (used to decide
  // whether a DequantizeLinear is shared across multiple target ops).
  auto distinct_consumer_count = [&](const std::string& tensor) -> int {
    const auto it = consumers.find(tensor);
    if (it == consumers.end()) return 0;
    std::unordered_set<int> nodes;
    for (const auto& e : it->second) nodes.insert(e.node_idx);
    return static_cast<int>(nodes.size());
  };

  // ---- Pass 1: build candidate groups (no finalisation yet) -----------------
  // Unlike the previous implementation, a DequantizeLinear feeding MULTIPLE
  // targets is allowed to be absorbed — but only tentatively.  A shared DQ is
  // kept only if EVERY one of its consumers is a target that also absorbs it
  // (resolved in pass 2).  This mirrors ORT's EnsureUniqueDQForNodeUnit: a DQ
  // that fans out to several quantized ops is fine, whereas one that also feeds
  // a float/unsupported op must stay on CPU so no consumer sees a mixed type.
  std::vector<QDQNodeGroup> candidates;
  candidates.reserve(static_cast<size_t>(num_nodes));

  // dq node index → set of target node indices that tentatively absorbed it.
  std::unordered_map<int, std::unordered_set<int>> dq_absorbing_targets;
  // dq node index → number of distinct consumer nodes of its output.
  std::unordered_map<int, int> dq_total_consumers;

  for (int i : ctx.topo_order) {
    if (i < 0 || i >= num_nodes) continue;
    const auto& node = graph.node(i);
    const std::string& op = node.op_type();

    // DQ/Q nodes are surrounding nodes, not targets.
    if (op == "DequantizeLinear" || op == "QuantizeLinear") continue;

    QDQNodeGroup group;
    group.target_node_index = i;
    group.target_node       = &node;
    group.input_quant_params.resize(static_cast<size_t>(node.input_size()));

    bool found_q = false;

    // --- Scan inputs for DQ producers (shared DQ allowed, resolved later) -----
    for (int j = 0; j < node.input_size(); ++j) {
      const std::string& inp = node.input(j);
      if (inp.empty() || ctx.IsInitializer(inp)) continue;

      const auto prod_it = output_to_node_idx.find(inp);
      if (prod_it == output_to_node_idx.end()) continue;
      const int  prod_idx  = prod_it->second;
      const auto& producer = graph.node(prod_idx);

      if (producer.op_type() != "DequantizeLinear") continue;

      QuantParamTensor qp;
      if (!ReadQuantParam(producer, ctx, qp)) continue;

      group.input_quant_params[static_cast<size_t>(j)] = std::move(qp);
      group.dq_node_indices.push_back(prod_idx);

      dq_absorbing_targets[prod_idx].insert(i);
      if (!dq_total_consumers.count(prod_idx))
        dq_total_consumers[prod_idx] = distinct_consumer_count(inp);
    }

    // --- Scan outputs for Q consumers (single-consumer only) ------------------
    if (node.output_size() > 0 && !node.output(0).empty()) {
      const std::string& primary_out = node.output(0);
      const auto con_it = consumers.find(primary_out);
      if (con_it != consumers.end() && con_it->second.size() == 1) {
        const int consumer_idx = con_it->second[0].node_idx;
        const auto& consumer   = graph.node(consumer_idx);
        if (consumer.op_type() == "QuantizeLinear") {
          QuantParamTensor qp;
          if (ReadQuantParam(consumer, ctx, qp)) {
            group.output_quant_params = std::move(qp);
            group.q_node_indices.push_back(consumer_idx);
            found_q = true;
          }
        }
      }
    }

    // Handle  edges with explicit NEURON_QUANTIZE / NEURON_DEQUANTIZE.
    if (group.dq_node_indices.empty() || !found_q) continue;
    candidates.push_back(std::move(group));
  }

  // ---- Pass 2: resolve shared DQs -------------------------------------------
  // A DQ is safe to absorb only if the number of targets that absorbed it equals
  // its total number of distinct consumers (i.e. no float/unsupported consumer).
  std::unordered_set<int> reverted_dq;
  for (const auto& kv : dq_absorbing_targets) {
    const int dq_idx = kv.first;
    const int absorbed_by = static_cast<int>(kv.second.size());
    const int total = dq_total_consumers[dq_idx];
    if (absorbed_by < total) reverted_dq.insert(dq_idx);
  }

  // ---- Pass 3: finalise groups ----------------------------------------------
  for (auto& group : candidates) {
    const auto& node = *group.target_node;

    // Drop any reverted (unsafely-shared) DQ from this group.
    if (!reverted_dq.empty()) {
      std::vector<int> kept_dq;
      for (int dq_idx : group.dq_node_indices) {
        if (reverted_dq.count(dq_idx)) {
          // Clear the corresponding input_quant_params entry.
          for (int j = 0; j < node.input_size(); ++j) {
            const auto prod_it = output_to_node_idx.find(node.input(j));
            if (prod_it != output_to_node_idx.end() && prod_it->second == dq_idx)
              group.input_quant_params[static_cast<size_t>(j)].reset();
          }
        } else {
          kept_dq.push_back(dq_idx);
        }
      }
      group.dq_node_indices = std::move(kept_dq);
    }

    const bool found_dq = !group.dq_node_indices.empty();
    const bool found_q  = !group.q_node_indices.empty();
    if (!found_dq && !found_q) continue;

    // Reject if any activation input has a DQ producer that was NOT absorbed
    // (mixed QUANT8/FLOAT32 operands → Neuron "input types incompatible").
    {
      bool any_unabsorbed_dq = false;
      for (int j = 0; j < node.input_size() && !any_unabsorbed_dq; ++j) {
        const auto& inp = node.input(j);
        if (inp.empty() || ctx.IsInitializer(inp)) continue;
        const auto prod_it = output_to_node_idx.find(inp);
        if (prod_it == output_to_node_idx.end()) continue;  // graph input
        if (graph.node(prod_it->second).op_type() != "DequantizeLinear") continue;
        if (!group.input_quant_params[static_cast<size_t>(j)].has_value())
          any_unabsorbed_dq = true;
      }
      if (any_unabsorbed_dq) continue;
    }

    // Reject if DQ-absorbed inputs would be mixed with non-DQ float inputs
    // (incl. float initializers), which Neuron equally rejects.
    if (found_dq) {
      bool any_non_dq_activation = false;
      for (int j = 0; j < node.input_size() && !any_non_dq_activation; ++j) {
        const auto& inp = node.input(j);
        if (inp.empty()) continue;
        if (ctx.IsInitializer(inp)) {
          const auto* init = ctx.GetInitializer(inp);
          if (init && (init->data_type() == onnx::TensorProto::FLOAT ||
                       init->data_type() == onnx::TensorProto::FLOAT16)) {
            any_non_dq_activation = true;
          }
          continue;
        }
        if (!group.input_quant_params[static_cast<size_t>(j)].has_value())
          any_non_dq_activation = true;
      }
      if (any_non_dq_activation) continue;
    }

    // Mark absorbed DQ/Q node indices (idempotent for a shared DQ used by
    // several surviving groups).
    for (int idx : group.dq_node_indices) absorbed_indices.insert(idx);
    for (int idx : group.q_node_indices)  absorbed_indices.insert(idx);

    // Key by target node's first non-empty output.
    std::string key;
    for (const auto& out : node.output())
      if (!out.empty()) { key = out; break; }
    if (key.empty()) continue;

    qdq_groups.emplace(std::move(key), std::move(group));
  }
}

}  // namespace neuron
}  // namespace onnxruntime
