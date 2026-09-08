// Copyright (c) MediaTek Inc. All rights reserved.
// Licensed under the MIT License.

// Builder for com.microsoft.GroupQueryAttention (Llama-style decode/prefill).
// Standalone plugin EP version — uses onnx::NodeProto + NeuronGraphContext.
//
// Neuron has no fused attention op, so the node is decomposed into primitive
// ops: rotary via SLICE/MUL/SUB/ADD/CONCAT, KV-cache write via keep-mask MUL +
// one-hot BATCH_MATMUL outer product + ADD, then BATCH_MATMUL → mask ADD →
// SOFTMAX → BATCH_MATMUL.
//
// The lowering is all-int8 (MDLA refuses unrelaxed float, GATHER, CAST and
// QUANTIZE/DEQUANTIZE), which the capability checker turns into a contract on
// the model: int8 KV cache with per-layer scales at inputs 12/13, per-tensor
// INT8 QDQ around Q/K/V and the output.  Position-dependent tensors (rotary
// rows, KV-write selectors, causal mask) cannot be derived on the NPU, so for
// decode they are host-computed "synth" inputs (wrapper::Model::SynthInput)
// filled from seqlens_k each inference; prefill needs constants only.
//
// Scales: q_rot carries the attention scale in its declared scale, logits use
// kDefaultLogitsScale, softmax output is fixed (1/256, -128), and the mask holds
// codes {0, -128} at kMaskScale so the ADD saturates masked logits to -128.

#include <cmath>

#include "base_op_builder.h"

namespace onnxruntime {
namespace neuron {

using namespace onnxruntime::neuron::op_builder_helpers;
using namespace onnxruntime::neuron::wrapper;

namespace {

// Input slots of com.microsoft.GroupQueryAttention (int8-KV-cache form).
enum : int {
  kQuery = 0, kKey = 1, kValue = 2, kPastKey = 3, kPastValue = 4,
  kSeqlensK = 5, kTotalSeqLen = 6, kCosCache = 7, kSinCache = 8,
  kKScale = 12, kVScale = 13,
};

constexpr float kDefaultLogitsScale = 0.157f;  // covers ~±20 nats pre-softmax
constexpr float kMaskScale = 0.5f;             // code -128 -> -64 real
constexpr float kProbsScale = 1.0f / 256.0f;
constexpr int32_t kProbsZeroPoint = -128;

// Shared operand names (one instance per compiled partition; every GQA node in
// the partition reuses them).
constexpr const char* kCosRowQ = "_gqa_shared/cos_row_q";
constexpr const char* kSinRowQ = "_gqa_shared/sin_row_q";
constexpr const char* kCosRowK = "_gqa_shared/cos_row_k";
constexpr const char* kSinRowK = "_gqa_shared/sin_row_k";
constexpr const char* kKeep = "_gqa_shared/keep";
constexpr const char* kOneHot = "_gqa_shared/onehot";
constexpr const char* kMaskRow = "_gqa_shared/mask_row";
constexpr const char* kCosTabQ = "_gqa_shared/cos_tab_q";
constexpr const char* kSinTabQ = "_gqa_shared/sin_tab_q";
constexpr const char* kCosTabK = "_gqa_shared/cos_tab_k";
constexpr const char* kSinTabK = "_gqa_shared/sin_tab_k";
constexpr const char* kMask4 = "_gqa_shared/causal_mask";
constexpr const char* kOne = "_gqa_shared/one";

// Everything the lowering needs, resolved once up front.
struct GqaParams {
  uint32_t num_heads, kv_heads, group, head_size, half, seq_len, max_seq;
  float scale_attr;
  float sq, sk, sv;      // q/k/v activation scales (from the QDQ group)
  int32_t zq, zk, zv;
  float so; int32_t zo;  // output scale (absorbed output QuantizeLinear)
  float kcs, vcs;        // cache scales (k_scale/v_scale inputs)
  float s_qrot, s_q;     // rotated-Q scale; declared (scale_attr-folded) value
};

NeuronStatus ResolveParams(const onnx::NodeProto& node, const NeuronGraphContext& ctx,
                           GqaParams& p) {
  const auto* qdq = ctx.GetQDQGroup(node.output(0));
  if (qdq == nullptr || !qdq->output_quant_params.has_value())
    return NeuronStatus{false, "GQA: missing QDQ group"};
  for (const int i : {kQuery, kKey, kValue}) {
    if (!qdq->input_quant_params[static_cast<size_t>(i)].has_value())
      return NeuronStatus{false, MakeNeuronString("GQA: input ", i, " has no quant params")};
  }
  p.num_heads = static_cast<uint32_t>(NeuronGraphContext::GetAttrInt(node, "num_heads", 0));
  p.kv_heads = static_cast<uint32_t>(NeuronGraphContext::GetAttrInt(node, "kv_num_heads", 0));
  if (p.num_heads == 0 || p.kv_heads == 0 || p.num_heads % p.kv_heads != 0)
    return NeuronStatus{false, "GQA: bad head counts"};
  p.group = p.num_heads / p.kv_heads;

  const Shape q_shape = ctx.GetShape(node.input(kQuery));
  const Shape past_shape = ctx.GetShape(node.input(kPastKey));
  if (q_shape.size() != 3 || past_shape.size() != 4)
    return NeuronStatus{false, "GQA: unexpected query/past rank"};
  p.seq_len = q_shape[1];
  p.head_size = q_shape[2] / p.num_heads;
  p.half = p.head_size / 2;
  p.max_seq = past_shape[2];
  if (p.head_size == 0 || p.head_size % 2 != 0 || p.seq_len == 0 || p.seq_len > p.max_seq)
    return NeuronStatus{false, "GQA: unsupported geometry"};

  p.scale_attr = NeuronGraphContext::GetAttrFloat(
      node, "scale", 1.0f / std::sqrt(static_cast<float>(p.head_size)));

  const auto& qp = *qdq->input_quant_params[kQuery];
  const auto& kp = *qdq->input_quant_params[kKey];
  const auto& vp = *qdq->input_quant_params[kValue];
  p.sq = qp.scale; p.zq = qp.zero_point;
  p.sk = kp.scale; p.zk = kp.zero_point;
  p.sv = vp.scale; p.zv = vp.zero_point;
  p.so = qdq->output_quant_params->scale;
  p.zo = qdq->output_quant_params->zero_point;

  NEURON_RETURN_IF_ERROR(ctx.GetScalarFloat(node.input(kKScale), p.kcs));
  NEURON_RETURN_IF_ERROR(ctx.GetScalarFloat(node.input(kVScale), p.vcs));
  if (p.kcs <= 0.0f || p.vcs <= 0.0f)
    return NeuronStatus{false, "GQA: non-positive cache scale"};

  const float q_amp = static_cast<float>(std::max(127 - p.zq, 128 + p.zq));
  p.s_qrot = std::sqrt(2.0f) * p.sq * q_amp / 127.0f;
  p.s_q = p.scale_attr * p.s_qrot;
  return NeuronStatus::OK();
}

// Rows [0, rows) of a FLOAT [*, half] rotary-cache initializer.
NeuronStatus SliceRotaryCache(const NeuronGraphContext& ctx, const std::string& name,
                              uint32_t rows, uint32_t half, std::vector<float>& out) {
  const auto* init = ctx.GetInitializer(name);
  if (init == nullptr) return NeuronStatus{false, "GQA: rotary cache is not an initializer"};
  std::vector<float> full;
  NEURON_RETURN_IF_ERROR(NeuronGraphContext::GetFloatData(*init, full));
  if (full.size() < static_cast<size_t>(rows) * half)
    return NeuronStatus{false, "GQA: rotary cache shorter than the KV buffer"};
  out.assign(full.begin(), full.begin() + static_cast<size_t>(rows) * half);
  return NeuronStatus::OK();
}

// int8 codes round(x*127) for a float table.
std::vector<int8_t> TrigCodes(const std::vector<float>& table) {
  std::vector<int8_t> codes(table.size());
  for (size_t i = 0; i < table.size(); ++i)
    codes[i] = static_cast<int8_t>(std::lround(table[i] * 127.0f));
  return codes;
}

}  // namespace

class GroupQueryAttentionOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder,
                             const onnx::NodeProto& node,
                             const NeuronGraphContext& ctx) const override;

  // Attention is the reason to claim this partition at all.
  bool IsComputeHeavy(const onnx::NodeProto& /*node*/) const override { return true; }

 private:
  NeuronStatus AddToModelBuilderImpl(ModelBuilder& model_builder,
                                     const onnx::NodeProto& node,
                                     const NeuronGraphContext& ctx) const override;

  NeuronStatus BuildDecode(ModelBuilder& mb, const onnx::NodeProto& node,
                           const NeuronGraphContext& ctx, const GqaParams& p) const;
  NeuronStatus BuildPrefill(ModelBuilder& mb, const onnx::NodeProto& node,
                            const NeuronGraphContext& ctx, const GqaParams& p) const;

  // SLICE halves, 4x MUL by the cos/sin rows, SUB/ADD, CONCAT back.  Works for
  // rank-2 [rows, head] (decode; row operands [1, half]) and rank-3
  // [seq, heads, head] (prefill; table operands [seq, 1, half]).
  NeuronStatus EmitRotary(ModelBuilder& mb, const std::string& base,
                          const std::string& src, const Shape& src_shape,
                          float src_scale, int32_t src_zp,
                          const std::string& cos_name, const std::string& sin_name,
                          float out_scale, const std::string& out_name) const;

  bool IsQuantizedOp(const onnx::NodeProto& node,
                     const NeuronGraphContext& ctx) const override {
    return ctx.GetQDQGroup(node.output(0)) != nullptr;
  }

  // The QK^T and P·V matmuls lower to NEURON_BATCH_MATMUL.
  NeuronRuntimeVersion GetMinSupportedNeuronVersion(
      const onnx::NodeProto& /*node*/,
      const neuron::OpSupportCheckParams& /*params*/) const override {
    return {5, 1, 2};
  }

  bool IsNodeUnitTypeSupported(const onnx::NodeProto& /*node*/) const override { return true; }
};

// ---------------------------------------------------------------------------

void GroupQueryAttentionOpBuilder::AddInitializersToSkip(
    ModelBuilder& model_builder,
    const onnx::NodeProto& node,
    const NeuronGraphContext& ctx) const {
  // The rotary caches are exported at the model's full context length (131072
  // rows for Llama 3.2), which is both far larger than this graph addresses and
  // over MDLA's 65535 per-dimension limit.  The builder consumes sliced copies;
  // keep the originals (and the float cache-scale scalars) out of the Neuron
  // model entirely.
  for (const int idx : {kCosCache, kSinCache, kKScale, kVScale}) {
    if (node.input_size() > idx && !node.input(idx).empty())
      model_builder.AddInitializerToSkip(node.input(idx));
  }
  // seqlens_k / total_sequence_length are consumed on the HOST only (synth
  // inputs are derived from seqlens_k in ComputeImpl); nothing on the NPU can
  // read them, so they must not become Neuron model inputs.
  for (const int idx : {kSeqlensK, kTotalSeqLen}) {
    if (node.input_size() > idx && !node.input(idx).empty())
      model_builder.AddInputToSkip(node.input(idx));
  }
  // Prefill ignores the past cache by contract (positions start at 0, the
  // cache is empty); don't register it as an unused Neuron input.
  const Shape q_shape = ctx.GetShape(node.input(kQuery));
  if (q_shape.size() == 3 && q_shape[1] > 1) {
    model_builder.AddInputToSkip(node.input(kPastKey));
    model_builder.AddInputToSkip(node.input(kPastValue));
  }
}

NeuronStatus GroupQueryAttentionOpBuilder::EmitRotary(
    ModelBuilder& mb, const std::string& base, const std::string& src,
    const Shape& src_shape, const float src_scale, const int32_t src_zp,
    const std::string& cos_name, const std::string& sin_name,
    const float out_scale, const std::string& out_name) const {
  auto& shaper = mb.GetShaper();
  const auto& idx = mb.GetOperandIndices();
  const size_t rank = src_shape.size();
  const uint32_t head = src_shape[rank - 1];
  const uint32_t half = head / 2;

  Shape half_shape = src_shape;
  half_shape[rank - 1] = half;

  // SLICE the two halves.
  std::vector<int32_t> begin0(rank, 0), begin1(rank, 0), size_v;
  begin1[rank - 1] = static_cast<int32_t>(half);
  for (const uint32_t d : half_shape) size_v.push_back(static_cast<int32_t>(d));
  const Shape idx_shape{static_cast<uint32_t>(rank)};
  const OperandType idx_type(Type::TENSOR_INT32, idx_shape);
  const std::string b0 = mb.GetUniqueName(base + "/b0");
  const std::string b1 = mb.GetUniqueName(base + "/b1");
  const std::string sz = mb.GetUniqueName(base + "/sz");
  NEURON_RETURN_IF_ERROR(mb.AddOperandFromPersistMemoryBuffer(b0, begin0.data(), idx_type));
  NEURON_RETURN_IF_ERROR(mb.AddOperandFromPersistMemoryBuffer(b1, begin1.data(), idx_type));
  NEURON_RETURN_IF_ERROR(mb.AddOperandFromPersistMemoryBuffer(sz, size_v.data(), idx_type));

  const OperandType half_in(Type::TENSOR_QUANT8_ASYMM_SIGNED, half_shape, src_scale, src_zp);
  const OperandType half_out(Type::TENSOR_QUANT8_ASYMM_SIGNED, half_shape, out_scale, 0);
  const std::string x1 = mb.GetUniqueName(base + "/x1");
  const std::string x2 = mb.GetUniqueName(base + "/x2");
  shaper.AddShape(x1, half_shape);
  shaper.AddShape(x2, half_shape);
  NEURON_RETURN_IF_ERROR(mb.AddOperation(NEURON_SLICE, {idx.at(src), idx.at(b0), idx.at(sz)},
                                         {x1}, {half_in}));
  NEURON_RETURN_IF_ERROR(mb.AddOperation(NEURON_SLICE, {idx.at(src), idx.at(b1), idx.at(sz)},
                                         {x2}, {half_in}));

  // x1*cos, x2*sin, x2*cos, x1*sin; then q1 = x1c - x2s, q2 = x2c + x1s.
  auto mul = [&](const char* tag, const std::string& a,
                 const std::string& b, std::string& out) -> NeuronStatus {
    out = mb.GetUniqueName(base + tag);
    shaper.AddShape(out, half_shape);
    std::vector<uint32_t> in{idx.at(a), idx.at(b)};
    ADD_SCALAR_OPERAND(mb, in, static_cast<int32_t>(NEURON_FUSED_NONE));
    return mb.AddOperation(NEURON_MUL, in, {out}, {half_out});
  };
  std::string x1c, x2s, x2c, x1s;
  NEURON_RETURN_IF_ERROR(mul("/x1c", x1, cos_name, x1c));
  NEURON_RETURN_IF_ERROR(mul("/x2s", x2, sin_name, x2s));
  NEURON_RETURN_IF_ERROR(mul("/x2c", x2, cos_name, x2c));
  NEURON_RETURN_IF_ERROR(mul("/x1s", x1, sin_name, x1s));

  auto addsub = [&](const char* tag, int op, const std::string& a,
                    const std::string& b, std::string& out) -> NeuronStatus {
    out = mb.GetUniqueName(base + tag);
    shaper.AddShape(out, half_shape);
    std::vector<uint32_t> in{idx.at(a), idx.at(b)};
    ADD_SCALAR_OPERAND(mb, in, static_cast<int32_t>(NEURON_FUSED_NONE));
    return mb.AddOperation(op, in, {out}, {half_out});
  };
  std::string r1, r2;
  NEURON_RETURN_IF_ERROR(addsub("/r1", NEURON_SUB, x1c, x2s, r1));
  NEURON_RETURN_IF_ERROR(addsub("/r2", NEURON_ADD, x2c, x1s, r2));

  shaper.AddShape(out_name, src_shape);
  const OperandType out_type(Type::TENSOR_QUANT8_ASYMM_SIGNED, src_shape, out_scale, 0);
  std::vector<uint32_t> cin{idx.at(r1), idx.at(r2)};
  ADD_SCALAR_OPERAND(mb, cin, static_cast<int32_t>(rank - 1));
  return mb.AddOperation(NEURON_CONCATENATION, cin, {out_name}, {out_type});
}

// ---------------------------------------------------------------------------
// Decode (S == 1): host-fed position machinery, KV write via keep/one-hot.
// ---------------------------------------------------------------------------
NeuronStatus GroupQueryAttentionOpBuilder::BuildDecode(
    ModelBuilder& mb, const onnx::NodeProto& node, const NeuronGraphContext& ctx,
    const GqaParams& p) const {
  auto& shaper = mb.GetShaper();
  const auto& idx = mb.GetOperandIndices();
  const std::string& base = node.name();

  // ---- Shared host-fed inputs (idempotent across the partition's GQA nodes) --
  const Shape row_shape{1, p.half};
  const float s_row_q = p.scale_attr / 127.0f;
  const float s_row_k = 1.0f / 127.0f;
  uint32_t unused = 0;
  using SI = wrapper::Model::SynthInput;
  NEURON_RETURN_IF_ERROR(mb.AddSynthModelInput(
      kCosRowQ, OperandType(Type::TENSOR_QUANT8_ASYMM_SIGNED, row_shape, s_row_q, 0),
      SI::kCosRow, unused));
  NEURON_RETURN_IF_ERROR(mb.AddSynthModelInput(
      kSinRowQ, OperandType(Type::TENSOR_QUANT8_ASYMM_SIGNED, row_shape, s_row_q, 0),
      SI::kSinRow, unused));
  NEURON_RETURN_IF_ERROR(mb.AddSynthModelInput(
      kCosRowK, OperandType(Type::TENSOR_QUANT8_ASYMM_SIGNED, row_shape, s_row_k, 0),
      SI::kCosRow, unused));
  NEURON_RETURN_IF_ERROR(mb.AddSynthModelInput(
      kSinRowK, OperandType(Type::TENSOR_QUANT8_ASYMM_SIGNED, row_shape, s_row_k, 0),
      SI::kSinRow, unused));
  NEURON_RETURN_IF_ERROR(mb.AddSynthModelInput(
      kKeep, OperandType(Type::TENSOR_QUANT8_ASYMM_SIGNED, Shape{1, p.max_seq, 1}, 1.0f, 0),
      SI::kKeep, unused));
  NEURON_RETURN_IF_ERROR(mb.AddSynthModelInput(
      kOneHot,
      OperandType(Type::TENSOR_QUANT8_ASYMM_SIGNED, Shape{p.kv_heads, p.max_seq, 1}, 1.0f, 0),
      SI::kOneHot, unused));
  NEURON_RETURN_IF_ERROR(mb.AddSynthModelInput(
      kMaskRow,
      OperandType(Type::TENSOR_QUANT8_ASYMM_SIGNED, Shape{1, 1, p.max_seq}, kMaskScale, 0),
      SI::kMask, unused));

  // Host-side fp32 rotary tables for the fill (shared; first node wins).
  {
    std::vector<float> cos_rows, sin_rows;
    NEURON_RETURN_IF_ERROR(
        SliceRotaryCache(ctx, node.input(kCosCache), p.max_seq, p.half, cos_rows));
    NEURON_RETURN_IF_ERROR(
        SliceRotaryCache(ctx, node.input(kSinCache), p.max_seq, p.half, sin_rows));
    mb.SetSynthRotaryTables(std::move(cos_rows), std::move(sin_rows), p.half);
    mb.SetSynthSeqlensName(node.input(kSeqlensK));
  }

  // ---- Q: [1,1,H] -> [num_heads, head] -> rotate -> [kv, group, head] --------
  const std::string q2d = mb.GetUniqueName(base + "/q2d");
  NEURON_RETURN_IF_ERROR(mb.AddReshape(node.input(kQuery), {p.num_heads, p.head_size}, q2d));
  const std::string q_rot = mb.GetUniqueName(base + "/q_rot");
  NEURON_RETURN_IF_ERROR(EmitRotary(mb, base + "/q", q2d, {p.num_heads, p.head_size},
                                    p.sq, p.zq, kCosRowQ, kSinRowQ, p.s_q, q_rot));
  const std::string q_r = mb.GetUniqueName(base + "/q_r");
  NEURON_RETURN_IF_ERROR(mb.AddReshape(q_rot, {p.kv_heads, p.group, p.head_size}, q_r));

  // ---- K: [1,1,kv*head] -> [kv, head] -> rotate -> [kv, 1, head] -------------
  const std::string k2d = mb.GetUniqueName(base + "/k2d");
  NEURON_RETURN_IF_ERROR(mb.AddReshape(node.input(kKey), {p.kv_heads, p.head_size}, k2d));
  const std::string k_rot = mb.GetUniqueName(base + "/k_rot");
  NEURON_RETURN_IF_ERROR(EmitRotary(mb, base + "/k", k2d, {p.kv_heads, p.head_size},
                                    p.sk, p.zk, kCosRowK, kSinRowK, p.kcs, k_rot));
  const std::string k_r = mb.GetUniqueName(base + "/k_r");
  NEURON_RETURN_IF_ERROR(mb.AddReshape(k_rot, {p.kv_heads, 1, p.head_size}, k_r));

  // ---- V: [1,1,kv*head] -> [kv, 1, head], still at (sv, zv) ------------------
  const std::string v_r = mb.GetUniqueName(base + "/v_r");
  NEURON_RETURN_IF_ERROR(mb.AddReshape(node.input(kValue), {p.kv_heads, 1, p.head_size}, v_r));

  // ---- KV write: present = past*keep + onehot (x) new -----------------------
  const Shape cache3{p.kv_heads, p.max_seq, p.head_size};
  auto kv_write = [&](const char* tag, const std::string& past_name,
                      const std::string& new_name, float cache_scale,
                      const std::string& present_out, std::string& present3) -> NeuronStatus {
    const OperandType cache_type(Type::TENSOR_QUANT8_ASYMM_SIGNED, cache3, cache_scale, 0);
    const std::string past3 = mb.GetUniqueName(base + tag + std::string("/past3"));
    NEURON_RETURN_IF_ERROR(mb.AddReshape(past_name, cache3, past3));

    const std::string kept = mb.GetUniqueName(base + tag + std::string("/kept"));
    shaper.AddShape(kept, cache3);
    std::vector<uint32_t> in{idx.at(past3), idx.at(kKeep)};
    ADD_SCALAR_OPERAND(mb, in, static_cast<int32_t>(NEURON_FUSED_NONE));
    NEURON_RETURN_IF_ERROR(mb.AddOperation(NEURON_MUL, in, {kept}, {cache_type}));

    // One-hot outer product places the new row; broadcast with an output larger
    // than both inputs is refused by MDLA, BATCH_MATMUL is not.
    const std::string placed = mb.GetUniqueName(base + tag + std::string("/placed"));
    shaper.AddShape(placed, cache3);
    std::vector<uint32_t> bin{idx.at(kOneHot), idx.at(new_name)};
    ADD_SCALAR_OPERAND(mb, bin, false);
    ADD_SCALAR_OPERAND(mb, bin, false);
    NEURON_RETURN_IF_ERROR(mb.AddOperation(NEURON_BATCH_MATMUL, bin, {placed}, {cache_type}));

    present3 = mb.GetUniqueName(base + tag + std::string("/present3"));
    shaper.AddShape(present3, cache3);
    std::vector<uint32_t> ain{idx.at(kept), idx.at(placed)};
    ADD_SCALAR_OPERAND(mb, ain, static_cast<int32_t>(NEURON_FUSED_NONE));
    NEURON_RETURN_IF_ERROR(mb.AddOperation(NEURON_ADD, ain, {present3}, {cache_type}));

    // The rank-4 graph output; the rank-3 operand feeds the attention matmuls.
    return mb.AddReshape(present3, {1, p.kv_heads, p.max_seq, p.head_size}, present_out);
  };
  std::string present_k3, present_v3;
  NEURON_RETURN_IF_ERROR(kv_write("/k", node.input(kPastKey), k_r, p.kcs,
                                  node.output(1), present_k3));
  NEURON_RETURN_IF_ERROR(kv_write("/v", node.input(kPastValue), v_r, p.vcs,
                                  node.output(2), present_v3));

  // ---- Attention core --------------------------------------------------------
  const Shape logits_shape{p.kv_heads, p.group, p.max_seq};
  const OperandType logits_type(Type::TENSOR_QUANT8_ASYMM_SIGNED, logits_shape,
                                kDefaultLogitsScale, 0);
  const std::string logits = mb.GetUniqueName(base + "/logits");
  shaper.AddShape(logits, logits_shape);
  {
    std::vector<uint32_t> in{idx.at(q_r), idx.at(present_k3)};
    ADD_SCALAR_OPERAND(mb, in, false);
    ADD_SCALAR_OPERAND(mb, in, true);  // adj_y: K rows are [seq, head]
    NEURON_RETURN_IF_ERROR(mb.AddOperation(NEURON_BATCH_MATMUL, in, {logits}, {logits_type}));
  }
  const std::string masked = mb.GetUniqueName(base + "/masked");
  shaper.AddShape(masked, logits_shape);
  {
    std::vector<uint32_t> in{idx.at(logits), idx.at(kMaskRow)};
    ADD_SCALAR_OPERAND(mb, in, static_cast<int32_t>(NEURON_FUSED_NONE));
    NEURON_RETURN_IF_ERROR(mb.AddOperation(NEURON_ADD, in, {masked}, {logits_type}));
  }
  const std::string probs = mb.GetUniqueName(base + "/probs");
  shaper.AddShape(probs, logits_shape);
  const OperandType probs_type(Type::TENSOR_QUANT8_ASYMM_SIGNED, logits_shape,
                               kProbsScale, kProbsZeroPoint);
  {
    std::vector<uint32_t> in{idx.at(masked)};
    ADD_SCALAR_OPERAND(mb, in, 1.0f);
    ADD_SCALAR_OPERAND(mb, in, static_cast<int32_t>(2));
    NEURON_RETURN_IF_ERROR(mb.AddOperation(NEURON_SOFTMAX, in, {probs}, {probs_type}));
  }
  const Shape attn_shape{p.kv_heads, p.group, p.head_size};
  const OperandType attn_type(Type::TENSOR_QUANT8_ASYMM_SIGNED, attn_shape, p.so, p.zo);
  const std::string attn = mb.GetUniqueName(base + "/attn");
  shaper.AddShape(attn, attn_shape);
  {
    std::vector<uint32_t> in{idx.at(probs), idx.at(present_v3)};
    ADD_SCALAR_OPERAND(mb, in, false);
    ADD_SCALAR_OPERAND(mb, in, false);
    NEURON_RETURN_IF_ERROR(mb.AddOperation(NEURON_BATCH_MATMUL, in, {attn}, {attn_type}));
  }
  return mb.AddReshape(attn, {1, 1, p.num_heads * p.head_size}, node.output(0));
}

// ---------------------------------------------------------------------------
// Prefill (S > 1): everything position-dependent is a constant; the past cache
// is empty by contract (positions 0..S-1).
// ---------------------------------------------------------------------------
NeuronStatus GroupQueryAttentionOpBuilder::BuildPrefill(
    ModelBuilder& mb, const onnx::NodeProto& node, const NeuronGraphContext& ctx,
    const GqaParams& p) const {
  auto& shaper = mb.GetShaper();
  const auto& idx = mb.GetOperandIndices();
  const std::string& base = node.name();
  const uint32_t S = p.seq_len;
  const uint32_t rest = p.max_seq - S;

  // ---- Shared constants (idempotent across the partition's GQA nodes) --------
  const float s_row_q = p.scale_attr / 127.0f;
  const float s_row_k = 1.0f / 127.0f;
  if (!Contains(idx, kCosTabQ)) {
    std::vector<float> cos_rows, sin_rows;
    NEURON_RETURN_IF_ERROR(
        SliceRotaryCache(ctx, node.input(kCosCache), S, p.half, cos_rows));
    NEURON_RETURN_IF_ERROR(
        SliceRotaryCache(ctx, node.input(kSinCache), S, p.half, sin_rows));
    const std::vector<int8_t> cos_codes = TrigCodes(cos_rows);
    const std::vector<int8_t> sin_codes = TrigCodes(sin_rows);
    const Shape tab_shape{S, 1, p.half};
    for (const auto& [name, codes, scale] :
         {std::tuple{kCosTabQ, &cos_codes, s_row_q}, std::tuple{kSinTabQ, &sin_codes, s_row_q},
          std::tuple{kCosTabK, &cos_codes, s_row_k}, std::tuple{kSinTabK, &sin_codes, s_row_k}}) {
      NEURON_RETURN_IF_ERROR(mb.AddOperandFromPersistMemoryBuffer(
          name, codes->data(),
          OperandType(Type::TENSOR_QUANT8_ASYMM_SIGNED, tab_shape, scale, 0)));
    }
    // Causal mask [1,1,S,max_seq]: 0 for j <= t, code -128 beyond.
    std::vector<int8_t> mask(static_cast<size_t>(S) * p.max_seq);
    for (uint32_t t = 0; t < S; ++t)
      for (uint32_t j = 0; j < p.max_seq; ++j)
        mask[static_cast<size_t>(t) * p.max_seq + j] = (j <= t) ? 0 : -128;
    NEURON_RETURN_IF_ERROR(mb.AddOperandFromPersistMemoryBuffer(
        kMask4, mask.data(),
        OperandType(Type::TENSOR_QUANT8_ASYMM_SIGNED, Shape{1, 1, S, p.max_seq},
                    kMaskScale, 0)));
    // Requantization helper: multiply by exactly 1.0 (movement ops cannot change
    // scale; MUL can).
    const int8_t one = 1;
    NEURON_RETURN_IF_ERROR(mb.AddOperandFromPersistMemoryBuffer(
        kOne, &one, OperandType(Type::TENSOR_QUANT8_ASYMM_SIGNED, Shape{1, 1, 1}, 1.0f, 0)));
  }

  auto transpose_102 = [&](const std::string& src, const Shape& to, float scale,
                           int32_t zp, const std::string& out) -> NeuronStatus {
    const std::vector<int32_t> perm{1, 0, 2};
    const std::string perm_name = mb.GetUniqueName(out + "/perm");
    shaper.AddShape(out, to);
    NEURON_RETURN_IF_ERROR(mb.AddOperandFromPersistMemoryBuffer(
        perm_name, perm.data(), OperandType(Type::TENSOR_INT32, Shape{3})));
    return mb.AddOperation(NEURON_TRANSPOSE, {idx.at(src), idx.at(perm_name)}, {out},
                           {OperandType(Type::TENSOR_QUANT8_ASYMM_SIGNED, to, scale, zp)});
  };

  // ---- Q: [1,S,H] -> [S,nh,head] -> rotate -> [nh,S,head] -> [kv, g*S, head] --
  const std::string q3 = mb.GetUniqueName(base + "/q3");
  NEURON_RETURN_IF_ERROR(mb.AddReshape(node.input(kQuery), {S, p.num_heads, p.head_size}, q3));
  const std::string q_rot = mb.GetUniqueName(base + "/q_rot");
  NEURON_RETURN_IF_ERROR(EmitRotary(mb, base + "/q", q3, {S, p.num_heads, p.head_size},
                                    p.sq, p.zq, kCosTabQ, kSinTabQ, p.s_q, q_rot));
  const std::string q_t = mb.GetUniqueName(base + "/q_t");
  NEURON_RETURN_IF_ERROR(
      transpose_102(q_rot, {p.num_heads, S, p.head_size}, p.s_q, 0, q_t));
  const std::string q_m = mb.GetUniqueName(base + "/q_m");
  NEURON_RETURN_IF_ERROR(mb.AddReshape(q_t, {p.kv_heads, p.group * S, p.head_size}, q_m));

  // ---- K: rotate, lay out as cache rows, append the empty region -------------
  const std::string k3 = mb.GetUniqueName(base + "/k3");
  NEURON_RETURN_IF_ERROR(mb.AddReshape(node.input(kKey), {S, p.kv_heads, p.head_size}, k3));
  const std::string k_rot = mb.GetUniqueName(base + "/k_rot");
  NEURON_RETURN_IF_ERROR(EmitRotary(mb, base + "/k", k3, {S, p.kv_heads, p.head_size},
                                    p.sk, p.zk, kCosTabK, kSinTabK, p.kcs, k_rot));
  const std::string k_t = mb.GetUniqueName(base + "/k_t");
  NEURON_RETURN_IF_ERROR(transpose_102(k_rot, {p.kv_heads, S, p.head_size}, p.kcs, 0, k_t));

  const Shape cache3{p.kv_heads, p.max_seq, p.head_size};
  auto append_empty = [&](const std::string& rows, float cache_scale,
                          const std::string& present_out,
                          std::string& present3) -> NeuronStatus {
    const OperandType cache_type(Type::TENSOR_QUANT8_ASYMM_SIGNED, cache3, cache_scale, 0);
    if (rest == 0) {
      present3 = rows;
      return mb.AddReshape(present3, {1, p.kv_heads, p.max_seq, p.head_size}, present_out);
    }
    // Zero codes ARE real zero at zp 0; scale must match for a quantized concat.
    const std::string zeros = mb.GetUniqueName(base + "/cache_zeros");
    const std::vector<int8_t> zero_data(
        static_cast<size_t>(p.kv_heads) * rest * p.head_size, 0);
    NEURON_RETURN_IF_ERROR(mb.AddOperandFromPersistMemoryBuffer(
        zeros, zero_data.data(),
        OperandType(Type::TENSOR_QUANT8_ASYMM_SIGNED, Shape{p.kv_heads, rest, p.head_size},
                    cache_scale, 0)));
    present3 = mb.GetUniqueName(base + "/present3");
    shaper.AddShape(present3, cache3);
    std::vector<uint32_t> in{idx.at(rows), idx.at(zeros)};
    ADD_SCALAR_OPERAND(mb, in, static_cast<int32_t>(1));
    NEURON_RETURN_IF_ERROR(mb.AddOperation(NEURON_CONCATENATION, in, {present3}, {cache_type}));
    return mb.AddReshape(present3, {1, p.kv_heads, p.max_seq, p.head_size}, present_out);
  };
  std::string present_k3;
  NEURON_RETURN_IF_ERROR(append_empty(k_t, p.kcs, node.output(1), present_k3));

  // ---- V: lay out, requantize to the cache grid, append ----------------------
  const std::string v3 = mb.GetUniqueName(base + "/v3");
  NEURON_RETURN_IF_ERROR(mb.AddReshape(node.input(kValue), {S, p.kv_heads, p.head_size}, v3));
  const std::string v_t = mb.GetUniqueName(base + "/v_t");
  NEURON_RETURN_IF_ERROR(transpose_102(v3, {p.kv_heads, S, p.head_size}, p.sv, p.zv, v_t));
  const std::string v_rq = mb.GetUniqueName(base + "/v_rq");
  shaper.AddShape(v_rq, Shape{p.kv_heads, S, p.head_size});
  {
    std::vector<uint32_t> in{idx.at(v_t), idx.at(kOne)};
    ADD_SCALAR_OPERAND(mb, in, static_cast<int32_t>(NEURON_FUSED_NONE));
    NEURON_RETURN_IF_ERROR(mb.AddOperation(
        NEURON_MUL, in, {v_rq},
        {OperandType(Type::TENSOR_QUANT8_ASYMM_SIGNED, Shape{p.kv_heads, S, p.head_size},
                     p.vcs, 0)}));
  }
  std::string present_v3;
  NEURON_RETURN_IF_ERROR(append_empty(v_rq, p.vcs, node.output(2), present_v3));

  // ---- Attention core ---------------------------------------------------------
  const Shape logits3{p.kv_heads, p.group * S, p.max_seq};
  const OperandType logits3_type(Type::TENSOR_QUANT8_ASYMM_SIGNED, logits3,
                                 kDefaultLogitsScale, 0);
  const std::string logits = mb.GetUniqueName(base + "/logits");
  shaper.AddShape(logits, logits3);
  {
    std::vector<uint32_t> in{idx.at(q_m), idx.at(present_k3)};
    ADD_SCALAR_OPERAND(mb, in, false);
    ADD_SCALAR_OPERAND(mb, in, true);
    NEURON_RETURN_IF_ERROR(mb.AddOperation(NEURON_BATCH_MATMUL, in, {logits}, {logits3_type}));
  }
  const std::string l4 = mb.GetUniqueName(base + "/l4");
  NEURON_RETURN_IF_ERROR(mb.AddReshape(logits, {p.kv_heads, p.group, S, p.max_seq}, l4));
  const Shape logits4{p.kv_heads, p.group, S, p.max_seq};
  const std::string masked = mb.GetUniqueName(base + "/masked");
  shaper.AddShape(masked, logits4);
  {
    std::vector<uint32_t> in{idx.at(l4), idx.at(kMask4)};
    ADD_SCALAR_OPERAND(mb, in, static_cast<int32_t>(NEURON_FUSED_NONE));
    NEURON_RETURN_IF_ERROR(mb.AddOperation(
        NEURON_ADD, in, {masked},
        {OperandType(Type::TENSOR_QUANT8_ASYMM_SIGNED, logits4, kDefaultLogitsScale, 0)}));
  }
  const std::string probs = mb.GetUniqueName(base + "/probs");
  shaper.AddShape(probs, logits4);
  {
    std::vector<uint32_t> in{idx.at(masked)};
    ADD_SCALAR_OPERAND(mb, in, 1.0f);
    ADD_SCALAR_OPERAND(mb, in, static_cast<int32_t>(3));
    NEURON_RETURN_IF_ERROR(mb.AddOperation(
        NEURON_SOFTMAX, in, {probs},
        {OperandType(Type::TENSOR_QUANT8_ASYMM_SIGNED, logits4, kProbsScale, kProbsZeroPoint)}));
  }
  const std::string p3 = mb.GetUniqueName(base + "/p3");
  NEURON_RETURN_IF_ERROR(mb.AddReshape(probs, {p.kv_heads, p.group * S, p.max_seq}, p3));
  const Shape attn3{p.kv_heads, p.group * S, p.head_size};
  const std::string attn = mb.GetUniqueName(base + "/attn");
  shaper.AddShape(attn, attn3);
  {
    std::vector<uint32_t> in{idx.at(p3), idx.at(present_v3)};
    ADD_SCALAR_OPERAND(mb, in, false);
    ADD_SCALAR_OPERAND(mb, in, false);
    NEURON_RETURN_IF_ERROR(mb.AddOperation(
        NEURON_BATCH_MATMUL, in, {attn},
        {OperandType(Type::TENSOR_QUANT8_ASYMM_SIGNED, attn3, p.so, p.zo)}));
  }
  const std::string a4 = mb.GetUniqueName(base + "/a4");
  NEURON_RETURN_IF_ERROR(mb.AddReshape(attn, {p.kv_heads, p.group, S, p.head_size}, a4));
  // [kv, g, S, head] -> [S, kv, g, head]: BSNH with head-major (kv, g) = h.
  const std::string a_t = mb.GetUniqueName(base + "/a_t");
  {
    const std::vector<int32_t> perm{2, 0, 1, 3};
    const std::string perm_name = mb.GetUniqueName(a_t + "/perm");
    const Shape to{S, p.kv_heads, p.group, p.head_size};
    shaper.AddShape(a_t, to);
    NEURON_RETURN_IF_ERROR(mb.AddOperandFromPersistMemoryBuffer(
        perm_name, perm.data(), OperandType(Type::TENSOR_INT32, Shape{4})));
    NEURON_RETURN_IF_ERROR(mb.AddOperation(
        NEURON_TRANSPOSE, {idx.at(a4), idx.at(perm_name)}, {a_t},
        {OperandType(Type::TENSOR_QUANT8_ASYMM_SIGNED, to, p.so, p.zo)}));
  }
  return mb.AddReshape(a_t, {1, S, p.num_heads * p.head_size}, node.output(0));
}

NeuronStatus GroupQueryAttentionOpBuilder::AddToModelBuilderImpl(
    ModelBuilder& model_builder,
    const onnx::NodeProto& node,
    const NeuronGraphContext& ctx) const {
  GqaParams p;
  NEURON_RETURN_IF_ERROR(ResolveParams(node, ctx, p));
  LOGS_NEURON(VERBOSE) << "GQA '" << node.name() << "': S=" << p.seq_len
                       << " heads=" << p.num_heads << "/" << p.kv_heads
                       << " head_size=" << p.head_size << " max_seq=" << p.max_seq
                       << " kcs=" << p.kcs << " vcs=" << p.vcs;
  return p.seq_len == 1 ? BuildDecode(model_builder, node, ctx, p)
                        : BuildPrefill(model_builder, node, ctx, p);
}

// ---------------------------------------------------------------------------

void CreateGroupQueryAttentionOpBuilder(const std::string& op_type,
                                        OpBuilderRegistrations& op_registrations) {
  CreateSharedOpBuilderImpl<GroupQueryAttentionOpBuilder>(
      op_type, op_registrations, {"GroupQueryAttention"});
}

}  // namespace neuron
}  // namespace onnxruntime
