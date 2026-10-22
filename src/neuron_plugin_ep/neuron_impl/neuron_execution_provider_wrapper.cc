// neuron_execution_provider_wrapper.cc  (Plugin EP – refactored)
//
// Implements the NeuronWrapper_* C interface declared in
// ../src/neuron_execution_provider_wrapper.h.
//
// KEY ARCHITECTURAL CHANGE FROM THE ORIGINAL:
//   The original file cast OrtGraph* → const onnxruntime::GraphViewer* via
//   reinterpret_cast, then called NeuronExecutionProvider (an IExecutionProvider
//   subclass) which required dozens of ORT internal headers.
//
//   This refactored version:
//     1. Uses ort_graph_to_proto.h (header-only) to convert OrtGraph* →
//        onnx::GraphProto WITHOUT any ORT internal headers.
//     2. Builds a NeuronGraphContext from the proto (a lightweight lookup table).
//     3. Calls NeuronGraphCompiler (new class, no IExecutionProvider) which
//        determines supported ops and compiles the Neuron model.
//
//   There is NO reinterpret_cast to internal ORT types anywhere.

#define ORT_EP_UTILS_ORT_GRAPH_TO_PROTO_IMPL
#include "ort_graph_to_proto.h"   // header-only utility – copy from TRT EP

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "../neuron_execution_provider_wrapper.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <system_error>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <unistd.h>

#include <onnx/onnx_pb.h>

#include "neuron_graph_context.h"
#include "builders/helper.h"
#include "builders/neuron_model_builder.h"
#include "builders/onnx_ctx_model_helper.h"
#include "builders/op_builder.h"
#include "builders/op_builder_factory.h"
#include "neuron_lib/neuron_implementation.h"
#include "neuron_lib/neuron_wrapper_device.h"
#include "op_capability_abi/op_capability_loader.h"
#include "neuron_lib/neuron_wrapper_model.h"
#include "../neuron_provider_factory.h"

namespace {
// Bring onnxruntime into scope so that unqualified `neuron::wrapper::...`
// resolves to `onnxruntime::neuron::wrapper::...` throughout this file.
using namespace onnxruntime;

// ============================================================================
// NeuronWrapperState
// Per-session state.  No longer owns a NeuronExecutionProvider (which was an
// IExecutionProvider subclass); owns the raw Neuron SDK handle instead.
// ============================================================================
struct NeuronWrapperState {
  const OrtApi*            ort_api{nullptr};
  const OrtEpApi*          ep_api{nullptr};
  const OrtModelEditorApi* model_editor_api{nullptr};
  // Negotiated at load time; may be < ORT_API_VERSION.  See CreateEpFactories.
  uint32_t                 ort_api_version{0};

  // Provider option flags (copied from OrtKeyValuePairs at Create time).
  uint32_t neuron_flags{0};
  uint32_t min_group_size{1};
  // NEURON_FLAG_MAX_PARTITION_BYTES: split claimed groups whose constant
  // initializers exceed this many bytes (0 = unlimited).  MDLA codegen places
  // all of a network's constants in one contiguous static buffer, and the
  // Genio 720 driver cannot map one above ~1 GiB (NEURON_UNMAPPABLE; measured:
  // 960 MiB maps, 1024 MiB does not).  Default: 768 MiB, so a model that fits
  // one network is untouched and a larger one compiles without any flag; a cut
  // can only land on a QDQ-unit boundary and may overshoot the budget by one
  // unit (measured: 896 MiB produced a 904 MiB chunk on llama), hence the margin.
  uint64_t max_partition_bytes{768ull << 20};
  // NEURON_FLAG_PARTITION_COUNT: instead of a byte budget, cut each oversized
  // group into this many chunks balanced by constant bytes (0 = use the budget).
  uint32_t partition_count{0};
  // NEURON_FLAG_COMPUTE_LIGHT_MIN_NODES: keep a partition without any
  // conv/matmul once it has at least this many nodes (e.g. a detection head's
  // anchor decode); smaller compute-light fragments run faster on CPU than as
  // an NPU dispatch.  0 drops every compute-light partition.
  uint32_t compute_light_min_nodes{16};
  std::string optimization_string;
  std::unordered_set<std::string> force_skip_ops;
  // NEURON_FLAG_SKIP_NODE_PREFIXES: comma-separated node-name prefixes left on
  // CPU.  Precision escape hatch for individual nodes the NPU cannot run
  // faithfully, e.g. an attention matmul whose logits exceed fp16 range.
  std::vector<std::string> skip_node_prefixes;

  // EPContext (DLA cache) generation.  Mirrors ep.context_* session options.
  bool        context_cache_enabled{false};
  bool        context_embed_mode{false};  // ORT's documented default is 0
  std::string user_context_cache_path;

  // NEURON_FLAG_SKIP_CAPABILITY_PROBE; trades CPU fallback for partitioning speed
  bool skip_capability_probe{false};

  // Whether this Neuron stack can build an MDLA QuantizeLayer from a float
  // input; a device property probed once on first use (-1 = not yet known).
  // mutable: cached lazily during GetCapability, which sees const state.
  mutable int mdla_float_quantize{-1};

  // Neuron SDK handle.
  const NeuronApi* neuron_handle{nullptr};
  NeuronRuntimeVersion neuron_version{1, 0, 0};
  neuron::wrapper::DeviceVector neuron_target_devices;
  neuron::wrapper::TargetDeviceOption target_device_option{
      neuron::wrapper::TargetDeviceOption::ALL_DEVICES};

  // Map from fused-node name → compiled Neuron model (filled by Compile).
  std::unordered_map<std::string, std::unique_ptr<neuron::wrapper::Model>> neuron_models;
};

// ============================================================================
// NeuronComputeState
// Per-thread compute state created once by CreateStateImpl and reused across
// every inference call.  Caches all static metadata so ComputeImpl does zero
// heap allocations and zero ORT type-query calls per inference.
// ============================================================================
struct NeuronComputeState {
  std::unique_ptr<neuron::wrapper::Execution> execution;

  struct InputInfo {
    size_t ort_idx;    // which ORT input slot maps to this model input
    size_t byte_size;  // pre-computed, constant for static shapes
    int synth_kind{-1};  // wrapper::Model::SynthInput::Kind, or -1 for a real input
  };
  struct OutputInfo {
    size_t ort_idx;
    size_t byte_size;
    std::vector<int64_t> shape_i64;  // pre-built for ctx.GetOutput(); empty for scalars
  };
  std::vector<InputInfo>  input_infos;
  std::vector<OutputInfo> output_infos;

  std::vector<std::vector<uint8_t>> synth_buffers;
  const neuron::wrapper::Model* model{nullptr};
  size_t seqlens_ort_idx{SIZE_MAX};
  int32_t last_synth_pos{-1};  // skip refill when the position has not changed
};

// Fills one synth buffer for position `pos`.
// rotary rows round(cos*127) int8, keep/onehot at scale 1 zp 0, mask codes
// {0, -128}.
static void FillSynthInput(const neuron::wrapper::Model& model,
                           neuron::wrapper::Model::SynthInput::Kind kind,
                           int32_t pos, std::vector<uint8_t>& buf) {
  using Kind = neuron::wrapper::Model::SynthInput::Kind;
  int8_t* p = reinterpret_cast<int8_t*>(buf.data());
  const size_t n = buf.size();
  switch (kind) {
    case Kind::kCosRow:
    case Kind::kSinRow: {
      const auto& table = (kind == Kind::kCosRow) ? model.GetRotaryCos()
                                                  : model.GetRotarySin();
      const uint32_t half = model.GetRotaryHalf();
      const size_t row = static_cast<size_t>(pos) * half;
      for (uint32_t i = 0; i < half && i < n; ++i)
        p[i] = static_cast<int8_t>(std::lround(table[row + i] * 127.0f));
      break;
    }
    case Kind::kKeep:
      std::memset(p, 1, n);
      if (static_cast<size_t>(pos) < n) p[pos] = 0;
      break;
    case Kind::kOneHot: {
      std::memset(p, 0, n);
      const size_t max_seq = model.GetSynthMaxSeq();
      if (max_seq == 0) break;
      for (size_t off = pos; off < n; off += max_seq) p[off] = 1;
      break;
    }
    case Kind::kMask:
      std::memset(p, 0, std::min(n, static_cast<size_t>(pos) + 1));
      if (static_cast<size_t>(pos) + 1 < n)
        std::memset(p + pos + 1, 0x80, n - pos - 1);
      break;
  }
}

// ============================================================================
// NeuronNodeComputeInfo
// Wraps the per-fused-node execution state for the OrtNodeComputeInfo vtable.
// ============================================================================
struct NeuronNodeComputeInfo : OrtNodeComputeInfo {
  NeuronWrapperState* state{nullptr};
  std::string fused_node_name;
  const OrtApi* ort_api{nullptr};

  NeuronNodeComputeInfo(NeuronWrapperState* s, std::string name, const OrtApi* api)
      : state(s), fused_node_name(std::move(name)), ort_api(api) {
    ort_version_supported = s->ort_api_version;
    CreateState  = CreateStateImpl;
    Compute      = ComputeImpl;
    ReleaseState = ReleaseStateImpl;
  }

  // CreateState is called once per inference thread / execution context.
  // We build a NeuronComputeState here so ComputeImpl pays zero allocation cost.
  static OrtStatus* ORT_API_CALL CreateStateImpl(
      OrtNodeComputeInfo* this_ptr,
      OrtNodeComputeContext* /*ort_ctx*/,
      void** out_state) {
    auto* self = static_cast<NeuronNodeComputeInfo*>(this_ptr);
    auto it = self->state->neuron_models.find(self->fused_node_name);
    if (it == self->state->neuron_models.end()) {
      return self->ort_api->CreateStatus(ORT_EP_FAIL,
          "NeuronEP: compiled model not found for node");
    }
    auto* model = it->second.get();

    auto* cs = new (std::nothrow) NeuronComputeState{};
    if (!cs) return self->ort_api->CreateStatus(ORT_EP_FAIL, "NeuronEP: OOM in CreateState");

    // Create a dedicated NeuronExecution for this state.
    // Each state owns its execution so no mutex is needed at inference time.
    auto ns = model->PrepareForExecution(cs->execution);
    if (!ns.IsOK()) {
      delete cs;
      return self->ort_api->CreateStatus(ORT_EP_FAIL, ns.ErrorMessage().c_str());
    }

    // Pre-build input metadata.
    // Synth inputs (host-computed GQA position) have no ORT slot:
    // they get a state-owned buffer instead.
    const auto& model_inputs = model->GetInputs();
    cs->input_infos.resize(model_inputs.size());
    cs->synth_buffers.resize(model_inputs.size());
    cs->model = model;
    for (size_t i = 0; i < model_inputs.size(); ++i) {
      const auto& name = model_inputs[i];
      auto& info = cs->input_infos[i];
      info.byte_size = model->GetInputType(name).GetOperandBlobByteSize();
      bool synth = false;
      for (const auto& si : model->GetSynthInputs()) {
        if (si.name == name) {
          info.synth_kind = static_cast<int>(si.kind);
          info.ort_idx = SIZE_MAX;
          cs->synth_buffers[i].resize(info.byte_size);
          synth = true;
          break;
        }
      }
      if (!synth) info.ort_idx = model->GetMappedInputIdx(name);
    }
    if (!model->GetSynthInputs().empty()) {
      const auto& seqlens = model->GetSeqlensInputName();
      if (seqlens.empty()) {
        delete cs;
        return self->ort_api->CreateStatus(ORT_EP_FAIL,
            "NeuronEP: synth inputs present but no seqlens input recorded");
      }
      cs->seqlens_ort_idx = model->GetMappedInputIdx(seqlens);
    }

    // Pre-build output metadata.
    const auto& model_outputs = model->GetOutputs();
    cs->output_infos.resize(model_outputs.size());
    for (size_t i = 0; i < model_outputs.size(); ++i) {
      const auto& name = model_outputs[i];
      auto out_type = model->GetOutputType(name, *cs->execution);
      if (out_type.GetOperandBlobByteSize() == 0) {
        delete cs;
        return self->ort_api->CreateStatus(ORT_EP_FAIL,
            "NeuronEP: dynamic output shape not supported in this build");
      }
      cs->output_infos[i].ort_idx   = model->GetMappedOutputIdx(name);
      cs->output_infos[i].byte_size = out_type.GetOperandBlobByteSize();
      // Build the int64 shape vector once; leave empty for scalar outputs.
      if (!model->IsScalarOutput(name)) {
        const auto& dims = out_type.dimensions;
        auto& shape_i64 = cs->output_infos[i].shape_i64;
        shape_i64.assign(dims.begin(), dims.end());
        // Re-pad leading unit dims the operand had squeezed away,
        // so ORT sees the tensor at its ONNX rank.
        const size_t ort_rank = model->GetOrtOutputRank(name);
        while (shape_i64.size() < ort_rank)
          shape_i64.insert(shape_i64.begin(), 1);
      }
    }

    *out_state = cs;
    return nullptr;
  }

  static OrtStatus* ORT_API_CALL ComputeImpl(
      OrtNodeComputeInfo* this_ptr,
      void* state_ptr,
      OrtKernelContext* kernel_ctx) {
    auto* self = static_cast<NeuronNodeComputeInfo*>(this_ptr);
    auto* cs   = static_cast<NeuronComputeState*>(state_ptr);
    Ort::KernelContext ctx(kernel_ctx);

    // Derive the position once, refill on change.
    if (cs->seqlens_ort_idx != SIZE_MAX) {
      const int32_t pos = *static_cast<const int32_t*>(
          ctx.GetInput(cs->seqlens_ort_idx).GetTensorRawData());
      if (pos != cs->last_synth_pos) {
        for (size_t i = 0; i < cs->input_infos.size(); ++i) {
          if (cs->input_infos[i].synth_kind < 0) continue;
          FillSynthInput(*cs->model,
                         static_cast<neuron::wrapper::Model::SynthInput::Kind>(
                             cs->input_infos[i].synth_kind),
                         pos, cs->synth_buffers[i]);
        }
        cs->last_synth_pos = pos;
      }
    }

    // Set inputs — no allocations, no type queries.
    for (size_t i = 0; i < cs->input_infos.size(); ++i) {
      const auto& info = cs->input_infos[i];
      const void* buf = (info.synth_kind >= 0)
                            ? static_cast<const void*>(cs->synth_buffers[i].data())
                            : ctx.GetInput(info.ort_idx).GetTensorRawData();
      auto ns = cs->execution->SetInputFast(static_cast<int32_t>(i), buf, info.byte_size);
      if (!ns.IsOK())
        return self->ort_api->CreateStatus(ORT_EP_FAIL, ns.ErrorMessage().c_str());
    }

    // Set outputs — pre-built shapes, no copies.
    for (size_t i = 0; i < cs->output_infos.size(); ++i) {
      const auto& info = cs->output_infos[i];
      auto ort_out = ctx.GetOutput(info.ort_idx,
                                   info.shape_i64.data(),
                                   info.shape_i64.size());
      // GetTensorMutableData<uint8_t>() is a type-agnostic raw-pointer accessor;
      // the template parameter is irrelevant — the underlying C API returns void*.
      void* buf = ort_out.GetTensorMutableData<uint8_t>();
      auto ns = cs->execution->SetOutputFast(
          static_cast<int32_t>(i), buf, info.byte_size);
      if (!ns.IsOK())
        return self->ort_api->CreateStatus(ORT_EP_FAIL, ns.ErrorMessage().c_str());
    }

    {
      auto ns = cs->execution->Predict();
      if (!ns.IsOK())
        return self->ort_api->CreateStatus(ORT_EP_FAIL, ns.ErrorMessage().c_str());
    }
    return nullptr;
  }

  static void ORT_API_CALL ReleaseStateImpl(OrtNodeComputeInfo* /*this_ptr*/,
                                             void* state_ptr) {
    delete static_cast<NeuronComputeState*>(state_ptr);
  }
};

// ============================================================================
// Frees each initializer's bytes but keeps the TensorProto (name, dims) valid,
// since NeuronGraphContext holds pointers to them.  clear() would keep the
// string's capacity, so swap with an empty one instead.
// ============================================================================
static void ReleaseInitializerData(onnx::GraphProto& graph) {
  for (auto& tensor : *graph.mutable_initializer())
    std::string().swap(*tensor.mutable_raw_data());
}

// ============================================================================
// Helper: read a string key from OrtKeyValuePairs or return empty string.
// ============================================================================
static std::string GetKV(const OrtApi* ort_api,
                          const OrtKeyValuePairs* kv,
                          const char* key) {
  if (!kv) return {};
  const char* v = ort_api->GetKeyValue(kv, key);
  return v ? std::string{v} : std::string{};
}

// ============================================================================
// Bitset helpers for the convex partitioner (below).
// A "word bitset" is a std::vector<uint64_t>; bit n lives in word n/64, bit n%64.
// ============================================================================
namespace partition_detail {

inline void BitSet(std::vector<uint64_t>& bs, int n) {
  bs[static_cast<size_t>(n) >> 6] |= (1ull << (n & 63));
}
inline void BitOrInto(std::vector<uint64_t>& dst, const std::vector<uint64_t>& src) {
  for (size_t w = 0; w < dst.size(); ++w) dst[w] |= src[w];
}
// True if (a & b) has any bit set.
inline bool BitAndAny(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
  for (size_t w = 0; w < a.size(); ++w)
    if (a[w] & b[w]) return true;
  return false;
}

// Minimal union-find over node indices.
struct DisjointSet {
  std::vector<int> parent;
  explicit DisjointSet(int n) : parent(static_cast<size_t>(n)) {
    for (int i = 0; i < n; ++i) parent[static_cast<size_t>(i)] = i;
  }
  int Find(int x) {
    while (parent[static_cast<size_t>(x)] != x) {
      parent[static_cast<size_t>(x)] = parent[static_cast<size_t>(parent[static_cast<size_t>(x)])];
      x = parent[static_cast<size_t>(x)];
    }
    return x;
  }
  void Merge(int a, int b) {
    a = Find(a); b = Find(b);
    if (a != b) parent[static_cast<size_t>(b)] = a;
  }
};

}  // namespace partition_detail

// ============================================================================
// Predictive hardware-representability check (cheap; no compilation).
//
// Some quantized ops pass the software capability check but cannot be encoded
// for the target hardware because their quantization parameters are degenerate.
// The clearest case: a Conv/Gemm/MatMul whose INT32 bias has been saturated to
// INT32_MIN/MAX during quantization (happens when the input activation scale is
// vanishingly small, e.g. mobilenet_v3's SE block with a 4e-9 Relu scale).  The
// MDLA compiler rejects this as "invalid constant data".  We detect it from the
// bias initializer directly and treat the op as unsupported, so the convex
// partitioner routes it (and only it) to CPU.
// ============================================================================
// True if a node in `group` has an input that ComputeImpl reads on the host
// from the fused node's inputs and that input is a constant initializer.  With
// drop_constant_initializers ORT removes constant initializers from the fused
// node's inputs, so the by-name lookup in CreateState would find nothing.
static bool GroupReadsConstantAtRuntime(
    const onnx::GraphProto& graph,
    const onnxruntime::neuron::NeuronGraphContext& ctx,
    const std::vector<int>& group) {
  constexpr int kGqaSeqlensK = 5;  // GroupQueryAttention: read in ComputeImpl
  for (int idx : group) {
    const auto& node = graph.node(idx);
    if (node.op_type() == "GroupQueryAttention" &&
        node.input_size() > kGqaSeqlensK &&
        ctx.IsInitializer(node.input(kGqaSeqlensK)))
      return true;
  }
  return false;
}

static bool HasSaturatedInt32Bias(const onnx::NodeProto& node,
                                   const onnxruntime::neuron::NeuronGraphContext& ctx) {
  const std::string& op = node.op_type();
  if (op != "Conv" && op != "Gemm" && op != "MatMul" &&
      op != "QLinearConv" && op != "ConvTranspose")
    return false;
  // Bias is input(2) for Conv/ConvTranspose/Gemm; QLinearConv bias is input(8).
  int bias_slot = -1;
  if (op == "QLinearConv") bias_slot = (node.input_size() > 8) ? 8 : -1;
  else                     bias_slot = (node.input_size() > 2) ? 2 : -1;
  if (bias_slot < 0 || node.input(bias_slot).empty()) return false;

  // Resolve the bias initializer, following a DequantizeLinear if present.
  std::string bias_name = node.input(bias_slot);
  if (const auto* prod = ctx.GetProducer(bias_name)) {
    if (prod->op_type() == "DequantizeLinear" && prod->input_size() > 0)
      bias_name = prod->input(0);
  }
  const auto* init = ctx.GetInitializer(bias_name);
  if (!init || init->data_type() != onnx::TensorProto::INT32) return false;

  std::vector<int32_t> vals;
  if (!onnxruntime::neuron::NeuronGraphContext::GetInt32Data(*init, vals).IsOK())
    return false;
  for (int32_t v : vals)
    if (v == INT32_MIN || v == INT32_MAX) return true;
  return false;
}

// ============================================================================
// SplitGroupsByConstantBytes
//
// MDLA codegen emits one contiguous static buffer holding every constant of a
// network, and the driver refuses to map it above a platform budget (~1 GiB on
// the Genio 720: NEURON_UNMAPPABLE at NeuronCompilation_finish).  Split any
// group whose initializers exceed `budget` into consecutive slices of its
// members.  Members are in topological (proto) order, and a contiguous slice
// of a convex group is convex: a path a->x->b has idx(a) < idx(x) < idx(b), so
// x already lay between two members of the parent.  A cut never lands inside a
// QDQ unit (DQ/target/Q), which would leave a DQ without its target.
//
// count == 0: greedy chunks of at most `budget` bytes.  count > 0: an oversized
// group is cut at the legal boundaries closest to k*total/count, i.e. `count`
// chunks balanced by constant bytes; `budget` is then only checked and warned.
// ============================================================================
static std::vector<std::vector<int>> SplitGroupsByConstantBytes(
    const onnx::GraphProto& graph_proto,
    const onnxruntime::neuron::NeuronGraphContext& ctx,
    std::vector<std::vector<int>> groups,
    uint64_t budget, uint32_t count) {
  const int N = graph_proto.node_size();

  // unit_end[i]: highest node index of any QDQ unit containing node i.  A cut
  // after node j is legal only when every earlier member's unit ends <= j.
  std::vector<int> unit_end(static_cast<size_t>(N));
  for (int i = 0; i < N; ++i) unit_end[static_cast<size_t>(i)] = i;
  for (const auto& kv : ctx.qdq_groups) {
    const auto& g = kv.second;
    int end = g.target_node_index;
    for (int q : g.q_node_indices) end = std::max(end, q);
    auto bump = [&](int i) {
      if (i >= 0 && i < N)
        unit_end[static_cast<size_t>(i)] = std::max(unit_end[static_cast<size_t>(i)], end);
    };
    bump(g.target_node_index);
    for (int d : g.dq_node_indices) bump(d);
    for (int q : g.q_node_indices) bump(q);
  }

  // OrtGraphToProto inlines every initializer as raw_data (no handler is
  // passed), so this is exact even for weights that were external on disk.
  auto init_bytes = [&](const std::string& name) -> uint64_t {
    const auto* t = ctx.GetInitializer(name);
    if (!t) return 0;
    if (!t->raw_data().empty()) return t->raw_data().size();
    return 4ull * (t->float_data_size() + t->int32_data_size()) +
           8ull * (t->int64_data_size() + t->double_data_size() + t->uint64_data_size());
  };
  // Bytes node `idx` adds to a chunk that has already seen the tensors in `seen`.
  auto new_bytes = [&](int idx, const std::unordered_set<std::string>& seen) -> uint64_t {
    uint64_t b = 0;
    for (const auto& in : graph_proto.node(idx).input())
      if (!in.empty() && !seen.count(in)) b += init_bytes(in);
    return b;
  };
  auto mark_seen = [&](int idx, std::unordered_set<std::string>& seen) {
    for (const auto& in : graph_proto.node(idx).input())
      if (!in.empty()) seen.insert(in);
  };

  auto group_bytes = [&](const std::vector<int>& group) {
    std::unordered_set<std::string> seen;
    uint64_t total = 0;
    for (int idx : group) { total += new_bytes(idx, seen); mark_seen(idx, seen); }
    return total;
  };

  uint64_t largest = 0;
  if (count > 0)
    for (const auto& group : groups) largest = std::max(largest, group_bytes(group));

  std::vector<std::vector<int>> out;
  for (auto& group : groups) {
    std::unordered_set<std::string> seen;
    const uint64_t total = group_bytes(group);
    if (total <= budget && !(count > 0 && total == largest && total > 0)) {
      out.push_back(std::move(group));
      continue;
    }

    std::vector<int> chunk;
    uint64_t bytes = 0, done = 0;
    uint32_t emitted = 0;
    int pending = -1;
    seen.clear();
    auto cut_here = [&](uint64_t next) {
      if (count == 0) return bytes + next > budget;
      if (emitted + 1 >= count) return false;
      const uint64_t target = total * (emitted + 1) / count;
      return done + bytes >= target || done + bytes + next > target + next / 2;
    };
    auto emit = [&](const char* before) {
      LOGS_NEURON(INFO) << "Partition chunk " << (emitted + 1) << ": " << chunk.size() << " nodes, "
                        << (bytes >> 20) << " MiB of constants"
                        << (before ? std::string(", cut before node [") + before + "]" : "");
      if (bytes > budget) {
        LOGS_NEURON(WARNING) << "Partition holds " << (bytes >> 20) << " MiB of constants, above the "
                             << (budget >> 20) << " MiB per-network budget; it may fail to map";
      }
      out.push_back(std::move(chunk));
      chunk.clear(); seen.clear();
      done += bytes; bytes = 0; ++emitted;
    };
    for (int idx : group) {
      uint64_t next = new_bytes(idx, seen);
      if (!chunk.empty() && chunk.back() >= pending && cut_here(next)) {
        emit(graph_proto.node(idx).name().c_str());
        next = new_bytes(idx, seen);  // the new chunk has seen nothing yet
      }
      bytes += next;
      mark_seen(idx, seen);
      chunk.push_back(idx);
      pending = std::max(pending, unit_end[static_cast<size_t>(idx)]);
    }
    if (!chunk.empty()) emit(nullptr);
  }
  return out;
}

// ============================================================================
// Helper: determine which nodes in the graph are supported and partition them.
// Returns a list of node index groups (each group is one fused subgraph).
//
// Two phases:
//   Phase A — per-node support determination.  Each node is judged in
//     isolation (op-builder support, standalone-DQ/Q rejection, live-INT64
//     rejection).  Absorbed DQ/Q nodes inherit their QDQ target's verdict.
//     No greedy "flush on interruption" logic — support is independent of
//     grouping.
//
//   Phase B — maximal convex partitioning.  Supported nodes are grouped into
//     the fewest partitions such that each partition is CONVEX: no unsupported
//     node lies on any path between two nodes of the same partition.  This is
//     the invariant ORT's utils::CreateSupportedPartitions enforces and that
//     ep_plugin_provider_interfaces.cc requires of EpGraphSupportInfo_
//     AddNodesToFuse (a group that is non-convex is rejected wholesale).
//
//     Convexity is tested with node-reachability bitsets: the union of two
//     partitions is convex iff no unsupported node u is simultaneously
//     reachable-from a member (u in outReach) and can-reach a member
//     (u in inReach).  Merging maximally along supported producer→consumer
//     edges yields the minimum number of partitions, which minimises the
//     number of separate Neuron model compile/execute round-trips.
//
// QDQ handling:
//   NeuronGraphContext::DetectQDQGroups() (called in the ctor) identifies
//   DQ→Op→Q patterns and stores them in ctx.qdq_groups /
//   ctx.qdq_absorbed_node_indices.  Each group's DQ + target + Q nodes are
//   pre-unioned into one atomic unit so a QDQ group is never split across
//   partitions.  Unlike the previous implementation, QDQ groups ARE merged
//   with neighbouring supported ops whenever doing so stays convex — a chain
//   of quantized layers becomes ONE Neuron model instead of N.
// ============================================================================
//
// graph_proto is non-const: CapabilityGraphHandle strips tensor payloads in
// place and restores them before returning.
static std::vector<std::vector<int>> FindSupportedGroups(
    onnx::GraphProto& graph_proto,
    const NeuronWrapperState* state) {

  using namespace onnxruntime::neuron;
  using namespace partition_detail;

  // NeuronGraphContext constructor runs DetectQDQGroups internally.
  NeuronGraphContext ctx(graph_proto);

  const int N = graph_proto.node_size();
  std::vector<std::vector<int>> groups;
  if (N == 0) return groups;

  const auto& op_builders = GetOpBuilders();

  OpSupportCheckParams check_params;
  check_params.version  = state->neuron_version;
  check_params.devices  = state->neuron_target_devices;
  check_params.use_nchw = (state->neuron_flags & NEURON_FLAG_USE_NCHW) != 0;
  check_params.use_fp16 = (state->neuron_flags & NEURON_FLAG_USE_FP16) != 0;

  // One capability-library copy of the graph per partitioning pass, not per
  // node. The handle must outlive every IsOpSupported call below. The payload
  // elision only mutates fields inside existing TensorProtos, so ctx's
  // initializer pointers stay valid.
  op_capability::CapabilityGraphHandle cap_graph(graph_proto);
  if (!cap_graph.valid()) {
    LOGS_NEURON(ERROR) << "Could not hand the graph to the op capability library; "
                       << "no nodes will be assigned to Neuron.";
    return groups;
  }
  check_params.cap_graph = cap_graph.get();

  // ──────────────────────────────────────────────────────────────────────
  // Phase A: per-node support determination (independent of grouping).
  // ──────────────────────────────────────────────────────────────────────
  std::vector<char> supported(static_cast<size_t>(N), 0);

  for (int i = 0; i < N; ++i) {
    const auto& node = graph_proto.node(i);
    const std::string& op = node.op_type();

    // Absorbed DQ/Q nodes inherit their target's verdict — handled in Phase A2.
    if (ctx.qdq_absorbed_node_indices.count(i)) continue;

    // Force-skipped ops.
    if (state->force_skip_ops.count(op)) continue;

    // Force-skipped node-name prefixes.
    {
      bool skip = false;
      for (const auto& p : state->skip_node_prefixes)
        if (node.name().compare(0, p.size(), p) == 0) { skip = true; break; }
      if (skip) {
        LOGS_NEURON(VERBOSE) << "Node [" << node.name()
                             << "] matches NEURON_FLAG_SKIP_NODE_PREFIXES — leaving on CPU";
        continue;
      }
    }

    // Op-builder support check.
    const auto it = op_builders.find(op);
    if (it == op_builders.end() ||
        !it->second->IsOpSupported(node, ctx, check_params)) {
      continue;
    }

    // MDLA rejects any operand dimension above 65535.
    {
      constexpr uint32_t kMaxNeuronDim = 65535;
      auto oversized = [&](const std::string& name) {
        for (const uint32_t d : ctx.GetShape(name))
          if (d > kMaxNeuronDim) return true;
        return false;
      };
      bool bad_dim = false;
      for (const auto& inp : node.input())
        if (!inp.empty() && !ctx.IsInitializer(inp) && oversized(inp)) { bad_dim = true; break; }
      for (const auto& out : node.output())
        if (!bad_dim && !out.empty() && oversized(out)) bad_dim = true;
      if (bad_dim) {
        LOGS_NEURON(INFO) << "Node [" << node.name() << "] op_type [" << op
                          << "]: an activation dimension exceeds MDLA's "
                          << kMaxNeuronDim << " limit; running on CPU.";
        continue;
      }
    }

    // Find this node's QDQ group (for the own-DQ INT64 allowance).
    const QDQNodeGroup* qdq = nullptr;
    for (const auto& out : node.output()) {
      if (!out.empty()) { qdq = ctx.GetQDQGroup(out); if (qdq) break; }
    }

    if (qdq == nullptr) {
      bool dq_wrapped_const = false;
      for (const auto& inp : node.input()) {
        if (inp.empty() || ctx.IsInitializer(inp)) continue;
        const auto* p = ctx.GetProducer(inp);
        if (p && p->op_type() == "DequantizeLinear" && p->input_size() > 0 &&
            ctx.IsInitializer(p->input(0))) {
          dq_wrapped_const = true;
          break;
        }
      }
      if (dq_wrapped_const) {
        LOGS_NEURON(INFO) << "Node [" << node.name() << "] op [" << op
                          << "] consumes a standalone dequantized constant "
                             "(weight-only quantization) — leaving on CPU";
        continue;
      }
    }

    // Live INT64 input check.  Neuron cannot represent INT64 operands, and an
    // INT64 tensor crossing a partition boundary is unrepresentable.  We reject
    // any node with a live (non-initializer) INT64 input other than one coming
    // from this node's own absorbed DQ.  This is a hard invariant, so it is
    // enforced per-node here rather than relying on grouping.
    bool has_live_int64 = false;
    for (const auto& inp : node.input()) {
      if (inp.empty() || ctx.IsInitializer(inp)) continue;
      bool from_own_dq = false;
      if (qdq) {
        for (int dq_idx : qdq->dq_node_indices) {
          for (const auto& dq_out : graph_proto.node(dq_idx).output())
            if (dq_out == inp) { from_own_dq = true; break; }
          if (from_own_dq) break;
        }
      }
      if (from_own_dq) continue;
      if (ctx.GetElementType(inp) == onnx::TensorProto::INT64) {
        LOGS_NEURON(VERBOSE) << "Node [" << node.name() << "] op_type [" << op
                             << "]: live INT64 input '" << inp
                             << "' — falling back to CPU";
        has_live_int64 = true;
        break;
      }
    }
    if (has_live_int64) continue;

    // Degenerate quantization the hardware cannot encode (e.g. saturated INT32
    // bias) → leave on CPU rather than fail the whole session at compile time.
    if (HasSaturatedInt32Bias(node, ctx)) {
      LOGS_NEURON(WARNING) << "Node [" << node.name() << "] op [" << op
                           << "] has a saturated INT32 bias (degenerate quant "
                              "scale) — leaving on CPU";
      continue;
    }

    supported[static_cast<size_t>(i)] = 1;
  }

  // Phase A2: absorbed DQ/Q inherit their QDQ target's support verdict.
  for (const auto& kv : ctx.qdq_groups) {
    const QDQNodeGroup& g = kv.second;
    const int t = g.target_node_index;
    if (t < 0 || t >= N || !supported[static_cast<size_t>(t)]) continue;
    for (int d : g.dq_node_indices)
      if (d >= 0 && d < N) supported[static_cast<size_t>(d)] = 1;
    for (int q : g.q_node_indices)
      if (q >= 0 && q < N) supported[static_cast<size_t>(q)] = 1;
  }

  // ──────────────────────────────────────────────────────────────────────
  // Phase B: maximal convex partitioning of the supported node set.
  // ──────────────────────────────────────────────────────────────────────

  // Producer map: output name → node index.
  std::unordered_map<std::string, int> out_to_idx;
  out_to_idx.reserve(static_cast<size_t>(N) * 2);
  for (int i = 0; i < N; ++i)
    for (const auto& o : graph_proto.node(i).output())
      if (!o.empty()) out_to_idx.emplace(o, i);

  // Adjacency (data dependencies) over ALL nodes — convexity depends on
  // unsupported nodes lying between supported ones.
  std::vector<std::vector<int>> succ(static_cast<size_t>(N));
  std::vector<std::pair<int, int>> edges;  // (producer, consumer)
  for (int i = 0; i < N; ++i) {
    for (const auto& in : graph_proto.node(i).input()) {
      if (in.empty()) continue;
      auto it = out_to_idx.find(in);
      if (it == out_to_idx.end()) continue;
      const int p = it->second;
      if (p == i) continue;
      succ[static_cast<size_t>(p)].push_back(i);
      edges.emplace_back(p, i);
    }
  }

  const int W = (N + 63) / 64;

  // desc[i] = set of nodes reachable FROM i (strict descendants).
  // anc[i]  = set of nodes that can reach i (strict ancestors).
  // Computed via a single pass each over the topological order.
  std::vector<std::vector<uint64_t>> desc(
      static_cast<size_t>(N), std::vector<uint64_t>(static_cast<size_t>(W), 0));
  std::vector<std::vector<uint64_t>> anc(
      static_cast<size_t>(N), std::vector<uint64_t>(static_cast<size_t>(W), 0));

  // Descendants: reverse topological order.
  for (auto ti = static_cast<int>(ctx.topo_order.size()) - 1; ti >= 0; --ti) {
    const int i = ctx.topo_order[static_cast<size_t>(ti)];
    if (i < 0 || i >= N) continue;
    auto& di = desc[static_cast<size_t>(i)];
    for (int s : succ[static_cast<size_t>(i)]) {
      BitSet(di, s);
      BitOrInto(di, desc[static_cast<size_t>(s)]);
    }
  }
  // Ancestors: forward topological order (predecessor = any node whose desc
  // contains i is expensive; instead accumulate from succ edges).
  for (int ti = 0; ti < static_cast<int>(ctx.topo_order.size()); ++ti) {
    const int i = ctx.topo_order[static_cast<size_t>(ti)];
    if (i < 0 || i >= N) continue;
    // For each successor s of i, i (and i's ancestors) are ancestors of s.
    auto& ai = anc[static_cast<size_t>(i)];
    for (int s : succ[static_cast<size_t>(i)]) {
      auto& as = anc[static_cast<size_t>(s)];
      BitSet(as, i);
      BitOrInto(as, ai);
    }
  }

  // Unsupported-node bitset.
  std::vector<uint64_t> unsupported(static_cast<size_t>(W), 0);
  for (int i = 0; i < N; ++i)
    if (!supported[static_cast<size_t>(i)]) BitSet(unsupported, i);

  // Union-find seeded with QDQ atomic units (DQ + target + Q kept together).
  DisjointSet dsu(N);
  for (const auto& kv : ctx.qdq_groups) {
    const QDQNodeGroup& g = kv.second;
    const int t = g.target_node_index;
    if (t < 0 || t >= N || !supported[static_cast<size_t>(t)]) continue;
    for (int d : g.dq_node_indices)
      if (d >= 0 && d < N && supported[static_cast<size_t>(d)]) dsu.Merge(t, d);
    for (int q : g.q_node_indices)
      if (q >= 0 && q < N && supported[static_cast<size_t>(q)]) dsu.Merge(t, q);
  }

  // Per-root aggregate reachability, indexed by node (valid only at roots).
  //   mem      = member set of the partition
  //   outReach = union of desc[m] over members (nodes reachable FROM partition)
  //   inReach  = union of anc[m]  over members (nodes that can reach partition)
  std::vector<std::vector<uint64_t>> mem(static_cast<size_t>(N)),
      out_reach(static_cast<size_t>(N)), in_reach(static_cast<size_t>(N));
  auto init_root = [&](int r) {
    mem[static_cast<size_t>(r)].assign(static_cast<size_t>(W), 0);
    out_reach[static_cast<size_t>(r)].assign(static_cast<size_t>(W), 0);
    in_reach[static_cast<size_t>(r)].assign(static_cast<size_t>(W), 0);
  };
  for (int i = 0; i < N; ++i) {
    if (!supported[static_cast<size_t>(i)]) continue;
    const int r = dsu.Find(i);
    if (mem[static_cast<size_t>(r)].empty()) init_root(r);
    BitSet(mem[static_cast<size_t>(r)], i);
    BitOrInto(out_reach[static_cast<size_t>(r)], desc[static_cast<size_t>(i)]);
    BitOrInto(in_reach[static_cast<size_t>(r)], anc[static_cast<size_t>(i)]);
  }

  // A candidate partition is convex iff no unsupported node lies between two
  // members: (outReach ∩ inReach ∩ unsupported) is empty.  No allocation — this
  // runs once per edge per fixpoint pass.
  auto union_is_convex = [&](int ra, int rb) -> bool {
    const auto& oa = out_reach[static_cast<size_t>(ra)];
    const auto& ob = out_reach[static_cast<size_t>(rb)];
    const auto& ia = in_reach[static_cast<size_t>(ra)];
    const auto& ib = in_reach[static_cast<size_t>(rb)];
    for (int w = 0; w < W; ++w) {
      const uint64_t o  = oa[static_cast<size_t>(w)] | ob[static_cast<size_t>(w)];
      const uint64_t in = ia[static_cast<size_t>(w)] | ib[static_cast<size_t>(w)];
      if (o & in & unsupported[static_cast<size_t>(w)]) return false;
    }
    return true;
  };

  // Merge maximally along supported producer→consumer edges until fixpoint.
  bool changed = true;
  while (changed) {
    changed = false;
    for (const auto& e : edges) {
      const int a = e.first, b = e.second;
      if (!supported[static_cast<size_t>(a)] || !supported[static_cast<size_t>(b)]) continue;
      const int ra = dsu.Find(a), rb = dsu.Find(b);
      if (ra == rb) continue;
      if (!union_is_convex(ra, rb)) continue;
      // Commit the merge: fold rb's aggregate into ra.
      BitOrInto(mem[static_cast<size_t>(ra)], mem[static_cast<size_t>(rb)]);
      BitOrInto(out_reach[static_cast<size_t>(ra)], out_reach[static_cast<size_t>(rb)]);
      BitOrInto(in_reach[static_cast<size_t>(ra)], in_reach[static_cast<size_t>(rb)]);
      dsu.Merge(ra, rb);
      // dsu.Merge makes ra the parent (Merge(a,b) sets parent[b]=a), so ra
      // remains the representative and its aggregate is authoritative.
      changed = true;
    }
  }

  // ──────────────────────────────────────────────────────────────────────
  // Collect partitions: bucket supported nodes by root, emit in topo order.
  // ──────────────────────────────────────────────────────────────────────
  std::unordered_map<int, std::vector<int>> root_to_nodes;
  for (int i : ctx.topo_order) {
    if (i < 0 || i >= N || !supported[static_cast<size_t>(i)]) continue;
    root_to_nodes[dsu.Find(i)].push_back(i);
  }
  // Also include any supported nodes not present in topo_order (defensive).
  for (int i = 0; i < N; ++i) {
    if (!supported[static_cast<size_t>(i)]) continue;
    const int r = dsu.Find(i);
    auto& v = root_to_nodes[r];
    if (std::find(v.begin(), v.end(), i) == v.end()) v.push_back(i);
  }

  // A partition must have at least one real (non-constant) boundary input.
  // Neuron's identifyInputsAndOutputs() fails with NEURON_UNEXPECTED_NULL on a
  // subgraph with zero inputs, and an all-constant subgraph (e.g. a Concat of
  // Unsqueezed shape constants) belongs on CPU / constant-folding, not on the
  // NPU.  This mirrors the shared provider's GetCapability filter that resets
  // any ComputeCapability whose inputs are all constant initializers.
  auto partition_has_real_input = [&](const std::vector<int>& nodes) -> bool {
    std::unordered_set<std::string> internal_outputs;
    for (int idx : nodes)
      for (const auto& o : graph_proto.node(idx).output())
        if (!o.empty()) internal_outputs.insert(o);
    for (int idx : nodes) {
      for (const auto& in : graph_proto.node(idx).input()) {
        if (in.empty() || ctx.IsInitializer(in)) continue;
        if (internal_outputs.count(in)) continue;  // produced inside partition
        return true;                                // a real external input
      }
    }
    return false;
  };

  // A partition is only worth an NPU dispatch if it contains at least one
  // compute-heavy op (conv / matmul / gemm / attention / recurrent family).
  // Partitions made purely of memory-bound movement/elementwise ops (Reshape,
  // Transpose, Concat, Add, Sub, Slice, …) — e.g. a detection head's DFL/anchor
  // decode — cost more in Neuron dispatch + NPU<->CPU tensor round-trips than
  // they save, so they run faster left on CPU.  Dropping them here keeps the
  // conv backbone on the NPU while the decode stays on CPU as one region.
  auto partition_has_heavy_op = [&](const std::vector<int>& nodes) -> bool {
    for (int idx : nodes) {
      const auto& n = graph_proto.node(idx);
      const auto it = op_builders.find(n.op_type());
      if (it != op_builders.end() && it->second->IsComputeHeavy(n)) return true;
    }
    return false;
  };

  // A final partition must be convex: no unsupported node may lie on a path
  // between two of its members.  The convex-merge loop guarantees this for
  // merges it performs, but the QDQ atomic seeding (which force-unions a shared
  // DQ with all its consumer targets) can in principle produce a non-convex
  // seed.  ORT rejects the ENTIRE EP's capability if any AddNodesToFuse group is
  // non-convex, so we defensively drop such a partition to CPU instead.
  auto partition_is_convex = [&](int root) -> bool {
    const auto& o  = out_reach[static_cast<size_t>(root)];
    const auto& in = in_reach[static_cast<size_t>(root)];
    for (int w = 0; w < W; ++w)
      if (o[static_cast<size_t>(w)] & in[static_cast<size_t>(w)] &
          unsupported[static_cast<size_t>(w)])
        return false;
    return true;
  };

  groups.reserve(root_to_nodes.size());
  for (auto& kv : root_to_nodes) {
    if (kv.second.size() < state->min_group_size) continue;
    if (!partition_has_real_input(kv.second)) {
      LOGS_NEURON(VERBOSE) << "Dropping all-constant partition (size "
                           << kv.second.size() << ") — no non-constant input";
      continue;
    }
    if (!partition_is_convex(kv.first)) {
      LOGS_NEURON(WARNING) << "Dropping non-convex partition (size "
                           << kv.second.size()
                           << ") — falling back to CPU to avoid rejecting all capability";
      continue;
    }

    if (!partition_has_heavy_op(kv.second) &&
        (state->compute_light_min_nodes == 0 ||
         kv.second.size() < state->compute_light_min_nodes)) {
      LOGS_NEURON(VERBOSE) << "Dropping compute-light partition (size "
                           << kv.second.size()
                           << ") — no conv/matmul; runs faster on CPU than as an NPU dispatch";
      continue;
    }
    groups.push_back(std::move(kv.second));
  }

  if (state->max_partition_bytes > 0 || state->partition_count > 0) {
    groups = SplitGroupsByConstantBytes(
        graph_proto, ctx, std::move(groups),
        state->max_partition_bytes > 0 ? state->max_partition_bytes : UINT64_MAX,
        state->partition_count);
  }
  return groups;
}

// ---------------------------------------------------------------------------
// ExtractSubgraphProto
// ---------------------------------------------------------------------------
static void ExtractSubgraphProto(const onnx::GraphProto& parent,
                                 const std::vector<int>& node_indices,
                                 const onnxruntime::neuron::NeuronGraphContext& parent_ctx,
                                 onnx::GraphProto& out) {
  out.set_name("neuron_capability_probe");

  std::unordered_set<int> in_group(node_indices.begin(), node_indices.end());

  // Tensors produced inside the group.
  std::unordered_set<std::string> produced;
  for (int idx : node_indices)
    for (const auto& o : parent.node(idx).output())
      if (!o.empty()) produced.insert(o);

  // Tensors consumed outside the group (or that are graph outputs) must be
  // exposed as subgraph outputs.
  std::unordered_set<std::string> consumed_outside;
  for (int i = 0; i < parent.node_size(); ++i) {
    if (in_group.count(i)) continue;
    for (const auto& inp : parent.node(i).input())
      if (!inp.empty() && produced.count(inp)) consumed_outside.insert(inp);
  }
  for (const auto& go : parent.output())
    if (produced.count(go.name())) consumed_outside.insert(go.name());

  auto add_value_info = [&](const std::string& name, onnx::ValueInfoProto* dst) {
    const auto* vi = parent_ctx.GetValueInfo(name);
    if (vi) {
      *dst = *vi;
    } else {
      dst->set_name(name);
    }
    dst->set_name(name);
  };

  // Nodes, in the parent's topological order.
  std::vector<int> ordered(node_indices);
  std::sort(ordered.begin(), ordered.end());

  std::unordered_set<std::string> added_inputs;
  std::unordered_set<std::string> added_inits;

  for (int idx : ordered) {
    const auto& node = parent.node(idx);
    *out.add_node() = node;

    for (const auto& inp : node.input()) {
      if (inp.empty() || produced.count(inp)) continue;

      if (parent_ctx.IsInitializer(inp)) {
        if (added_inits.insert(inp).second) {
          const auto* init = parent_ctx.GetInitializer(inp);
          if (init) *out.add_initializer() = *init;
        }
        continue;
      }

      if (added_inputs.insert(inp).second)
        add_value_info(inp, out.add_input());
    }
  }

  for (const auto& name : produced) {
    if (consumed_outside.count(name))
      add_value_info(name, out.add_output());
    else
      add_value_info(name, out.add_value_info());
  }

  // keep the last node's outputs so the model is still well-formed
  if (out.output_size() == 0 && !ordered.empty()) {
    for (const auto& o : parent.node(ordered.back()).output())
      if (!o.empty()) add_value_info(o, out.add_output());
  }
}

constexpr const char* kMSInternalNHWCDomain = "com.ms.internal.nhwc";

// ops ORT permutes during layout transformation, which runs after GetCapability
static const std::unordered_set<std::string>& LayoutSensitiveOps() {
  static const std::unordered_set<std::string> kOps = {
      "Conv", "ConvTranspose", "QLinearConv",
      "AveragePool", "MaxPool", "LpPool", "MaxRoiPool",
      "GlobalAveragePool", "GlobalMaxPool", "GlobalLpPool", "QLinearAveragePool",
      "BatchNormalization", "InstanceNormalization", "LRN",
      "Resize", "Upsample", "DepthToSpace", "SpaceToDepth", "GridSample",
  };
  return kOps;
}

// ---------------------------------------------------------------------------
// FilterGroupsByDeviceSupport
// ---------------------------------------------------------------------------
static std::vector<std::vector<int>> FilterGroupsByDeviceSupport(
    const onnx::GraphProto&                        graph_proto,
    const onnxruntime::neuron::NeuronGraphContext& ctx,
    std::vector<std::vector<int>>                  groups,
    const NeuronWrapperState*                      state) {

  if (state->skip_capability_probe || groups.empty()) return groups;

  const auto probe_compiles = [&](const std::vector<int>& group,
                                  std::string& reason) -> bool {
    onnx::GraphProto sub;
    ExtractSubgraphProto(graph_proto, group, ctx, sub);
    try {
      onnxruntime::neuron::NeuronGraphContext sub_ctx(sub);
      onnxruntime::neuron::ModelBuilder probe(
          sub_ctx,
          *state->neuron_handle,
          state->neuron_version,
          state->neuron_target_devices,
          state->target_device_option,
          state->optimization_string);
      probe.SetUseNCHW(state->neuron_flags & NEURON_FLAG_USE_NCHW);
      probe.SetUseFp16(state->neuron_flags & NEURON_FLAG_USE_FP16);

      NeuronStatus ns = probe.ValidateDeviceSupport();
      if (!ns.IsOK()) {
        reason = ns.ErrorMessage();
        return false;
      }
      return true;
    } catch (const std::exception& e) {
      reason = std::string("exception: ") + e.what();
      return false;
    }
  };

  std::vector<std::vector<int>> kept;
  kept.reserve(groups.size());

  for (auto& group : groups) {
    {
      std::unordered_set<std::string> produced;
      for (int idx : group)
        for (const auto& o : graph_proto.node(idx).output())
          if (!o.empty()) produced.insert(o);

      std::vector<int> boundary_q;
      for (int idx : group) {
        const auto& n = graph_proto.node(idx);
        if (n.op_type() != "QuantizeLinear" || n.input_size() == 0) continue;
        const std::string& src = n.input(0);
        if (produced.count(src)) continue;
        if (ctx.GetElementType(src) == onnx::TensorProto::FLOAT)
          boundary_q.push_back(idx);
      }
      if (!boundary_q.empty()) {
        if (state->mdla_float_quantize < 0) {
          std::string reason;
          state->mdla_float_quantize =
              probe_compiles({boundary_q.front()}, reason) ? 1 : 0;
          if (state->mdla_float_quantize == 0)
            LOGS_NEURON(INFO) << "This Neuron stack cannot quantize a float "
                              << "input on the MDLA (" << reason << "); "
                              << "boundary QuantizeLinear nodes will run on CPU.";
        }
        if (state->mdla_float_quantize == 0) {
          const std::unordered_set<int> drop(boundary_q.begin(), boundary_q.end());
          group.erase(std::remove_if(group.begin(), group.end(),
                                     [&](int idx) { return drop.count(idx) != 0; }),
                      group.end());
          if (group.empty()) continue;
        }
      }
    }

    const auto& layout_sensitive = LayoutSensitiveOps();
    bool probe_would_be_stale = false;
    for (int idx : group) {
      const auto& n = graph_proto.node(idx);
      // before layout transformation this is not the graph Compile() gets;
      // after it, ORT converted the node at our request and we must not decline
      if (layout_sensitive.count(n.op_type()) || n.domain() == kMSInternalNHWCDomain) {
        probe_would_be_stale = true;
        break;
      }
    }
    if (probe_would_be_stale) {
      kept.push_back(std::move(group));
      continue;
    }

    bool ok = false;
    std::string reason;
    ok = probe_compiles(group, reason);

    if (ok) {
      kept.push_back(std::move(group));
    } else {
      LOGS_NEURON(INFO) << "Declining partition of " << group.size()
                        << " node(s); the Neuron compiler cannot build it ("
                        << reason << "). These nodes will run on CPU.";
    }
  }

  return kept;
}

}  // anonymous namespace

// ============================================================================
// NeuronWrapper_* implementations (C interface)
// ============================================================================

extern "C" {

// ---------------------------------------------------------------------------
// NeuronWrapper_Create
// ---------------------------------------------------------------------------
NeuronWrapperHandle NeuronWrapper_Create(
    const OrtApi*            ort_api,
    const OrtEpApi*          ep_api,
    const OrtModelEditorApi* model_editor_api,
    const OrtKeyValuePairs*  provider_options,
    uint32_t                 ort_api_version) {

  auto* s = new (std::nothrow) NeuronWrapperState{};
  if (!s) return nullptr;

  s->ort_api          = ort_api;
  s->ep_api           = ep_api;
  s->model_editor_api = model_editor_api;
  s->ort_api_version  = ort_api_version;

  // Parse provider options.
  auto get = [&](const char* k) { return GetKV(ort_api, provider_options, k); };
  if (get("NEURON_FLAG_USE_FP16")     == "1") s->neuron_flags |= NEURON_FLAG_USE_FP16;
  if (get("NEURON_FLAG_USE_NCHW")     == "1") s->neuron_flags |= NEURON_FLAG_USE_NCHW;
  if (get("NEURON_FLAG_CPU_DISABLED") == "1") s->neuron_flags |= NEURON_FLAG_CPU_DISABLED;
  if (get("NEURON_FLAG_CPU_ONLY")     == "1") s->neuron_flags |= NEURON_FLAG_CPU_ONLY;

  auto min_gs = get("NEURON_FLAG_MIN_GROUP_SIZE");
  if (!min_gs.empty()) s->min_group_size = static_cast<uint32_t>(std::stoi(min_gs));

  s->skip_capability_probe = (get("NEURON_FLAG_SKIP_CAPABILITY_PROBE") == "1");

  s->optimization_string = get("NEURON_FLAG_OPTIMIZATION_STRING");

  auto max_pb = get("NEURON_FLAG_MAX_PARTITION_BYTES");
  if (!max_pb.empty()) s->max_partition_bytes = std::stoull(max_pb);
  auto part_count = get("NEURON_FLAG_PARTITION_COUNT");
  if (!part_count.empty()) s->partition_count = static_cast<uint32_t>(std::stoul(part_count));
  auto light_min = get("NEURON_FLAG_COMPUTE_LIGHT_MIN_NODES");
  if (!light_min.empty()) s->compute_light_min_nodes = static_cast<uint32_t>(std::stoul(light_min));

  {
    std::string prefixes = get("NEURON_FLAG_SKIP_NODE_PREFIXES");
    size_t start = 0;
    while (start < prefixes.size()) {
      size_t end = prefixes.find(',', start);
      if (end == std::string::npos) end = prefixes.size();
      if (end > start) s->skip_node_prefixes.emplace_back(prefixes.substr(start, end - start));
      start = end + 1;
    }
  }

  // EPContext (DLA cache) generation.
  s->context_cache_enabled     = get("ep.context_enable") == "1";
  s->user_context_cache_path   = get("ep.context_file_path");
  {
    // ORT documents "0" (sidecar .bin) as the default for ep.context_embed_mode.
    const std::string embed = get("ep.context_embed_mode");
    s->context_embed_mode = (embed == "1");
  }

  // Force-skip ops.
  auto skip_str = get("NEURON_FLAG_FORCE_SKIP_OPS_STRING");
  if (!skip_str.empty()) {
    std::istringstream ss(skip_str);
    std::string token;
    while (std::getline(ss, token, ',')) {
      if (!token.empty()) s->force_skip_ops.insert(token);
    }
  }

  // Neuron SDK handle.
  s->neuron_handle = NeuronApiImplementation();
  if (!s->neuron_handle) { delete s; return nullptr; }
  if (s->neuron_handle->Neuron_getVersion(&s->neuron_version) != NEURON_NO_ERROR) {
    delete s; return nullptr;
  }

  bool cpu_disabled = s->neuron_flags & NEURON_FLAG_CPU_DISABLED;
  bool cpu_only     = s->neuron_flags & NEURON_FLAG_CPU_ONLY;

  if (cpu_disabled && cpu_only) { delete s; return nullptr; }

  s->target_device_option = neuron::wrapper::TargetDeviceOption::ALL_DEVICES;
  if (cpu_disabled) s->target_device_option = neuron::wrapper::TargetDeviceOption::CPU_DISABLED;
  else if (cpu_only) s->target_device_option = neuron::wrapper::TargetDeviceOption::CPU_ONLY;

  auto ns = onnxruntime::neuron::wrapper::GetTargetDevices(
      *s->neuron_handle, s->target_device_option, s->neuron_target_devices);
  if (!ns.IsOK()) { delete s; return nullptr; }

  return s;
}

// ---------------------------------------------------------------------------
// NeuronWrapper_Destroy
// ---------------------------------------------------------------------------
void NeuronWrapper_Destroy(NeuronWrapperHandle handle) {
  delete static_cast<NeuronWrapperState*>(handle);
}

// ---------------------------------------------------------------------------
// NeuronWrapper_GetCapability
// ---------------------------------------------------------------------------
OrtStatus* NeuronWrapper_GetCapability(
    NeuronWrapperHandle    handle,
    const OrtApi*          ort_api,
    const OrtEpApi*        ep_api,
    const OrtGraph*        graph,
    OrtEpGraphSupportInfo* support_info) try {

  auto* state = static_cast<NeuronWrapperState*>(handle);

  // A pre-compiled (EPContext) model carries the DLA in a node attribute and
  // needs none of the op-capability machinery: claim each EPContext node we own
  // as its own single-node partition and return.  This runs before
  // OrtGraphToProto so an embedded multi-MB DLA blob is never copied into a
  // proto just to be discarded, and it bypasses the partition filters below,
  // which would reject an EPContext node as an unknown, compute-light op.
  if (onnxruntime::neuron::GraphHasDlaNode(*ort_api, *graph)) {
    size_t num_nodes = 0;
    if (OrtStatus* st = ort_api->Graph_GetNumNodes(graph, &num_nodes)) return st;

    std::vector<const OrtNode*> nodes(num_nodes, nullptr);
    if (OrtStatus* st = ort_api->Graph_GetNodes(graph, nodes.data(), nodes.size()))
      return st;

    size_t claimed = 0;
    for (const OrtNode* node : nodes) {
      if (node == nullptr || !onnxruntime::neuron::IsValidDlaNode(*ort_api, *node))
        continue;

      OrtNodeFusionOptions opts{};
      opts.ort_version_supported      = state->ort_api_version;
      opts.drop_constant_initializers = false;

      if (OrtStatus* st = ep_api->EpGraphSupportInfo_AddNodesToFuse(
              support_info, &node, 1, &opts))
        return st;
      ++claimed;
    }

    LOGS_NEURON(INFO) << "NeuronEP GetCapability: EPContext model, claimed "
                      << claimed << " of " << num_nodes << " nodes";
    return nullptr;
  }

  // Ort::Status owns the OrtStatus; assigning to a raw OrtStatus* frees it here
  onnx::GraphProto graph_proto;
  {
    Ort::Status st = OrtEpUtils::OrtGraphToProto(*graph, graph_proto);
    if (!st.IsOK()) return st.release();
  }

  // Find supported node groups from the static op-capability checks, then
  // confirm each one against the hardware before claiming it.
  onnxruntime::neuron::NeuronGraphContext ctx(graph_proto);
  const auto proposed = FindSupportedGroups(graph_proto, state);
  const auto groups   = FilterGroupsByDeviceSupport(graph_proto, ctx, proposed, state);

  {
    size_t supported_nodes = 0;
    for (const auto& g : groups) supported_nodes += g.size();
    LOGS_NEURON(INFO) << "NeuronEP GetCapability: partitions=" << groups.size()
                      << " (proposed=" << proposed.size() << ")"
                      << " supported_nodes=" << supported_nodes
                      << " total_nodes=" << graph_proto.node_size();
  }

  if (groups.empty()) return nullptr;

  // Build a table of all OrtNode* pointers in graph order.
  // OrtGraphToProto serialises nodes via ort_graph.GetNodes() which iterates in
  // the same order as Graph_GetNode(graph, i, …), so graph_proto.node(i)
  // corresponds to all_ort_nodes[i].  We additionally build a name→index map
  // for a name-based cross-check on nodes that do carry names.
  size_t num_graph_nodes = 0;
  ort_api->Graph_GetNumNodes(graph, &num_graph_nodes);

  std::vector<const OrtNode*> all_ort_nodes(num_graph_nodes, nullptr);
  ort_api->Graph_GetNodes(graph, all_ort_nodes.data(), num_graph_nodes);
  std::unordered_map<std::string, size_t> ort_node_name_to_idx;
  for (size_t i = 0; i < num_graph_nodes; ++i) {
    if (all_ort_nodes[i]) {
      const char* nname = nullptr;
      ort_api->Node_GetName(all_ort_nodes[i], &nname);
      if (nname && nname[0] != '\0')
        ort_node_name_to_idx.emplace(nname, i);
    }
  }

  // For each group, collect OrtNode* pointers and register with support_info.
  for (const auto& group : groups) {
    std::vector<const OrtNode*> group_nodes;
    group_nodes.reserve(group.size());

    for (int proto_idx : group) {
      const auto& proto_node = graph_proto.node(proto_idx);
      const OrtNode* ort_node = nullptr;

      // Primary: look up by name (works for named nodes).
      if (!proto_node.name().empty()) {
        auto it = ort_node_name_to_idx.find(proto_node.name());
        if (it != ort_node_name_to_idx.end())
          ort_node = all_ort_nodes[it->second];
      }

      // Fallback: use the same index — valid because OrtGraphToProto and
      // Graph_GetNode(i) both traverse nodes in the same topological order.
      if (!ort_node && static_cast<size_t>(proto_idx) < num_graph_nodes)
        ort_node = all_ort_nodes[proto_idx];

      if (ort_node)
        group_nodes.push_back(ort_node);
    }

    if (group_nodes.empty()) continue;

    // Compile copies every constant into Neuron operands and inference never
    // reads ORT's, so let ORT release its copy once Compile returns.
    OrtNodeFusionOptions opts{};
    opts.ort_version_supported      = state->ort_api_version;
    opts.drop_constant_initializers =
        !GroupReadsConstantAtRuntime(graph_proto, ctx, group);

    OrtStatus* st = ep_api->EpGraphSupportInfo_AddNodesToFuse(
        support_info, group_nodes.data(), group_nodes.size(), &opts);
    if (st) return st;
  }

  return nullptr;
} catch (const std::exception& e) {
  return ort_api->CreateStatus(ORT_EP_FAIL,
      (std::string("NeuronEP GetCapability exception: ") + e.what()).c_str());
} catch (...) {
  // ORT calls GetCapability through a noexcept C function pointer, so anything
  // that escapes here calls std::terminate and takes the whole process down.
  // A failed GetCapability is harmless — ORT treats it as claiming nothing.
  return ort_api->CreateStatus(ORT_EP_FAIL,
      "NeuronEP GetCapability: unknown exception");
}

// ---------------------------------------------------------------------------
// NeuronWrapper_Compile
//
// KEY CHANGE: Uses ort_graph_to_proto.h to convert the fused subgraph OrtGraph*
// to onnx::GraphProto, then builds the Neuron model from protobuf.
// NO reinterpret_cast to ORT internal types.
// ---------------------------------------------------------------------------
OrtStatus* NeuronWrapper_Compile(
    NeuronWrapperHandle          handle,
    const OrtApi*                ort_api,
    const OrtEpApi*              ep_api,
    const OrtModelEditorApi*     /*model_editor_api*/,
    const OrtGraph* const*       graphs,
    const OrtNode*  const*       fused_nodes,
    size_t                       count,
    OrtNodeComputeInfo**         node_compute_infos,
    OrtNode**                    ep_context_nodes) try {

  auto* state = static_cast<NeuronWrapperState*>(handle);

  // Path of the model being compiled; empty when it was loaded from memory.
  std::string source_model_path;
  if (count > 0) {
    const ORTCHAR_T* mp = nullptr;
    if (ort_api->Graph_GetModelPath(graphs[0], &mp) == nullptr && mp != nullptr)
      source_model_path = std::filesystem::path(mp).string();
  }
  const std::filesystem::path context_model_path =
      onnxruntime::neuron::GetContextModelPath(state->user_context_cache_path,
                                               source_model_path);

  // Each partition gets its own DLA and its own EPContext node.  ORT requires a
  // node for every fused node once generation is enabled, so this must cover
  // all of them or none.  Not applicable when we are replaying a context model
  // we just loaded.
  const bool graph_is_context_model =
      count > 0 && onnxruntime::neuron::GraphHasDlaNode(*ort_api, *graphs[0]);
  const bool can_store_dla =
      state->context_cache_enabled && count > 0 && !graph_is_context_model;

  // Real model I/O per partition, collected as each one compiles.  Used to give
  // the EPContext node the same inputs/outputs the Neuron model actually has.
  std::vector<std::string> ctx_partition_names;
  std::vector<std::vector<std::string>> ctx_input_names, ctx_output_names;

  std::vector<std::filesystem::path> compiled_dla_paths;
  if (can_store_dla) {
    static std::atomic<uint64_t> dla_seq{0};
    const uint64_t batch = dla_seq.fetch_add(1);
    compiled_dla_paths.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      compiled_dla_paths.push_back(
          std::filesystem::temp_directory_path() /
          ("neuron_ep_" + std::to_string(::getpid()) + "_" +
           std::to_string(batch) + "_" + std::to_string(i) + ".dla"));
    }
  }

  for (size_t i = 0; i < count; ++i) {
    // Get the fused node name.
    const char* fused_name_cstr = nullptr;
    ort_api->Node_GetName(fused_nodes[i], &fused_name_cstr);
    std::string fused_name = fused_name_cstr ? fused_name_cstr : "";

    // see the ownership note in NeuronWrapper_GetCapability
    onnx::GraphProto subgraph_proto;
    {
      Ort::Status st = OrtEpUtils::OrtGraphToProto(*graphs[i], subgraph_proto);
      if (!st.IsOK()) return st.release();
    }

    // Build NeuronGraphContext.
    onnxruntime::neuron::NeuronGraphContext ctx(subgraph_proto);

    // Build and compile the Neuron model.
    onnxruntime::neuron::ModelBuilder builder(
        ctx,
        *state->neuron_handle,
        state->neuron_version,
        state->neuron_target_devices,
        state->target_device_option,
        state->optimization_string);

    builder.SetUseNCHW(state->neuron_flags & NEURON_FLAG_USE_NCHW);
    builder.SetUseFp16(state->neuron_flags & NEURON_FLAG_USE_FP16);
    builder.SetEpContextDir(context_model_path.parent_path());
    if (can_store_dla) {
      builder.SetStoreDla(true);
      builder.SetDlaPath(compiled_dla_paths[i]);
    }

    // An adapter-formatted DLA is restored whole via
    // NeuronModel_restoreFromCompiledNetwork.  Every other EPContext node is
    // restored by ComposeNeuronModel(), which lowers it to a single
    // com.mediatek.compiled_network RESTORE operation, so it goes through the
    // ordinary Compile() path.
    std::unique_ptr<onnxruntime::neuron::wrapper::Model> model;
    NeuronStatus ns;
    if (onnxruntime::neuron::GraphHasAdapterFormattedDlaNode(ctx)) {
      ns = builder.RestoredFromAdapterFormattedDla(model);
    } else {
      // Build and finish the Neuron model first.  Its weights then live in the
      // builder's own buffers, so drop the proto's copy before the compile,
      // which is the long, memory-heavy step.
      ns = builder.ValidateDeviceSupport();
      if (ns.IsOK()) {
        ReleaseInitializerData(subgraph_proto);
        ns = builder.Compile(model);
      }
    }
    if (!ns.IsOK()) {
      return ort_api->CreateStatus(ORT_EP_FAIL, ns.ErrorMessage().c_str());
    }

    // Build input / output maps from the fused node I/O.
    {
      size_t in_count = 0;
      ort_api->Node_GetNumInputs(fused_nodes[i], &in_count);
      std::vector<const OrtValueInfo*> inputs(in_count);
      ort_api->Node_GetInputs(fused_nodes[i], inputs.data(), in_count);
      std::unordered_map<std::string, size_t> input_map;
      for (size_t j = 0; j < in_count; ++j) {
        const char* name_cstr = nullptr;
        ort_api->GetValueInfoName(inputs[j], &name_cstr);
        if (name_cstr) input_map[name_cstr] = j;
      }
      model->SetInputMap(std::move(input_map));
    }
    {
      size_t out_count = 0;
      ort_api->Node_GetNumOutputs(fused_nodes[i], &out_count);
      std::vector<const OrtValueInfo*> outputs(out_count);
      ort_api->Node_GetOutputs(fused_nodes[i], outputs.data(), out_count);
      std::unordered_map<std::string, size_t> output_map;
      for (size_t j = 0; j < out_count; ++j) {
        const char* name_cstr = nullptr;
        ort_api->GetValueInfoName(outputs[j], &name_cstr);
        if (name_cstr) output_map[name_cstr] = j;
      }
      model->SetOutputMap(std::move(output_map));
    }

    if (can_store_dla) {
      ctx_partition_names.push_back(fused_name);
      ctx_input_names.push_back(model->GetInputs());
      ctx_output_names.push_back(model->GetOutputs());
    }

    state->neuron_models.emplace(fused_name, std::move(model));

    // Allocate OrtNodeComputeInfo.
    auto* nci = new (std::nothrow) NeuronNodeComputeInfo(state, fused_name, ort_api);
    if (!nci) return ort_api->CreateStatus(ORT_EP_FAIL, "NeuronEP: OOM");
    node_compute_infos[i] = nci;
  }

  // ORT assembles and writes the context model itself; the EP only supplies the
  // nodes.  ep_context_nodes is null when the session is not generating one.
  if (can_store_dla && ep_context_nodes != nullptr) {
    NeuronStatus ns = onnxruntime::neuron::CreateEpContextNodes(
        *ort_api, *state->model_editor_api, ctx_partition_names,
        ctx_input_names, ctx_output_names,
        ep_context_nodes, state->neuron_version, compiled_dla_paths,
        context_model_path, state->context_embed_mode);
    for (const auto& p : compiled_dla_paths) {
      std::error_code ec;
      std::filesystem::remove(p, ec);
    }
    if (!ns.IsOK())
      return ort_api->CreateStatus(ORT_EP_FAIL, ns.ErrorMessage().c_str());
  }

  return nullptr;
} catch (const std::exception& e) {
  return ort_api->CreateStatus(ORT_EP_FAIL,
      (std::string("NeuronEP Compile exception: ") + e.what()).c_str());
} catch (...) {
  return ort_api->CreateStatus(ORT_EP_FAIL, "NeuronEP Compile: unknown exception");
}

// ---------------------------------------------------------------------------
// NeuronWrapper_ReleaseNodeComputeInfos
// ---------------------------------------------------------------------------
void NeuronWrapper_ReleaseNodeComputeInfos(
    NeuronWrapperHandle  /*handle*/,
    OrtNodeComputeInfo** node_compute_infos,
    size_t               count) {
  for (size_t i = 0; i < count; ++i) {
    delete static_cast<NeuronNodeComputeInfo*>(node_compute_infos[i]);
  }
}

// ---------------------------------------------------------------------------
// NeuronWrapper_GetPreferredLayout
// ---------------------------------------------------------------------------
int NeuronWrapper_GetPreferredLayout(NeuronWrapperHandle handle) {
  auto* state = static_cast<NeuronWrapperState*>(handle);
  bool use_nchw = (state->neuron_flags & NEURON_FLAG_USE_NCHW) != 0;
  return use_nchw ? static_cast<int>(OrtEpDataLayout_NCHW)
                  : static_cast<int>(OrtEpDataLayout_NHWC);
}

// ---------------------------------------------------------------------------
// NeuronWrapper_ShouldSkipLayoutConversion
// ---------------------------------------------------------------------------
int NeuronWrapper_ShouldSkipLayoutConversion(
    NeuronWrapperHandle /*handle*/,
    const char*         /*domain*/,
    const char*         op_type,
    int                 /*target_layout*/) {
  if (!op_type) return 1;

  // Only spatial 4-D ops benefit from NHWC layout conversion.
  // Every other op must keep its original layout to avoid the transpose
  // optimizer applying a 4-D permutation [0,2,3,1] to non-4-D tensors.
  static const std::unordered_set<std::string_view> nhwc_ops = {
    "Conv", "ConvTranspose",
    "AveragePool", "MaxPool", "GlobalAveragePool", "GlobalMaxPool",
    "BatchNormalization", "InstanceNormalization",
    "Resize", "Upsample",
    "DepthToSpace", "SpaceToDepth",
    "LRN",
  };
  return nhwc_ops.count(op_type) ? 0 : 1;
}

}  // extern "C"
