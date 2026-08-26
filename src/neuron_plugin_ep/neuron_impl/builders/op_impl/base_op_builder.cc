// base_op_builder.cc
// Plugin EP version.
//
// Ported from in-tree base_op_builder.cc (292 lines) to the standalone plugin
// EP format.  The in-tree version depended on ORT internals (GraphViewer,
// NodeUnit, Initializer, utils::HasExternalDataInFile, …).  All such
// dependencies have been replaced with:
//   - NeuronGraphContext    (replaces GraphViewer)
//   - onnx::NodeProto       (replaces NodeUnit)
//   - NeuronStatus          (replaces onnxruntime::Status)
//
// External-initializer guard:
//   The in-tree HasExternalInitializer() loop iterated over NodeUnit quant_param
//   fields that do not exist in the plugin EP.  The plugin EP processes QDQ nodes
//   as plain QuantizeLinear / DequantizeLinear nodes; there is no separate
//   quant_param sidecar.  We therefore implement a simpler guard that rejects
//   any initializer whose TensorProto has data_location == EXTERNAL.

#include "base_op_builder.h"

#include <cstdint>
#include <vector>

#include <onnx/onnx_pb.h>

#include "../../op_capability_abi/op_capability_loader.h"

namespace onnxruntime {
namespace neuron {

namespace {

// Flattens OpSupportCheckParams into the POD form the C ABI accepts.
// `name_storage` must outlive the returned struct: device_names points into it.
NeuronCapParams ToCapParams(const OpSupportCheckParams& params,
                            std::vector<const char*>& name_storage) {
  name_storage.clear();
  name_storage.reserve(params.devices.size());
  for (const auto& device : params.devices) {
    name_storage.push_back(device.name.c_str());
  }

  NeuronCapParams out{};
  out.struct_size = static_cast<uint32_t>(sizeof(NeuronCapParams));
  out.version_major = static_cast<uint32_t>(params.version.major);
  out.version_minor = static_cast<uint32_t>(params.version.minor);
  out.version_patch = static_cast<uint32_t>(params.version.patch);
  out.device_names = name_storage.empty() ? nullptr : name_storage.data();
  out.device_count = name_storage.size();
  out.use_nchw = params.use_nchw ? 1 : 0;
  out.use_fp16 = params.use_fp16 ? 1 : 0;
  return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// AddToModelBuilder
// ---------------------------------------------------------------------------
NeuronStatus BaseOpBuilder::AddToModelBuilder(ModelBuilder& model_builder,
                                               const onnx::NodeProto& node,
                                               const NeuronGraphContext& ctx) const {
  NEURON_RETURN_IF_ERROR(AddToModelBuilderImpl(model_builder, node, ctx));
  LOGS_NEURON(VERBOSE) << "Operator name: [" << node.name()
                        << "] type: [" << node.op_type() << "] was added";
  return NeuronStatus::OK();
}

// ---------------------------------------------------------------------------
// IsOpSupported
// ---------------------------------------------------------------------------
bool BaseOpBuilder::IsOpSupported(const onnx::NodeProto& node,
                                   const NeuronGraphContext& ctx,
                                   const neuron::OpSupportCheckParams& params) const {
  // Version check runs unconditionally — GetMinSupportedNeuronVersion lives on
  // BaseOpBuilder, not in the op-capability library.
  const auto min_ver = GetMinSupportedNeuronVersion(node, params);
  if (!GreaterEqualVersion(params.version, min_ver)) {
    LOGS_NEURON(VERBOSE) << "Node [" << node.name() << "] op_type [" << node.op_type()
                         << "]: requires Neuron >= " << GetNeuronVersionDescription(min_ver);
    return false;
  }

  // OpSet check runs unconditionally.
  if (!HasSupportedOpSet(node)) return false;

  const auto& loader = op_capability::OpCapabilityLoader::Instance();

  // libneuron_op_capability.so is a hard dependency — EP init (CreateEpFactories)
  // already failed if it was not loaded. This path should never be reached.
  if (!loader.IsLoaded()) {
    LOGS_NEURON(ERROR) << "libneuron_op_capability.so is not loaded; "
                       << "rejecting operator [" << node.op_type() << "].";
    return false;
  }

  const int32_t has_capability = loader.HasCapability(node.op_type());
  if (has_capability == NEURON_CAP_ERROR) {
    // ERROR must not fall through to the default-supported path below.
    LOGS_NEURON(ERROR) << "Capability lookup failed for op_type [" << node.op_type()
                       << "]; rejecting operator.";
    return false;
  }

  if (has_capability == NEURON_CAP_TRUE) {
    // The library addresses nodes by index; resolve the reference back to one.
    const int node_index = ctx.GetNodeIndex(node);
    if (node_index < 0 || params.cap_graph == nullptr) {
      LOGS_NEURON(ERROR) << "Cannot query capability for node [" << node.name()
                         << "] op_type [" << node.op_type() << "]: "
                         << (node_index < 0 ? "node does not belong to this graph context"
                                            : "no capability graph handle for this call")
                         << "; rejecting operator.";
      return false;
    }

    const int32_t unit_type_ok =
        loader.IsNodeUnitTypeSupported(params.cap_graph, node_index);
    if (unit_type_ok == NEURON_CAP_ERROR) {
      LOGS_NEURON(ERROR) << "Capability check failed for node [" << node.name()
                         << "] op_type [" << node.op_type()
                         << "] (unit type); rejecting operator.";
      return false;
    }
    if (unit_type_ok != NEURON_CAP_TRUE) return false;

    if (!HasSupportedInputs(node, ctx, params)) return false;
    if (!HasSupportedOutputs(node, ctx, params)) return false;

    // Reject file-backed external initializers.
    for (int i = 0; i < node.input_size(); ++i) {
      const auto& name = node.input(i);
      if (name.empty()) continue;
      const auto* tp = ctx.GetInitializer(name);
      if (!tp) continue;
      if (tp->data_location() == onnx::TensorProto::EXTERNAL) {
        LOGS_NEURON(VERBOSE) << "Initializer [" << name
                             << "] has external data location – not supported";
        return false;
      }
    }

    std::vector<const char*> device_names;
    const NeuronCapParams cap_params = ToCapParams(params, device_names);

    const int32_t supported =
        loader.IsNodeSupported(params.cap_graph, node_index, &cap_params);
    if (supported == NEURON_CAP_ERROR) {
      LOGS_NEURON(ERROR) << "Capability check failed for node [" << node.name()
                         << "] op_type [" << node.op_type() << "]; rejecting operator.";
      return false;
    }
    return supported == NEURON_CAP_TRUE;
  }

  // Op exists in loaded library but has no capability registered —
  // treat as supported (no HW-specific constraints for this op).
  LOGS_NEURON(VERBOSE) << "No capability registered for operator ["
                       << node.op_type() << "], treating as supported.";
  return true;
}

// ---------------------------------------------------------------------------
// HasSupportedOpSet
// ---------------------------------------------------------------------------
bool BaseOpBuilder::HasSupportedOpSet(const onnx::NodeProto& node) const {
  // In the plugin EP the per-node opset version is not directly available on
  // NodeProto.  Op-level constraints are enforced by the op_capability library
  // and the opset import list in the model.  We keep this as an always-pass stub
  // so subclasses that override GetMinSupportedOpSet / GetMaxSupportedOpSet
  // can document their requirements without breaking the base.
  (void)node;
  return true;
}

// ---------------------------------------------------------------------------
// HasSupportedInputs
// ---------------------------------------------------------------------------
bool BaseOpBuilder::HasSupportedInputs(const onnx::NodeProto& node,
                                        const NeuronGraphContext& ctx,
                                        const neuron::OpSupportCheckParams& params) const {
  // Static-shape guard: we do not support dynamic-shape inputs.
  for (int i = 0; i < node.input_size(); ++i) {
    const auto& name = node.input(i);
    if (name.empty()) continue;
    // Initializers always have fully static shapes – skip them.
    if (ctx.IsInitializer(name)) continue;
    if (!ctx.HasStaticShape(name)) {
      LOGS_NEURON(VERBOSE) << "Node [" << node.name() << "] op_type [" << node.op_type()
                           << "] input[" << i << "] '" << name << "' has unknown/dynamic shape";
      return false;
    }
  }
  return HasSupportedInputsImpl(node, ctx, params);
}

// ---------------------------------------------------------------------------
// HasSupportedOutputs
// ---------------------------------------------------------------------------
bool BaseOpBuilder::HasSupportedOutputs(const onnx::NodeProto& node,
                                         const NeuronGraphContext& ctx,
                                         const neuron::OpSupportCheckParams& params) const {
  return HasSupportedOutputsImpl(node, ctx, params);
}

namespace {
bool IsRepresentableElementType(const onnx::NodeProto& node,
                                const NeuronGraphContext& ctx,
                                const std::string& name,
                                const char* direction,
                                int index) {
  if (name.empty()) return true;

  const int32_t elem_type = ctx.GetElementType(name);
  if (elem_type == 0) {
    return true;
  }

  switch (elem_type) {
    case onnx::TensorProto::FLOAT:
    case onnx::TensorProto::FLOAT16:
    case onnx::TensorProto::UINT8:
    case onnx::TensorProto::INT8:
    case onnx::TensorProto::INT32:
    case onnx::TensorProto::BOOL:
      return true;
    default:
      LOGS_NEURON(VERBOSE) << "Node [" << node.name() << "] op_type ["
                           << node.op_type() << "]: unsupported " << direction
                           << "[" << index << "] type " << elem_type;
      return false;
  }
}
}  // namespace

// ---------------------------------------------------------------------------
// HasSupportedInputsImpl
// ---------------------------------------------------------------------------
bool BaseOpBuilder::HasSupportedInputsImpl(
    const onnx::NodeProto& node,
    const NeuronGraphContext& ctx,
    const neuron::OpSupportCheckParams& /*params*/) const {
  if (node.input_size() == 0) return true;
  return IsRepresentableElementType(node, ctx, node.input(0), "input", 0);
}

// ---------------------------------------------------------------------------
// HasSupportedOutputsImpl
// ---------------------------------------------------------------------------
bool BaseOpBuilder::HasSupportedOutputsImpl(
    const onnx::NodeProto& node,
    const NeuronGraphContext& ctx,
    const neuron::OpSupportCheckParams& /*params*/) const {
  if (node.output_size() == 0) return true;
  return IsRepresentableElementType(node, ctx, node.output(0), "output", 0);
}

// ---------------------------------------------------------------------------
// IsTypeSupported
// ---------------------------------------------------------------------------
bool BaseOpBuilder::IsTypeSupported(const onnx::NodeProto& node,
                                     const NeuronGraphContext& ctx,
                                     size_t index,
                                     bool is_input,
                                     const std::unordered_set<int32_t>& supported_types) const {
  const auto& names = is_input ? node.input() : node.output();
  if (static_cast<int>(index) >= names.size()) return false;
  const auto& name = names[index];
  if (name.empty()) return true;  // optional / absent input

  int32_t elem_type = ctx.GetElementType(name);
  if (elem_type == 0) return true;  // unknown – accept

  if (!supported_types.count(elem_type)) {
    LOGS_NEURON(VERBOSE) << "Node [" << node.name() << "] op_type ["
                         << node.op_type() << "]: "
                         << (is_input ? "input" : "output") << "[" << index
                         << "] type " << elem_type << " not in supported set";
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// IsQuantizedTypeSupported
// ---------------------------------------------------------------------------
bool BaseOpBuilder::IsQuantizedTypeSupported(
    const onnx::NodeProto& node,
    const NeuronGraphContext& ctx,
    size_t index,
    bool is_input,
    const std::set<std::tuple<int32_t, bool>>& /*supported_quantized_types*/) const {
  // Default: delegate to plain IsTypeSupported with INT8/UINT8.
  return IsTypeSupported(node, ctx, index, is_input,
                         {onnx::TensorProto::UINT8, onnx::TensorProto::INT8});
}

}  // namespace neuron
}  // namespace onnxruntime
