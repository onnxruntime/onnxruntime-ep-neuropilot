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

#include <onnx/onnx_pb.h>

#include "../../op_capability_abi/op_capability_loader.h"

namespace onnxruntime {
namespace neuron {

// ---------------------------------------------------------------------------
// AddToModelBuilder (non-virtual, calls AddToModelBuilderImpl)
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

  const auto* capability = loader.GetCapability(node.op_type());
  if (capability) {
    if (!capability->IsNodeUnitTypeSupported(node)) return false;

    if (!HasSupportedInputOutputs(node, ctx, params)) return false;

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

    return capability->IsOpSupported(node, ctx, params);
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
// HasSupportedInputOutputs (delegates to subclass hook)
// ---------------------------------------------------------------------------
bool BaseOpBuilder::HasSupportedInputOutputs(const onnx::NodeProto& node,
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
  return HasSupportedInputOutputsImpl(node, ctx, params);
}

// ---------------------------------------------------------------------------
// HasSupportedInputOutputsImpl (default: check input 0 is float or quantized)
// ---------------------------------------------------------------------------
bool BaseOpBuilder::HasSupportedInputOutputsImpl(
    const onnx::NodeProto& node,
    const NeuronGraphContext& ctx,
    const neuron::OpSupportCheckParams& /*params*/) const {
  if (node.input_size() == 0) return true;  // e.g. Constant

  const auto& input0 = node.input(0);
  if (input0.empty()) return true;

  int32_t elem_type = ctx.GetElementType(input0);
  if (elem_type == 0) {
    // Type not found – might be a dynamic input or an initializer.
    // Accept it and let the op builder fail at compile time if needed.
    return true;
  }
  // Accept float, float16, uint8, int8, int32, bool.
  // INT64 is intentionally excluded: Neuron SDK cannot represent INT64 operands,
  // so any node whose primary input is INT64 must fall back to CPU.
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
                           << node.op_type() << "]: unsupported input type "
                           << elem_type;
      return false;
  }
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
