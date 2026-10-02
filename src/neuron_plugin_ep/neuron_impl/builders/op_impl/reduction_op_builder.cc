// reduction_op_builder.cc
// Plugin EP version – uses onnx::NodeProto + NeuronGraphContext.
//
// CHANGES FROM ORIGINAL:
//   node_unit.OpType() → node.op_type()
//   node_unit.Name() → node.name()
//   node_unit.Inputs()[i].node_arg.Name() → node.input(i)
//   node_unit.Outputs()[i].node_arg.Name() → node.output(i)
//   node_unit.SinceVersion() → detected via presence of input[1] (see below)
//   inputs[1].node_arg.Exists() → !node.input(1).empty()
//   model_builder.GetInitializerTensors().at(name) → *ctx.GetInitializer(name)
//   Initializer unpacked_tensor(...).DataAsSpan<int64_t>() → GetInt64Data(...)
//   helper.Get("keepdims", 1) → NeuronGraphContext::GetAttrInt(node, "keepdims", 1)
//   helper.Get("noop_with_empty_axes", 0) → NeuronGraphContext::GetAttrInt(node, ...)
//   helper.Get("axes", vector<int64_t>{}) → NeuronGraphContext::GetAttrInts(node, "axes")
//   graph_viewer.GetConstantInitializer(name) → ctx.IsInitializer(name)
//   ORT_RETURN_IF_ERROR → NEURON_RETURN_IF_ERROR
//   ORT_MAKE_STATUS → NEURON_MAKE_STATUS
//   Status → NeuronStatus
// Shared by: {"ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd", "ReduceSum"}

#include "base_op_builder.h"
#include "../op_builder_helpers.h"

namespace onnxruntime {
namespace neuron {
using namespace onnxruntime::neuron::op_builder_helpers;
using namespace onnxruntime::neuron::wrapper;

namespace {

// ONNX moved "axes" from an attribute to input[1] in opset 18 (opset 13 for
// ReduceSum). OrtGraphToProto emits no "since_version" attribute, so the opset
// number is unavailable here; the two forms are instead told apart by whether
// input[1] is present, which is equivalent: the older form declares only one
// input, and the newer form has no "axes" attribute.
bool ReductionAxesFromInput(const onnx::NodeProto& node) {
  return node.input_size() > 1 && !node.input(1).empty();
}

// Returns 0 for the axes-from-input form when the axes tensor is not a readable
// constant, so callers must reject those nodes on their own.
size_t ReductionAxesCount(const onnx::NodeProto& node, const NeuronGraphContext& ctx) {
  if (!ReductionAxesFromInput(node)) {
    return NeuronGraphContext::GetAttrInts(node, "axes").size();
  }
  const auto* axes_tensor = ctx.GetInitializer(node.input(1));
  if (axes_tensor == nullptr) return 0;
  std::vector<int64_t> raw_axes;
  if (!NeuronGraphContext::GetInt64Data(*axes_tensor, raw_axes).IsOK()) return 0;
  return raw_axes.size();
}

}  // namespace

class ReductionOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder,
                              const onnx::NodeProto& node,
                              const NeuronGraphContext& /*ctx*/) const override;

 private:
  NeuronStatus AddToModelBuilderImpl(ModelBuilder& model_builder,
                                      const onnx::NodeProto& node,
                                      const NeuronGraphContext& ctx) const override;

  bool HasSupportedInputsImpl(const onnx::NodeProto& node,
                              const NeuronGraphContext& ctx,
                              const neuron::OpSupportCheckParams& params) const override;
};

bool ReductionOpBuilder::HasSupportedInputsImpl(
    const onnx::NodeProto& node,
    const NeuronGraphContext& ctx,
    const neuron::OpSupportCheckParams& params) const {
  if (ReductionAxesFromInput(node) && !ctx.IsInitializer(node.input(1))) {
    LOGS_NEURON(VERBOSE) << "Node [" << node.name() << "] op_type [" << node.op_type()
                         << "]: axes input '" << node.input(1)
                         << "' is not a constant initializer";
    return false;
  }

  // Empty axes with noop_with_empty_axes=1 is an identity, which has no
  // corresponding Neuron operation.
  if (NeuronGraphContext::GetAttrInt(node, "noop_with_empty_axes", 0) != 0 &&
      ReductionAxesCount(node, ctx) == 0) {
    LOGS_NEURON(VERBOSE) << "Node [" << node.name() << "] op_type [" << node.op_type()
                         << "]: noop_with_empty_axes=1 with empty axes is not supported";
    return false;
  }

  return BaseOpBuilder::HasSupportedInputsImpl(node, ctx, params);
}

void ReductionOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder,
                                                 const onnx::NodeProto& node,
                                                 const NeuronGraphContext& /*ctx*/) const {
  // The axes tensor (input 1) is consumed directly; skip it.
  if (ReductionAxesFromInput(node)) {
    model_builder.AddInitializerToSkip(node.input(1));
  }
}

NeuronStatus ReductionOpBuilder::AddToModelBuilderImpl(
    ModelBuilder& model_builder,
    const onnx::NodeProto& node,
    const NeuronGraphContext& ctx) const {
  const auto& op_type = node.op_type();
  const auto& input   = node.input(0);
  const auto& output  = node.output(0);

  auto& shaper(model_builder.GetShaper());
  const auto input_shape = shaper[input];
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  int32_t op_code;
  if (op_type == "ReduceMean") {
    op_code = NEURON_MEAN;
  } else if (op_type == "ReduceMax") {
    op_code = NEURON_REDUCE_MAX;
  } else if (op_type == "ReduceMin") {
    op_code = NEURON_REDUCE_MIN;
  } else if (op_type == "ReduceProd") {
    op_code = NEURON_REDUCE_PROD;
  } else if (op_type == "ReduceSum") {
    op_code = NEURON_REDUCE_SUM;
  } else {
    return NEURON_MAKE_STATUS("ReductionOpBuilder, unknown op: ", op_type);
  }

  const bool keepdims             = NeuronGraphContext::GetAttrInt(node, "keepdims", 1) != 0;
  const bool noop_with_empty_axes = NeuronGraphContext::GetAttrInt(node, "noop_with_empty_axes", 0) != 0;

  // Collect axes.
  std::vector<int32_t> axes;
  if (ReductionAxesFromInput(node)) {
    const auto* axes_tensor = ctx.GetInitializer(node.input(1));
    if (!axes_tensor) {
      LOGS_NEURON(ERROR) << "Neuron reduction: axes input '" << node.input(1)
                         << "' is not a constant initializer.";
      return NeuronStatus{false, "Non-constant axes for reduction operation"};
    }
    std::vector<int64_t> raw_axes;
    NEURON_RETURN_IF_ERROR(NeuronGraphContext::GetInt64Data(*axes_tensor, raw_axes));
    axes = OnnxAxesToNeuron(gsl::make_span(raw_axes), input_shape.size());
  } else {
    const auto axes_int64 = NeuronGraphContext::GetAttrInts(node, "axes");
    axes = OnnxAxesToNeuron(gsl::make_span(axes_int64), input_shape.size());
  }

  // If axes is still empty and noop_with_empty_axes is not set, reduce all dimensions.
  if (axes.empty() && !noop_with_empty_axes) {
    axes.resize(input_shape.size());
    std::iota(axes.begin(), axes.end(), 0);
  }

  // Emitting no operation here would leave `output` unregistered and break its
  // consumers; HasSupportedInputsImpl already keeps such nodes off this EP.
  if (axes.empty()) {
    return NEURON_MAKE_STATUS("ReductionOpBuilder, ", op_type,
                              ": noop_with_empty_axes=1 with empty axes is not supported");
  }

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));  // data

  const auto axes_name = model_builder.GetUniqueName(node.name() + input + "_axes");
  Shape axes_dimen = {static_cast<uint32_t>(axes.size())};
  const OperandType axes_operand_type(Type::TENSOR_INT32, axes_dimen);
  NEURON_RETURN_IF_ERROR(
      model_builder.AddOperandFromPersistMemoryBuffer(axes_name, axes.data(), axes_operand_type));
  input_indices.push_back(operand_indices.at(axes_name));  // axes

  const int32_t input_rank = static_cast<int32_t>(input_shape.size());

  // Compute output shape.
  std::vector<uint32_t> output_dimen;
  output_dimen.reserve(static_cast<size_t>(input_rank));
  for (int32_t i = 0; i < input_rank; i++) {
    if (std::find(axes.begin(), axes.end(), i) == axes.end()) {
      output_dimen.push_back(input_shape[i]);
    } else if (keepdims) {
      output_dimen.push_back(1);
    }
  }
  // Neuron requires output shape to be {1} if all dimensions are reduced.
  if (output_dimen.empty()) output_dimen.push_back(1);

  shaper.AddShape(output, output_dimen);

  ADD_SCALAR_OPERAND(model_builder, input_indices, keepdims ? 1 : 0);

  const auto& in_type = operand_types.at(input);
  float y_scale = in_type.operandType.scale;
  int32_t y_zero_point = in_type.operandType.zeroPoint;
  if (const auto* qdq = ctx.GetQDQGroup(output);
      qdq && qdq->output_quant_params.has_value()) {
    y_scale = qdq->output_quant_params->scale;
    y_zero_point = qdq->output_quant_params->zero_point;
  }
  const OperandType output_operand_type(in_type.type, output_dimen, y_scale,
                                        y_zero_point);
  NEURON_RETURN_IF_ERROR(model_builder.AddOperation(op_code, input_indices,
                                                     {output}, {output_operand_type}));

  return NeuronStatus::OK();
}
void CreateReductionOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  CreateSharedOpBuilderImpl<ReductionOpBuilder>(
      op_type, op_registrations,
      {
          "ReduceMean",
          "ReduceMax",
          "ReduceMin",
          "ReduceProd",
          "ReduceSum",
      });
}

}  // namespace neuron
}  // namespace onnxruntime
