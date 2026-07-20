// conv_op_builder.cc
// Plugin EP version – uses onnx::NodeProto + NeuronGraphContext.
//
// CHANGES FROM ORIGINAL:
//   node_unit.Inputs()[i].node_arg.Name() → node.input(i)
//   node_unit.Outputs()[i].node_arg.Name() → node.output(i)
//   node_unit.OpType() → node.op_type()
//   node_unit.Name() → node.name()
//   helper.Get("group", 1) → NeuronGraphContext::GetAttrInt(node, "group", 1)
//   helper.Get("dilations", ...) → NeuronGraphContext::GetAttrInts(node, "dilations")
//   helper.Get("strides", ...) → NeuronGraphContext::GetAttrInts(node, "strides")
//   helper.Get("auto_pad", ...) → NeuronGraphContext::GetAttrString(node, "auto_pad", ...)
//   helper.Get("pads", ...) → NeuronGraphContext::GetAttrInts(node, "pads")
//   GetOnnxDataTypeInfoFromNodeUnit → ctx.GetElementType / operand_types.at(name)
//   GetConvType(node_unit, ...) → GetConvType(node, ctx)
//   model_builder.AddOperandFromConstantInitializer(node_unit, ArgType::kInput, idx, ...)
//     → model_builder.AddOperandFromConstantInitializer(node, ctx, idx, ...)
//   Initializer unpacked_xxx → ctx.GetInitializer + GetRawInitializerData
//   FindActivation(node_unit) → FindActivation(node, ctx)
//   ORT_RETURN_IF_ERROR → NEURON_RETURN_IF_ERROR
//   ORT_MAKE_STATUS → NeuronStatus{false, MakeNeuronString(...)}
//   Status → NeuronStatus
//
// Quantized conv (QLinearConv):
//   In the plugin EP, quantized inputs arrive via DequantizeLinear nodes in the
//   graph. Input[0]=input tensor, Input[1]=weight, Input[2]=bias (optional).
//   Quantization scales/ZPs are already encoded in the operand_types by the DQ
//   builder. No quant_param struct from NodeUnit is used here.

#include "base_op_builder.h"

namespace onnxruntime {
namespace neuron {
using namespace onnxruntime::neuron::op_builder_helpers;
using namespace onnxruntime::neuron::wrapper;

class ConvOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder,
                              const onnx::NodeProto& node,
                              const NeuronGraphContext& ctx) const override;

 private:
  NeuronStatus AddToModelBuilderImpl(ModelBuilder& model_builder,
                                      const onnx::NodeProto& node,
                                      const NeuronGraphContext& ctx) const override;


  bool HasSupportedInputOutputsImpl(const onnx::NodeProto& node,
                                     const NeuronGraphContext& ctx,
                                     const OpSupportCheckParams& params) const override;
};

// Add operator related

void ConvOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder,
                                           const onnx::NodeProto& node,
                                           const NeuronGraphContext& ctx) const {
  // For QDQ Conv, node.input(1) is the DQ output (not a direct initializer).
  // Skip the underlying quantized weight initializer instead so that
  // RegisterInitializers doesn't register it with scale=0.
  const std::string& weight_input = node.input(1);
  const auto* weight_dq = ctx.GetProducer(weight_input);
  if (weight_dq && weight_dq->op_type() == "DequantizeLinear" &&
      weight_dq->input_size() > 0 && ctx.GetInitializer(weight_dq->input(0))) {
    model_builder.AddInitializerToSkip(weight_dq->input(0));
  } else {
    model_builder.AddInitializerToSkip(weight_input);
  }
  if (node.input_size() > 2 && !node.input(2).empty()) {
    const std::string& bias_input = node.input(2);
    const auto* bias_dq = ctx.GetProducer(bias_input);
    if (bias_dq && bias_dq->op_type() == "DequantizeLinear" &&
        bias_dq->input_size() > 0 && ctx.GetInitializer(bias_dq->input(0))) {
      // Bias comes through an absorbed DQ.  Skip the underlying INT32
      // initializer so RegisterInitializers does not register it with scale=0.
      // AddQDQOperandAlias will soft-fail (source not found) and the Conv
      // builder registers the bias operand directly with per-channel scales.
      model_builder.AddInitializerToSkip(bias_dq->input(0));
    } else {
      model_builder.AddInitializerToSkip(bias_input);
    }
  }
}

NeuronStatus ConvOpBuilder::AddToModelBuilderImpl(
    ModelBuilder& model_builder,
    const onnx::NodeProto& node,
    const NeuronGraphContext& ctx) const {
  const auto& operand_types(model_builder.GetOperandTypes());
  auto& shaper(model_builder.GetShaper());

  const auto& input  = node.input(0);
  const auto& weight = node.input(1);

  ConvType conv_type = GetConvType(node, ctx);
  const bool is_conv_transpose  = (conv_type == ConvType::ConvTranspose);
  const bool conv_2d            = (conv_type == ConvType::Regular);
  const bool grouped_conv_2d    = (conv_type == ConvType::Grouped);
  const bool depthwise_conv_2d  = (conv_type == ConvType::Depthwise);

  // Determine weight layout
  ModelBuilder::DataLayout weight_layout = ModelBuilder::DataLayout::NO_CHANGE;
  if (is_conv_transpose) {
    // ONNX ConvTranspose weight: [C_in, C_out/group, kH, kW] → [C_out/group, kH, kW, C_in]
    weight_layout = ModelBuilder::DataLayout::L_1230;
  } else {
    // Conv: ONNX weight [O, I/group, kH, kW]
    weight_layout = (conv_2d || grouped_conv_2d) ? ModelBuilder::DataLayout::L_0231
                                                 : ModelBuilder::DataLayout::L_1230;
  }

  NEURON_RETURN_IF_ERROR(
      model_builder.AddOperandFromConstantInitializer(node, ctx, 1,
                                                      /* dequantize_to_float32 */ false,
                                                      weight_layout,
                                                      weight));

  const bool hasBias = (node.input_size() > 2 && !node.input(2).empty());
  const std::string bias = hasBias ? node.input(2) : weight + "_bias";
  const auto& weight_shape = shaper[weight];

  // Helper: compute bias OperandType for a quantized Conv.
  // Neuron SDK requires bias_scale[c] = input_scale * weight_scale[c].
  // For per-channel weight, compute per-channel INT32 scales via the QDQ group's
  // stored weight quant params.  The per-channel scales are then communicated to
  // Neuron via NeuronModel_setOperandSymmPerChannelQuantParams on the INT32 bias
  // operand (triggered inside AddNewNeuronOperand when channelQuant is set).
  // For per-tensor weight, a single scalar scale suffices.
  const auto make_quant_bias_type = [&](const Shape& bias_shape) -> OperandType {
    const float input_scale = operand_types.at(input).operandType.scale;

    // Try to get per-channel weight scales from the QDQ group for this node.
    const auto* qdq = ctx.GetQDQGroup(node.output(0));
    const bool per_ch =
        qdq && qdq->input_quant_params.size() > 1 &&
        qdq->input_quant_params[1].has_value() &&
        qdq->input_quant_params[1]->IsPerChannel() &&
        !qdq->input_quant_params[1]->per_channel_scales.empty();

    if (per_ch) {
      const auto& w_scales = qdq->input_quant_params[1]->per_channel_scales;
      std::vector<float> b_scales(w_scales.size());
      for (size_t c = 0; c < w_scales.size(); ++c)
        b_scales[c] = input_scale * w_scales[c];
      wrapper::OnnxDataTypeInfo info;
      info.onnx_data_type = onnx::TensorProto::INT32;
      info.scales         = b_scales;
      info.zero_points    = std::vector<int32_t>(b_scales.size(), 0);
      info.channel_dim    = 0;  // bias is 1-D; channel axis is 0
      return OperandType(info, bias_shape);
    }

    // Per-tensor: single scale = input_scale * weight_scale.
    const float weight_scale = operand_types.at(weight).operandType.scale;
    return OperandType(Type::TENSOR_INT32, bias_shape, input_scale * weight_scale);
  };

  if (!hasBias) {
    Shape bias_shape;
    if (is_conv_transpose) {
      const int64_t group = NeuronGraphContext::GetAttrInt(node, "group", 1);
      bias_shape = Shape{static_cast<uint32_t>(weight_shape[1] * static_cast<uint32_t>(group))};
    } else {
      bias_shape = (conv_2d || grouped_conv_2d || depthwise_conv_2d)
                       ? Shape{weight_shape[0]}
                       : Shape{weight_shape[3]};
    }

    const auto& weight_type = operand_types.at(weight).type;
    if (weight_type == Type::TENSOR_FLOAT32) {
      std::vector<float> buffer(bias_shape[0], 0.0f);
      const OperandType bias_operand_type(Type::TENSOR_FLOAT32, bias_shape, 0.0f);
      NEURON_RETURN_IF_ERROR(
          model_builder.AddOperandFromPersistMemoryBuffer(bias, buffer.data(), bias_operand_type));
    } else if (weight_type == Type::TENSOR_QUANT8_ASYMM ||
               weight_type == Type::TENSOR_QUANT8_SYMM ||
               weight_type == Type::TENSOR_QUANT8_SYMM_PER_CHANNEL ||
               weight_type == Type::TENSOR_QUANT8_ASYMM_SIGNED ||
               weight_type == Type::TENSOR_QUANT8_ASYMM_PER_CHANNEL ||
               weight_type == Type::TENSOR_QUANT8_ASYMM_SIGNED_PER_CHANNEL) {
      std::vector<int32_t> buffer(bias_shape[0], 0);
      const OperandType bias_operand_type = make_quant_bias_type(bias_shape);
      NEURON_RETURN_IF_ERROR(
          model_builder.AddOperandFromPersistMemoryBuffer(bias, buffer.data(), bias_operand_type));
    } else {
      return NeuronStatus{false, MakeNeuronString("Conv: unknown weight type")};
    }
  } else {
    // Bias is provided.
    //
    // Check first if bias arrives via an absorbed DQ node
    // (e.g., bias_quantized → DequantizeLinear → bias).  In that case the
    // underlying initializer holds raw INT32 data and the DQ node's stored
    // quant scales may not satisfy Neuron's strict bias_scale ≈
    // input_scale * weight_scale check.  We bypass AddOperandFromConstantInitializer
    // and register the bias operand directly using make_quant_bias_type, which
    // always computes the scales as input_scale * weight_scale[c].
    const auto* bias_dq_producer = ctx.GetProducer(bias);
    const bool bias_via_dq =
        (bias_dq_producer != nullptr &&
         bias_dq_producer->op_type() == "DequantizeLinear" &&
         bias_dq_producer->input_size() > 0 &&
         ctx.GetInitializer(bias_dq_producer->input(0)) != nullptr);

    if (bias_via_dq) {
      const auto* initializer = ctx.GetInitializer(bias_dq_producer->input(0));
      NEURON_RETURN_IF_NOT(initializer != nullptr, "Conv: bias initializer not found");

      Shape bias_dimen;
      for (auto d : initializer->dims()) bias_dimen.push_back(static_cast<uint32_t>(d));
      if (bias_dimen.empty()) bias_dimen.push_back(1);

      const void* raw = op_builder_helpers::GetRawInitializerData(*initializer);
      NEURON_RETURN_IF_NOT(raw != nullptr, "Conv: bias initializer has no data");

      const OperandType bias_operand_type = make_quant_bias_type(bias_dimen);
      NEURON_RETURN_IF_ERROR(
          model_builder.AddOperandFromPersistMemoryBuffer(bias, raw, bias_operand_type));
    } else {
      // GetElementType uses value_info which does NOT include initializers in ONNX
      // opset ≥ 9 models (initializers are not required to appear in graph.input()).
      // Fall back to the initializer's own data_type when value_info returns 0.
      int32_t bias_elem_type = ctx.GetElementType(bias);
      if (bias_elem_type == 0) {
        const auto* init = ctx.GetInitializer(bias);
        if (init) bias_elem_type = init->data_type();
      }
      if (bias_elem_type == onnx::TensorProto::INT32) {
        const auto* initializer = ctx.GetInitializer(bias);
        NEURON_RETURN_IF_NOT(initializer != nullptr, "Conv: bias initializer not found");

        Shape bias_dimen;
        for (auto d : initializer->dims()) bias_dimen.push_back(static_cast<uint32_t>(d));
        if (bias_dimen.empty()) bias_dimen.push_back(1);

        const void* raw = op_builder_helpers::GetRawInitializerData(*initializer);
        NEURON_RETURN_IF_NOT(raw != nullptr, "Conv: bias initializer has no data");

        const OperandType bias_operand_type = make_quant_bias_type(bias_dimen);
        NEURON_RETURN_IF_ERROR(
            model_builder.AddOperandFromPersistMemoryBuffer(bias, raw, bias_operand_type));
      } else {
        NEURON_RETURN_IF_ERROR(
            model_builder.AddOperandFromConstantInitializer(node, ctx, 2,
                                                            /* dequantize_to_float32 */ false,
                                                            ModelBuilder::DataLayout::NO_CHANGE,
                                                            bias));
      }
    }
  }

  // Attributes
  const int64_t group = NeuronGraphContext::GetAttrInt(node, "group", 1);

  auto onnx_dilations_raw = NeuronGraphContext::GetAttrInts(node, "dilations");
  std::vector<int32_t> onnx_dilations =
      onnx_dilations_raw.size() >= 2
          ? std::vector<int32_t>{static_cast<int32_t>(onnx_dilations_raw[0]),
                                  static_cast<int32_t>(onnx_dilations_raw[1])}
          : std::vector<int32_t>{1, 1};

  auto onnx_strides_raw = NeuronGraphContext::GetAttrInts(node, "strides");
  std::vector<int32_t> onnx_strides =
      onnx_strides_raw.size() >= 2
          ? std::vector<int32_t>{static_cast<int32_t>(onnx_strides_raw[0]),
                                  static_cast<int32_t>(onnx_strides_raw[1])}
          : std::vector<int32_t>{1, 1};

  auto onnx_pads_raw = NeuronGraphContext::GetAttrInts(node, "pads");
  std::vector<int32_t> onnx_pads =
      onnx_pads_raw.size() >= 4
          ? std::vector<int32_t>{static_cast<int32_t>(onnx_pads_raw[0]),
                                  static_cast<int32_t>(onnx_pads_raw[1]),
                                  static_cast<int32_t>(onnx_pads_raw[2]),
                                  static_cast<int32_t>(onnx_pads_raw[3])}
          : std::vector<int32_t>{0, 0, 0, 0};

  const auto auto_pad_type =
      StringToAutoPadType(NeuronGraphContext::GetAttrString(node, "auto_pad", "NOTSET"));

  bool use_auto_pad = false;
  int32_t neuron_padding_code = NEURON_PADDING_SAME;
  const bool use_nchw = model_builder.UseNCHW();
  const auto input_shape = shaper[input];

  // Weight after layout transposition: shaper[weight] holds the new dims.
  const auto& weight_shape_ref = shaper[weight];
  const auto weight_size_y = weight_shape_ref[1];
  const auto weight_size_x = weight_shape_ref[2];

  NEURON_RETURN_IF_ERROR(
      HandleAutoPad(input_shape, weight_size_y, weight_size_x,
                    onnx_strides, onnx_dilations,
                    auto_pad_type, use_nchw,
                    onnx_pads, neuron_padding_code, use_auto_pad));

  const auto& operand_indices(model_builder.GetOperandIndices());
  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  input_indices.push_back(operand_indices.at(weight));
  input_indices.push_back(operand_indices.at(bias));

  if (use_auto_pad && !is_conv_transpose) {
    ADD_SCALAR_OPERAND(model_builder, input_indices, neuron_padding_code);
  } else {
    // ONNX pads: [top, left, bottom, right]
    // Neuron expects: left, right, top, bottom
    ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_pads[1]);
    ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_pads[3]);
    ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_pads[0]);
    ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_pads[2]);
  }

  // Strides: width then height
  ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_strides[1]);
  ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_strides[0]);

  if (!conv_2d && !is_conv_transpose) {
    if (depthwise_conv_2d) {
      const int32_t depthwiseMultiplier =
          static_cast<int32_t>(shaper[weight][3]) / static_cast<int32_t>(group);
      ADD_SCALAR_OPERAND(model_builder, input_indices, depthwiseMultiplier);
    } else {  // grouped_conv_2d
      ADD_SCALAR_OPERAND(model_builder, input_indices, static_cast<int32_t>(group));
    }
  }

  int32_t fuse_code = model_builder.FindActivation(node, ctx);
  ADD_SCALAR_OPERAND(model_builder, input_indices, fuse_code);
  ADD_SCALAR_OPERAND(model_builder, input_indices, use_nchw);

  // Neuron Grouped Conv does not support dilations
  if (!grouped_conv_2d && !is_conv_transpose) {
    ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_dilations[1]);
    ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_dilations[0]);
  }

  const auto& output = node.output(0);
  int32_t operationCode = NEURON_CONV_2D;
  if (is_conv_transpose)
    operationCode = NEURON_TRANSPOSE_CONV_2D;
  else if (grouped_conv_2d)
    operationCode = NEURON_GROUPED_CONV_2D;
  else if (depthwise_conv_2d)
    operationCode = NEURON_DEPTHWISE_CONV_2D;

  // Determine output operand type.
  // For QDQ Conv, the output is a quantized tensor whose scale and zero-point
  // come from the Q node that wraps this Conv's output.  NeuronGraphContext
  // has already detected this group and stored it in ctx.qdq_groups.
  OperandType output_operand_type(operand_types.at(input).type, shaper[output]);
  if (const auto* qdq = ctx.GetQDQGroup(output)) {
    if (qdq->output_quant_params.has_value()) {
      const auto& qp = *qdq->output_quant_params;
      if (qp.IsPerChannel()) {
        // Per-channel quantized output.
        output_operand_type = OperandType(
            qp.IsSigned() ? Type::TENSOR_QUANT8_SYMM_PER_CHANNEL
                           : Type::TENSOR_QUANT8_ASYMM_PER_CHANNEL,
            shaper[output], 0.0f /*scale ignored*/, 0 /*zp ignored*/);
        // Per-channel params set separately via NeuronModel_setOperandSymmPerChannelQuantParams.
      } else {
        output_operand_type = OperandType(
            qp.IsSigned() ? Type::TENSOR_QUANT8_ASYMM_SIGNED
                           : Type::TENSOR_QUANT8_ASYMM,
            shaper[output], qp.scale, qp.zero_point);
      }
    }
  }

  NEURON_RETURN_IF_ERROR(model_builder.AddOperation(operationCode, input_indices,
                                                    {output}, {output_operand_type}));
  return NeuronStatus::OK();
}

// Operator support related
bool ConvOpBuilder::HasSupportedInputOutputsImpl(
    const onnx::NodeProto& node,
    const NeuronGraphContext& ctx,
    const OpSupportCheckParams& params) const {
  // Check input[0] (the tensor input) supports float or quantized types.
  return IsTypeSupported(node, ctx, 0, /*is_input*/ true,
                         {onnx::TensorProto::FLOAT,
                          onnx::TensorProto::FLOAT16,
                          onnx::TensorProto::INT8,
                          onnx::TensorProto::UINT8});
}

void CreateConvOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  CreateSharedOpBuilderImpl<ConvOpBuilder>(
      op_type, op_registrations,
      {
          "Conv",
          "QLinearConv",
          "ConvTranspose",
      });
}

}  // namespace neuron
}  // namespace onnxruntime
