// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "op_builder_helpers.h"

#include <algorithm>
#include <functional>
#include <gsl/span>
#include <optional>
#include <utility>
#include <numeric>

#include "../neuron_types_internal.h"
#include "helper.h"
#include <onnx/onnx_pb.h>

namespace onnxruntime::neuron::op_builder_helpers {

using onnxruntime::neuron::wrapper::OperandType, onnxruntime::neuron::wrapper::Type;

// ---------------------------------------------------------------------------
// ComputePad — compute head/tail padding for one spatial dimension
// ---------------------------------------------------------------------------
NeuronStatus ComputePad(int32_t input_size, int32_t stride, uint32_t weight_size,
                         int32_t dilation, AutoPadType auto_pad,
                         int64_t& pad_head, int64_t& pad_tail) {
  if (auto_pad == AutoPadType::NOTSET) return NeuronStatus::OK();
  int32_t eff_kernel = (static_cast<int32_t>(weight_size) - 1) * dilation + 1;
  int64_t total_pad = std::max<int64_t>(
      0, (int64_t)(((input_size + stride - 1) / stride - 1) * stride + eff_kernel - input_size));
  if (auto_pad == AutoPadType::VALID) {
    pad_head = 0; pad_tail = 0;
  } else if (auto_pad == AutoPadType::SAME_LOWER) {
    pad_head = (total_pad + 1) / 2;
    pad_tail = total_pad / 2;
  } else {  // SAME_UPPER
    pad_head = total_pad / 2;
    pad_tail = (total_pad + 1) / 2;
  }
  return NeuronStatus::OK();
}

NeuronStatus AddNeuronTranspose(ModelBuilder& model_builder,
                          const std::string& data_input,
                          const std::string& perm_input,
                          const gsl::span<const int32_t> perm,
                          const std::string& output) {
  auto& shaper(model_builder.GetShaper());

  // Calculate transpose output shape

  const Shape input_dimen = shaper[data_input];
  ORT_RETURN_IF_NOT(perm.size() == input_dimen.size(), "Invalid perm is given!");
  size_t size = input_dimen.size();
  Shape output_dimen(size);
  for (size_t i = 0; i < size; i++)
    output_dimen[i] = input_dimen[perm[i]];

  shaper.AddShape(output, output_dimen);

  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(data_input));  // input

  Shape perm_dimen = {static_cast<uint32_t>(perm.size())};
  OperandType perm_operand_type(Type::TENSOR_INT32, perm_dimen);
  ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(perm_input, perm.data(), perm_operand_type));
  uint32_t perm_idx = operand_indices.at(perm_input);
  input_indices.push_back(perm_idx);  // permutation

  OperandType output_operand_type = operand_types.at(data_input);
  output_operand_type.SetDimensions(output_dimen);
  return model_builder.AddOperation(NEURON_TRANSPOSE, input_indices, {output},
                                    {output_operand_type});
}

NeuronStatus AddNeuronReshape(ModelBuilder& model_builder,
                        const std::string& data_input,
                        const std::string& shape_input,
                        const std::vector<int32_t>& shape_value,
                        const std::string& output) {
  auto& shaper = model_builder.GetShaper();
  // Calculate reshape output shape

  const Shape input_dimen = shaper[data_input];
  uint32_t input_size = ShapeSize(input_dimen);
  Shape output_dimen(shape_value.size());

  int64_t capacity = 1;
  int unk_dim_idx = -1;
  for (size_t i = 0; i < shape_value.size(); i++) {
    int32_t dim_i = shape_value[i];
    ORT_RETURN_IF_NOT(dim_i != 0, "Neuron does not support 0 reshape dimension");
    if (dim_i == -1) {
      ORT_RETURN_IF_NOT(unk_dim_idx == -1, "Only one input dimension of Attr(shape) can be unknown!");
      unk_dim_idx = static_cast<int>(i);
    } else {
      capacity *= dim_i;
      output_dimen[i] = static_cast<uint32_t>(dim_i);
    }
  }

  if (unk_dim_idx != -1) {
    if (input_size == 0)
      output_dimen[unk_dim_idx] = 0;
    else
      output_dimen[unk_dim_idx] = static_cast<uint32_t>(input_size / capacity);

    capacity *= output_dimen[unk_dim_idx];
  }

  ORT_RETURN_IF_NOT(capacity == input_size, "Invalid shape is given!");

  shaper.AddShape(output, output_dimen);

  const auto& operand_indices = model_builder.GetOperandIndices();
  const auto& operand_types = model_builder.GetOperandTypes();

  // Add input
  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(data_input));

  // Add new shape
  const Shape shape_dimen{static_cast<uint32_t>(shape_value.size())};
  const OperandType shape_operand_type{Type::TENSOR_INT32, shape_dimen};
  ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(shape_input, shape_value.data(),
                                                                      shape_operand_type));
  input_indices.push_back(operand_indices.at(shape_input));

  // For reshape, the output type should be the same as the input type except the shape is different
  OperandType output_operand_type{operand_types.at(data_input)};
  output_operand_type.SetDimensions(output_dimen);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(NEURON_RESHAPE,
                                                 input_indices, {output}, {output_operand_type}));

  return NeuronStatus::OK();
}

NeuronStatus AddNeuronSplit(ModelBuilder& model_builder,
                      const std::string& input,
                      int32_t axis,
                      const std::vector<std::string>& outputs) {
  const auto& operand_indices = model_builder.GetOperandIndices();
  const auto& operand_types = model_builder.GetOperandTypes();
  auto& shaper = model_builder.GetShaper();

  const auto input_rank = shaper[input].size();
  axis = static_cast<int32_t>(HandleNegativeAxis(axis, input_rank));

  const auto count = static_cast<int32_t>(outputs.size());

  // Calculate split output shape
  {
    const auto input_shape = shaper[input];
    ORT_RETURN_IF_NOT(input_shape[axis] % count == 0,
                      "count [", count, "] does not evenly divide dimension ", axis, " [", input_shape[axis], "]");

    Shape output_shape = input_shape;
    output_shape[axis] = static_cast<uint32_t>(input_shape[axis] / count);
    for (const auto& output_name : outputs) {
      shaper.AddShape(output_name, output_shape);
    }
  }

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  ORT_RETURN_IF_ERROR(AddScalarOperand(model_builder, input_indices, axis));
  ORT_RETURN_IF_ERROR(AddScalarOperand(model_builder, input_indices, count));

  const OperandType& input_operand_type = operand_types.at(input);
  std::vector<OperandType> output_operand_types;
  output_operand_types.reserve(count);
  std::transform(outputs.begin(), outputs.end(), std::back_inserter(output_operand_types),
                 [&](const std::string& output) {
                   OperandType output_operand_type = input_operand_type;
                   output_operand_type.SetDimensions(shaper[output]);
                   return output_operand_type;
                 });

  ORT_RETURN_IF_ERROR(model_builder.AddOperation(NEURON_SPLIT,
                                                 input_indices, outputs, output_operand_types));

  return NeuronStatus::OK();
}

NeuronStatus AddNeuronUnevenSplit(ModelBuilder& model_builder,
                                  const std::string& input,
                                  int32_t axis,
                                  const std::vector<int64_t>& split_sizes,
                                  const std::vector<std::string>& outputs) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  const Shape input_shape = shaper[input];
  const auto rank = input_shape.size();
  axis = static_cast<int32_t>(HandleNegativeAxis(axis, rank));

  ORT_RETURN_IF_NOT(split_sizes.size() == outputs.size(),
                    "Split: ", split_sizes.size(), " sizes for ", outputs.size(), " outputs");

  const OperandType& input_operand_type = operand_types.at(input);
  int32_t offset = 0;
  for (size_t i = 0; i < outputs.size(); ++i) {
    const auto size_at_axis = static_cast<int32_t>(split_sizes[i]);

    std::vector<int32_t> begin(rank, 0), sizes(rank);
    for (size_t d = 0; d < rank; ++d) sizes[d] = static_cast<int32_t>(input_shape[d]);
    begin[axis] = offset;
    sizes[axis] = size_at_axis;

    Shape output_shape = input_shape;
    output_shape[axis] = static_cast<uint32_t>(size_at_axis);
    shaper.AddShape(outputs[i], output_shape);

    std::vector<uint32_t> input_indices{operand_indices.at(input)};
    const Shape param_dimen{static_cast<uint32_t>(rank)};
    const std::string begin_name = model_builder.GetUniqueName(outputs[i] + "/begin");
    const std::string sizes_name = model_builder.GetUniqueName(outputs[i] + "/sizes");
    ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(
        begin_name, begin.data(), OperandType(Type::TENSOR_INT32, param_dimen)));
    ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(
        sizes_name, sizes.data(), OperandType(Type::TENSOR_INT32, param_dimen)));
    input_indices.push_back(operand_indices.at(begin_name));
    input_indices.push_back(operand_indices.at(sizes_name));

    OperandType output_operand_type = input_operand_type;
    output_operand_type.SetDimensions(output_shape);
    ORT_RETURN_IF_ERROR(model_builder.AddOperation(NEURON_SLICE, input_indices,
                                                   {outputs[i]}, {output_operand_type}));
    offset += size_at_axis;
  }
  return NeuronStatus::OK();
}

NeuronStatus AddNeuronBatchNormalization(ModelBuilder& model_builder,
                                   const std::string& input1,
                                   const std::string& input2,
                                   const std::string& input3,
                                   const std::string& output1,
                                   const std::string& output2,
                                   int32_t fuse_code,
                                   float output_scale,
                                   int32_t output_zero_point) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  // Add Neuron Mul
  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input1));
  input_indices.push_back(operand_indices.at(input2));

  ADD_SCALAR_OPERAND(model_builder, input_indices, NEURON_FUSED_NONE);

  const Shape& shape1 = shaper[input1];
  const Shape& shape2 = shaper[input2];
  Shape output1_shape;
  // broadcasting support for eltwise shape operation
  ORT_RETURN_IF_ERROR(op_builder_helpers::PerformBroadcasting(shape1, shape2, output1_shape));
  const OperandType output_operand_type(operand_types.at(input1).type, output1_shape,
                                        output_scale, output_zero_point);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(NEURON_MUL, input_indices,
                                                 {output1}, {output_operand_type}));

  // Add Neuron Add
  input_indices.clear();
  input_indices.push_back(operand_indices.at(output1));
  input_indices.push_back(operand_indices.at(input3));

  ADD_SCALAR_OPERAND(model_builder, input_indices, fuse_code);

  const Shape& shape3 = shaper[input3];
  Shape output2_shape;
  ORT_RETURN_IF_ERROR(op_builder_helpers::PerformBroadcasting(output1_shape, shape3, output2_shape));
  const OperandType output_operand_type2(operand_types.at(input3).type, output2_shape,
                                         output_scale, output_zero_point);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(NEURON_ADD, input_indices,
                                                 {output2}, {output_operand_type2}));
  return NeuronStatus::OK();
}

bool IsSupportedBatchMatMul(const onnx::NodeProto& node, const NeuronGraphContext& /*ctx*/,
                            const NeuronRuntimeVersion& version) {
  // Currently, batch MatMul is composed of various operations including NEURON_SPLIT which requires
  if (!GreaterEqualVersion(version, {4, 1, 0})) {
    LOGS_NEURON(VERBOSE) << "Minimum Neuron feature level required: 4.1.0"
                         << ", actual: " << GetNeuronVersionDescription(version);
    return false;
  }

  // Only support non-QDQ MatMul for now.
  if (node.op_type() != "MatMul") {
    LOGS_NEURON(VERBOSE) << "Unsupported op type: " << node.op_type();
    return false;
  }

  return true;
}

NeuronStatus BuildNormalBatchMatMul(ModelBuilder& model_builder, const onnx::NodeProto& node,
                                    const NeuronGraphContext& ctx) {
  // Build a normal NEURON_BATCH_MATMUL operation
  const std::string& a = node.input(0);
  std::string b = node.input(1);
  const std::string& output = node.output(0);
  Shape a_shape;
  ORT_RETURN_IF_NOT(GetShape(ctx, a, a_shape), "Failed to get A's shape.");

  Shape b_shape;
  ORT_RETURN_IF_NOT(GetShape(ctx, b, b_shape), "Failed to get B's shape.");

  const auto& operand_indices = model_builder.GetOperandIndices();

  if (a_shape.size() == 3 && a_shape[0] == 1 && b_shape.size() == 2) {
    // Reshape B from 2d to [1, b1, b2] to support BatMatMul
    const std::string new_b = model_builder.GetUniqueName(b + "/expanded");
    ORT_RETURN_IF_ERROR(model_builder.AddExpandDims(b, 0, new_b));
    b = new_b; // Replace the old b with this expaned tensor;
    LOGS_NEURON(VERBOSE) << "Added ExpandDims for MATMUL b";
  }

  // For a QDQ MatMul (DQ→MatMul→Q) the output operand must carry the Q node's
  // quantization params.  Leaving scale=0 (the default) makes the MDLA compiler
  // reject a downstream Add with "Act1/Act2 shift exceeds range", because the
  // zero scale propagates through the following Transpose/Reshape operands.
  float y_scale = 0.0f;
  int32_t y_zero_point = 0;
  if (const auto* qdq = ctx.GetQDQGroup(output);
      qdq && qdq->output_quant_params.has_value()) {
    y_scale      = qdq->output_quant_params->scale;
    y_zero_point = qdq->output_quant_params->zero_point;
  }

  const OperandType output_operand_type(model_builder.GetOperandTypes().at(a).type,
                                        model_builder.GetShaper()[output],
                                        y_scale, y_zero_point);
  std::vector<uint32_t> input_indices = {operand_indices.at(a), operand_indices.at(b)};
  const bool adb_x = false;
  const bool adb_y = false;
  ORT_RETURN_IF_ERROR(AddScalarOperand(model_builder, input_indices, adb_x));
  ORT_RETURN_IF_ERROR(AddScalarOperand(model_builder, input_indices, adb_y));
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(NEURON_BATCH_MATMUL,
                                                 input_indices, {output}, {output_operand_type}));
  return NeuronStatus::OK();
}

NeuronStatus BuildBatchMatMul(ModelBuilder& model_builder, const onnx::NodeProto& node,
                              const NeuronGraphContext& ctx) {
  if (auto version = model_builder.GetNeuronVersion(); GreaterEqualVersion(version, {5, 1, 2})) {
    LOGS_NEURON(VERBOSE) << "Use NEURON_BATCH_MATMUL for MATMUL operation.";
    return BuildNormalBatchMatMul(model_builder, node, ctx);
  }
  // For version < 5.1.2, we will implement batch MatMul by composing Neuron operations

  const std::string& a = node.input(0);
  std::string b = node.input(1);

  Shape a_shape;
  ORT_RETURN_IF_NOT(GetShape(ctx, a, a_shape), "Failed to get A's shape.");

  Shape b_shape;
  ORT_RETURN_IF_NOT(GetShape(ctx, b, b_shape), "Failed to get B's shape.");

  if (a_shape.size() == 3 && a_shape[0] == 1 && b_shape.size() == 2) {
    // Reshape B from 2d to [1, b1, b2] to support BatMatMul
    std::string new_b;
    ORT_RETURN_IF_ERROR(model_builder.AddExpandDims(b, 0, new_b));
    b = new_b;
    b_shape.insert(b_shape.begin(), 1);
    model_builder.GetShaper().AddShape(new_b, b_shape);
    LOGS_NEURON(VERBOSE) << "Added ExpandDims for MATMUL b";
  }

  const std::string& output = node.output(0);

  std::vector<std::string> gemm_a_inputs{a};
  std::vector<std::string> gemm_b_inputs{b};

  const auto m = a_shape[a_shape.size() - 2],
             k = a_shape[a_shape.size() - 1],
             n = b_shape[b_shape.size() - 1];

  const bool reshape_leading_dimensions = a_shape.size() > 3;
  const auto batch_size = ShapeSize(a_shape, 0, a_shape.size() - 2);

  auto add_reshape = [&model_builder](const std::string& input, const Shape& new_shape,
                                      const std::string& output) -> NeuronStatus {
    const std::string new_shape_name = model_builder.GetUniqueName(input + "/new_shape");
    std::vector<int32_t> new_shape_i32{};
    new_shape_i32.reserve(new_shape.size());
    std::transform(new_shape.begin(), new_shape.end(), std::back_inserter(new_shape_i32),
                   [](uint32_t d) { return static_cast<int32_t>(d); });
    ORT_RETURN_IF_ERROR(AddNeuronReshape(model_builder, input, new_shape_name, new_shape_i32, output));
    return NeuronStatus::OK();
  };

  auto add_reshape_generate_output = [&model_builder, &add_reshape](const std::string& input, const Shape& new_shape,
                                                                    std::string& output) -> NeuronStatus {
    std::string reshaped = model_builder.GetUniqueName(input + "/reshaped");
    ORT_RETURN_IF_ERROR(add_reshape(input, new_shape, reshaped));
    output = std::move(reshaped);
    return NeuronStatus::OK();
  };

  // collapse leading dimensions to a single one
  if (reshape_leading_dimensions) {
    const Shape a_new_shape_value = {batch_size, m, k},
                b_new_shape_value = {batch_size, k, n};
    std::string a_reshaped, b_reshaped;

    ORT_RETURN_IF_ERROR(add_reshape_generate_output(gemm_a_inputs.front(), a_new_shape_value, a_reshaped));
    gemm_a_inputs.front() = a_reshaped;

    ORT_RETURN_IF_ERROR(add_reshape_generate_output(gemm_b_inputs.front(), b_new_shape_value, b_reshaped));
    gemm_b_inputs.front() = b_reshaped;
  }

  // transpose B
  {
    const std::string b_new_perm = model_builder.GetUniqueName(b + "/new_perm"),
                      b_transposed = model_builder.GetUniqueName(b + "/transposed");
    ORT_RETURN_IF_ERROR(AddNeuronTranspose(model_builder, gemm_b_inputs.front(), b_new_perm,
                                           AsSpan<int32_t>({0, 2, 1}), b_transposed));
    gemm_b_inputs.front() = b_transposed;
  }

  // split batch
  {
    auto add_split = [&model_builder, batch_size](const std::string& input,
                                                  std::vector<std::string>& outputs_result) -> NeuronStatus {
      std::vector<std::string> outputs;
      outputs.reserve(batch_size);
      for (size_t i = 0; i < batch_size; ++i) {
        outputs.push_back(model_builder.GetUniqueName(MakeNeuronString(input, "/split_", i)));
      }
      ORT_RETURN_IF_ERROR(AddNeuronSplit(model_builder, input, 0, outputs));
      outputs_result = std::move(outputs);
      return NeuronStatus::OK();
    };

    std::vector<std::string> a_split_outputs;
    ORT_RETURN_IF_ERROR(add_split(gemm_a_inputs.front(), a_split_outputs));
    gemm_a_inputs = std::move(a_split_outputs);

    std::vector<std::string> b_split_outputs;
    ORT_RETURN_IF_ERROR(add_split(gemm_b_inputs.front(), b_split_outputs));
    gemm_b_inputs = std::move(b_split_outputs);
  }

  // GEMM per matrix pair
  std::vector<std::string> gemm_outputs;
  gemm_outputs.reserve(batch_size);
  {
    const std::string bias = model_builder.GetUniqueName(node.name() + "/zero_bias");
    {
      if (model_builder.GetOperandTypes().at(b).type != Type::TENSOR_FLOAT32) {
        ORT_NOT_IMPLEMENTED("Only float input is supported now.");
      }
      const Shape bias_shape{n};
      const std::vector<float> buffer(n, 0.0f);
      const OperandType bias_operand_type(Type::TENSOR_FLOAT32, bias_shape);
      ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(bias, buffer.data(), bias_operand_type));
    }

    auto add_fc = [&model_builder, &bias](const std::string& a, const std::string& b_transposed,
                                          const std::string& output) -> NeuronStatus {
      const auto& operand_indices = model_builder.GetOperandIndices();
      const auto& operand_types = model_builder.GetOperandTypes();
      auto& shaper = model_builder.GetShaper();
      std::vector<uint32_t> input_indices;
      input_indices.push_back(operand_indices.at(a));             // A
      input_indices.push_back(operand_indices.at(b_transposed));  // B'
      input_indices.push_back(operand_indices.at(bias));          // C
      int32_t fuse_code = NEURON_FUSED_NONE;
      ORT_RETURN_IF_ERROR(AddScalarOperand(model_builder, input_indices, fuse_code));

      const auto a_dimen = shaper[a];
      const auto b_transposed_dimen = shaper[b_transposed];  // num_units, input_size
      Shape output_dimen{a_dimen[0], b_transposed_dimen[0]};
      shaper.AddShape(output, output_dimen);
      const OperandType output_operand_type(operand_types.at(a).type, output_dimen);
      ORT_RETURN_IF_ERROR(model_builder.AddOperation(NEURON_FULLY_CONNECTED, input_indices,
                                                     {output}, {output_operand_type}));
      return NeuronStatus::OK();
    };

    for (uint32_t i = 0; i < batch_size; ++i) {
      const auto &gemm_a_input = gemm_a_inputs[i],
                 &gemm_b_input = gemm_b_inputs[i];

      // make inputs 2D ([1, x, y] -> [x, y])
      std::string a_2d, b_transposed_2d;
      ORT_RETURN_IF_ERROR(add_reshape_generate_output(gemm_a_input, Shape{m, k}, a_2d));
      ORT_RETURN_IF_ERROR(add_reshape_generate_output(gemm_b_input, Shape{n, k}, b_transposed_2d));

      const std::string gemm_output = model_builder.GetUniqueName(MakeNeuronString(node.name(), "/gemm_", i));
      ORT_RETURN_IF_ERROR(add_fc(a_2d, b_transposed_2d, gemm_output));

      // reshape output for concatenation ([x, y] -> [1, x, y])
      std::string gemm_output_3d;
      ORT_RETURN_IF_ERROR(add_reshape_generate_output(gemm_output, Shape{1, m, n}, gemm_output_3d));

      gemm_outputs.push_back(gemm_output_3d);
    }
  }

  // concat batch
  const std::string joined_gemm_output =
      reshape_leading_dimensions ? model_builder.GetUniqueName(node.name() + "/joined_gemm_output") : output;
  {
    auto add_concat = [&model_builder](const std::vector<std::string>& inputs,
                                       const std::string& output) -> NeuronStatus {
      const auto& operand_indices = model_builder.GetOperandIndices();
      const auto& operand_types = model_builder.GetOperandTypes();
      auto& shaper = model_builder.GetShaper();
      std::vector<uint32_t> input_indices;
      input_indices.reserve(inputs.size() + 1);
      std::transform(inputs.begin(), inputs.end(), std::back_inserter(input_indices),
                     [&operand_indices](const std::string& input) { return operand_indices.at(input); });
      const int32_t axis = 0;
      ORT_RETURN_IF_ERROR(AddScalarOperand(model_builder, input_indices, axis));

      // Calculate_concat_output_shape
      {
        std::vector<Shape> dimens;
        for (const auto& input_name : inputs) {
          const Shape dimen = shaper[input_name];
          dimens.push_back(dimen);
        }

        // If one of the inputs has dynamic shape (at axis), we will keep the dimen[axis] as 0 (dynamic)
        auto output_dimen = dimens[0];
        if (output_dimen[axis] != 0) {
          for (size_t i = 1; i < dimens.size(); i++) {
            if (dimens[i][axis] == 0) {
              output_dimen[axis] = 0;
              break;
            }
            output_dimen[axis] += dimens[i][axis];
          }
        }
        shaper.AddShape(output, output_dimen);
      }

      OperandType output_operand_type = operand_types.at(inputs[0]);
      output_operand_type.SetDimensions(shaper[output]);
      ORT_RETURN_IF_ERROR(model_builder.AddOperation(NEURON_CONCATENATION,
                                                     input_indices, {output}, {output_operand_type}));
      return NeuronStatus::OK();
    };

    ORT_RETURN_IF_ERROR(add_concat(gemm_outputs, joined_gemm_output));
  }

  // reshape to original dimensions
  if (reshape_leading_dimensions) {
    Shape new_shape = a_shape;
    new_shape[new_shape.size() - 2] = m;
    new_shape[new_shape.size() - 1] = n;
    ORT_RETURN_IF_ERROR(add_reshape(joined_gemm_output, new_shape, output));
  }

  return NeuronStatus::OK();
}

NeuronStatus AddInitializerInNewLayout(ModelBuilder& model_builder,
                                 const std::string& name,
                                 const OperandType& source_operand_type,
                                 ModelBuilder::DataLayout new_layout,
                                 bool is_per_tensor_u8s8) {
  const auto& tensor = *model_builder.GetInitializerTensors().at(name);
  const Shape& shape = source_operand_type.dimensions;
  ORT_RETURN_IF_NOT(shape.size() == 4,
                    "The initializer is not 4D: ", name, " actual dim ", shape.size());

  // TODO support other data types
  const uint8_t* src = nullptr;

  switch (tensor.data_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
    case ONNX_NAMESPACE::TensorProto_DataType_INT8: {
      break;
    }
    default:
      return NeuronStatus(NeuronStatusCode::INVALID_ARGUMENT,
                          MakeNeuronString("The initializer of graph ", name,
                                           " doesn't have valid type: ", tensor.data_type()));
  }
  src = reinterpret_cast<const uint8_t*>(GetRawInitializerData(tensor));
  const auto out_t = shape[0], in_t = shape[1],
             h_t = shape[2], w_t = shape[3];
  Shape dest_shape;
  if (new_layout == ModelBuilder::DataLayout::L_0231)
    dest_shape = {out_t, h_t, w_t, in_t};  // L_0231
  else
    dest_shape = {in_t, h_t, w_t, out_t};  // L_1230 for depthwise conv weight

  OperandType operand_type = source_operand_type;
  operand_type.SetDimensions(dest_shape);
  std::unique_ptr<uint8_t[]> buffer_holder(new uint8_t[operand_type.GetOperandBlobByteSize()]);
  uint8_t* buffer = buffer_holder.get();
  size_t element_size = operand_type.GetElementByteSize();

  uint8_t bit_flip_val = is_per_tensor_u8s8 ? 0x80 : 0;
  for (uint32_t out = 0; out < out_t; out++) {
    for (uint32_t in = 0; in < in_t; in++) {
      for (uint32_t h = 0; h < h_t; h++) {
        for (uint32_t w = 0; w < w_t; w++) {
          auto onnx_idx = out * in_t * h_t * w_t +
                          in * h_t * w_t +
                          h * w_t +
                          w;

          uint32_t Neuron_idx;
          if (new_layout == ModelBuilder::DataLayout::L_0231) {  // L_0231
            Neuron_idx = out * h_t * w_t * in_t +
                         h * w_t * in_t +
                         w * in_t +
                         in;
          } else {  // L_1230 for depthwise conv weight
            Neuron_idx = in * h_t * w_t * out_t +
                         h * w_t * out_t +
                         w * out_t +
                         out;
          }

          for (size_t i = 0; i < element_size; i++) {
            buffer[element_size * Neuron_idx + i] = src[element_size * onnx_idx + i] ^ bit_flip_val;
          }
        }
      }
    }
  }

  return model_builder.AddOperandFromPersistMemoryBuffer(name, &buffer[0], operand_type);
}

NeuronStatus AddInitializerTransposed(ModelBuilder& model_builder,
                                const OperandType& source_operand_type,
                                const std::string& name,
                                bool is_per_tensor_u8s8) {
  const auto& tensor = *model_builder.GetInitializerTensors().at(name);
  const Shape& shape = source_operand_type.dimensions;

  ORT_RETURN_IF_NOT(shape.size() == 2,
                    "The initializer is not 2D: ", name, " actual dim ", shape.size());

  // TODO support other data types
  const uint8_t* src = nullptr;
  switch (tensor.data_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
    case ONNX_NAMESPACE::TensorProto_DataType_INT8: {
      break;
    }
    default:
      return NeuronStatus(NeuronStatusCode::INVALID_ARGUMENT,
                          MakeNeuronString("The initializer of graph ", name,
                                           " doesn't have valid type: ", tensor.data_type()));
  }
  src = reinterpret_cast<const uint8_t*>(GetRawInitializerData(tensor));
  // could be float/u8/s8, so we have to use raw data here.
  const auto x_t = shape[0], y_t = shape[1];
  Shape dest_shape = {y_t, x_t};
  OperandType operand_type = source_operand_type;
  operand_type.SetDimensions(dest_shape);
  std::unique_ptr<uint8_t[]> buffer_holder(new uint8_t[operand_type.GetOperandBlobByteSize()]);
  uint8_t* buffer = buffer_holder.get();
  size_t element_size = operand_type.GetElementByteSize();
  uint8_t bit_flip_val = is_per_tensor_u8s8 ? 0x80 : 0;
  for (uint32_t x = 0; x < x_t; x++) {
    for (uint32_t y = 0; y < y_t; y++) {
      for (size_t i = 0; i < element_size; i++) {
        buffer[element_size * (y * x_t + x) + i] = src[element_size * (x * y_t + y) + i] ^ bit_flip_val;
      }
    }
  }

  return model_builder.AddOperandFromPersistMemoryBuffer(name, &buffer[0], operand_type);
}

NeuronStatus ComputeConvPads(
    const Shape& input_dimen,
    const uint32_t weight_size_y, const uint32_t weight_size_x,
    const std::vector<int32_t>& onnx_pads, const std::vector<int32_t>& onnx_strides, const std::vector<int32_t>& onnx_dilations,
    AutoPadType auto_pad_type, bool nchw,
    std::vector<int32_t>& pads_out) {
  const int32_t input_size_y = nchw ? input_dimen[2] : input_dimen[1];
  const int32_t input_size_x = nchw ? input_dimen[3] : input_dimen[2];
  const int32_t stride_y = onnx_strides[0];
  const int32_t stride_x = onnx_strides[1];
  const int32_t dilation_y = onnx_dilations[0];
  const int32_t dilation_x = onnx_dilations[1];

  int64_t padding_top = onnx_pads[0];
  int64_t padding_bottom = onnx_pads[2];
  int64_t padding_left = onnx_pads[1];
  int64_t padding_right = onnx_pads[3];

  ORT_RETURN_IF_ERROR(ComputePad(input_size_y,
                                 stride_y, weight_size_y, dilation_y,
                                 auto_pad_type,
                                 padding_top, padding_bottom));
  ORT_RETURN_IF_ERROR(ComputePad(input_size_x,
                                 stride_x, weight_size_x, dilation_x,
                                 auto_pad_type,
                                 padding_left, padding_right));

  pads_out = {static_cast<int32_t>(padding_top), static_cast<int32_t>(padding_left),
              static_cast<int32_t>(padding_bottom), static_cast<int32_t>(padding_right)};

  return NeuronStatus::OK();
}

NeuronStatus HandleAutoPad(const Shape& input_shape,
                     const uint32_t weight_size_y,
                     const uint32_t weight_size_x,
                     const std::vector<int32_t>& onnx_strides,
                     const std::vector<int32_t>& onnx_dilations,
                     AutoPadType auto_pad_type,
                     bool use_nchw,
                     std::vector<int32_t>& onnx_pads,
                     int32_t& neuron_padding_code,
                     bool& use_auto_pad) {
  use_auto_pad = false;
  if (auto_pad_type != AutoPadType::NOTSET) {
    ORT_RETURN_IF_ERROR(ComputeConvPads(input_shape, weight_size_y, weight_size_x,
                                        onnx_pads, onnx_strides, onnx_dilations,
                                        auto_pad_type, use_nchw,
                                        onnx_pads));

    if (AutoPadType::VALID == auto_pad_type || AutoPadType::SAME_UPPER == auto_pad_type) {
      use_auto_pad = true;
      neuron_padding_code = (AutoPadType::VALID == auto_pad_type) ? NEURON_PADDING_VALID
                                                                  : NEURON_PADDING_SAME;
    }
  } else if (onnx_dilations == std::vector<int32_t>{1, 1}) {
    // Since Neuron runs more efficiently using auto_pad, we try to map the NOTSET padding to auto_pad
    std::vector<int32_t> same_upper_pads;
    ORT_RETURN_IF_ERROR(ComputeConvPads(input_shape, weight_size_y, weight_size_x,
                                        onnx_pads, onnx_strides, onnx_dilations,
                                        AutoPadType::SAME_UPPER, use_nchw,
                                        same_upper_pads));
    if (onnx_pads == same_upper_pads) {
      use_auto_pad = true;
      neuron_padding_code = NEURON_PADDING_SAME;
    }
  }

  return NeuronStatus::OK();
}

NeuronStatus IsValidInputQuantizedType(const ModelBuilder& model_builder,
                                 const std::string& input_name,
                                 float scale,
                                 int32_t zero_point) {
  const OperandType& input_operand_type = model_builder.GetOperandTypes().at(input_name);
  if (input_operand_type.operandType.scale != scale || input_operand_type.operandType.zeroPoint != zero_point) {
    std::string msg = MakeNeuronString(
        "Quantization parameters in NeuronOperand are differnt from Onnxruntime quantization parameters. Input [",
        input_name, "] Neuron input scale: ", input_operand_type.operandType.scale,
        " zero point: ", input_operand_type.operandType.zeroPoint,
        ", ONNX input scale: ", scale, " zero point: ", zero_point);
    //LOGS_NEURON(ERROR) << msg;
    //return NeuronStatus(NeuronStatusCode::INVALID_ARGUMENT, msg);
  }

  return NeuronStatus::OK();
}

NeuronStatus IsValidConvWeightQuantizedType(const ModelBuilder& model_builder,
                                      const std::string& input_name,
                                      float scale,
                                      int32_t zero_point,
                                      const std::optional<std::vector<float>>& scales) {
  // first verify as the weight has no per-channel quantization
  //ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input_name, scale, zero_point));

  if (scales) {
    const OperandType& input_operand_type = model_builder.GetOperandTypes().at(input_name);
    if (!input_operand_type.channelQuant) {
      return NeuronStatus(NeuronStatusCode::INVALID_ARGUMENT,
                          MakeNeuronString("Input [", input_name, "] has no channelQuant"));
    }

    if (input_operand_type.channelQuant.value().scales != scales.value()) {
      return NeuronStatus(NeuronStatusCode::INVALID_ARGUMENT,
                          MakeNeuronString("Input [", input_name, "] has mismatch scales between onnx and Neuron"));
    }
  }

  return NeuronStatus::OK();
}

void AddQuantizationScaleAndZeroPointToSkip(ModelBuilder& model_builder,
                                            const std::string& scale_name,
                                            const std::string& zero_point_name) {
  if (!scale_name.empty()) {
    model_builder.AddInitializerToSkip(scale_name);
    LOGS_NEURON(VERBOSE) << scale_name << " is skipped";
  }
  if (!zero_point_name.empty()) {
    model_builder.AddInitializerToSkip(zero_point_name);
    LOGS_NEURON(VERBOSE) << zero_point_name << " is skipped";
  }
}

void AddInputToSkip(ModelBuilder& model_builder, const std::string& input_name,
                    const std::string& scale_name,
                    const std::string& zero_point_name) {
  if (!input_name.empty()) model_builder.AddInitializerToSkip(input_name);
  AddQuantizationScaleAndZeroPointToSkip(model_builder, scale_name, zero_point_name);
}

NeuronStatus AddBinaryOperator(int32_t op_type,
                         ModelBuilder& model_builder,
                         const std::string& input1,
                         const std::string& input2,
                         bool add_activation,
                         int32_t fuse_code,
                         const std::string& output,
                         float output_scale,
                         int32_t output_zero_point) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input1));  // input 1
  input_indices.push_back(operand_indices.at(input2));  // input 2

  if (add_activation) {
    ADD_SCALAR_OPERAND(model_builder, input_indices, fuse_code);
  }

  // Diagnostic: MDLA rejects a requantized elementwise op whose input/output
  // scale ratio leaves [6e-14, 32768] ("Act1/Act2 shift exceeds range").  Log
  // the actual scales the plugin assigned so a hardware reject maps to the exact
  // op and reveals whether an input scale is wrong.
  {
    auto scale_of = [&](const std::string& nm) -> float {
      const auto it = operand_types.find(nm);
      return it != operand_types.end() ? it->second.operandType.scale : -1.0f;
    };
    const float s1 = scale_of(input1);
    const float s2 = scale_of(input2);
    LOGS_NEURON(VERBOSE) << "BinaryOp out [" << output << "] in1(" << input1
                         << ")_scale=" << s1 << " in2(" << input2
                         << ")_scale=" << s2 << " out_scale=" << output_scale;
    if (output_scale > 0.0f) {
      for (float s : {s1, s2}) {
        if (s > 0.0f) {
          const float r = s / output_scale;
          if (r < 6e-14f || r > 32768.0f)
            LOGS_NEURON(WARNING) << "BinaryOp [" << output
                                 << "] input/output scale ratio " << r
                                 << " outside MDLA range [6e-14, 32768] — will be rejected";
        }
      }
    }
  }

  const OperandType output_operand_type(operand_types.at(input1).type, shaper[output],
                                        output_scale, output_zero_point);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(op_type, input_indices,
                                                 {output}, {output_operand_type}));

  return NeuronStatus::OK();
}

NeuronStatus AddIdentityOp(ModelBuilder& model_builder, const onnx::NodeProto& node,
                           const NeuronGraphContext& /*ctx*/) {
  // Identity is not really going to do anything
  // Just register the dimension and type, with same index and new name
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  const auto& input = node.input(0);
  const auto& output = node.output(0);

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));  // input

  const OperandType output_operand_type(operand_types.at(input).type,
                                        model_builder.GetShaper()[output]);
  model_builder.RegisterOperand(output, operand_indices.at(input), output_operand_type);
  return NeuronStatus::OK();
}

NeuronStatus AddSqueezeOp(ModelBuilder& model_builder,
                    const std::string& node_name,
                    const std::string& input, const std::string& output,
                    std::vector<int32_t> axes) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  const auto input_shape = shaper[input];
  auto input_dims = input_shape.size();
  for (auto& axis : axes) {
    axis = static_cast<int32_t>(HandleNegativeAxis(axis, input_dims));
  }

  // Neuron requires the axes to be provided
  // If the Op is squeezing all by not specifying axes, the axes is pre-populate
  // with axes of all single dimensions by the caller
  if (axes.empty()) {  // Squeeze all
    for (size_t i = 0; i < input_dims; i++) {
      if (input_shape[i] == 1)
        axes.push_back(static_cast<int32_t>(i));
    }
  }

  const auto axes_name = model_builder.GetUniqueName(node_name + input + "_axes");
  Shape axes_dimen = {static_cast<uint32_t>(axes.size())};
  const OperandType axes_operand_type(Type::TENSOR_INT32, axes_dimen);
  ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(axes_name, axes.data(), axes_operand_type));

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));      // input
  input_indices.push_back(operand_indices.at(axes_name));  // axes

  // Shape inference calculation for squeeze
  int32_t input_size = static_cast<int32_t>(input_shape.size());
  std::unordered_set<int32_t> axes_to_be_squeezed;

  for (const auto& axis : axes) {
    axes_to_be_squeezed.insert(axis);
  }

  // Calculate output dimensions after squeezing
  std::vector<uint32_t> output_dimen;
  output_dimen.reserve(input_size - axes_to_be_squeezed.size());
  for (int32_t i = 0; i < input_size; i++) {
    if (!Contains(axes_to_be_squeezed, i))
      output_dimen.push_back(input_shape[i]);
  }

  // In case of a tensor has all 1's in dimension such as {1,1,1,1} and gets squeezed all
  // the output shape will be {1}
  if (output_dimen.empty())
    output_dimen.push_back(1);
  shaper.AddShape(output, output_dimen);

  const OperandType& input_operand_type = operand_types.at(input);
  OperandType output_operand_type(static_cast<NeuronOperandType>(input_operand_type));
  output_operand_type.SetDimensions(output_dimen);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(NEURON_SQUEEZE, input_indices,
                                                 {output}, {output_operand_type}));
  return NeuronStatus::OK();
}

NeuronStatus GetAxesForSqueezeAndUnSqueeze(ModelBuilder& model_builder, const onnx::NodeProto& node,
                                           const NeuronGraphContext& /*ctx*/,
                                           std::vector<int32_t>& axes) {
  // Opset 13+ stores axes in input[1] (an INT64 initializer).
  // Opset < 13 stores axes in the "axes" attribute; input[1] is absent/empty.
  // OrtGraphToProto never emits a "since_version" attribute, so we detect the
  // opset variant by checking whether input[1] is present instead.
  if (node.input_size() > 1 && !node.input(1).empty()) {
    // Opset 13+: axes from initializer at input[1].
    // For Squeeze, axes is optional even in opset 13+; if the name is absent
    // we leave axes empty (squeeze all dims of size 1).
    const auto& initializers(model_builder.GetInitializerTensors());
    const auto& axes_name = node.input(1);
    if (initializers.count(axes_name)) {
      const auto& axes_tensor = *initializers.at(axes_name);
      std::vector<int64_t> raw_axes;
      for (auto v : axes_tensor.int64_data()) raw_axes.push_back(v);
      axes = OnnxAxesToNeuron(gsl::make_span(raw_axes), std::nullopt);
    }
  } else {
    // Opset < 13: axes from the "axes" attribute.
    for (const auto& attr : node.attribute()) {
      if (attr.name() == "axes") {
        std::vector<int64_t> axes_int64(attr.ints().begin(), attr.ints().end());
        axes = OnnxAxesToNeuron(gsl::make_span(axes_int64), std::nullopt);
        break;
      }
    }
  }

  return NeuronStatus::OK();
}

NeuronStatus AddMinMaxOperator(ModelBuilder& model_builder, const onnx::NodeProto& node,
                               const NeuronGraphContext& /*ctx*/,
                               const std::string& input1, const std::string& input2) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  const auto& output = node.output(0);

  const auto& op_type(node.op_type());
  int32_t op_code;
  if (op_type == "Min")
    op_code = NEURON_MINIMUM;
  else if (op_type == "Max")
    op_code = NEURON_MAXIMUM;
  else {
    return NeuronStatus(NeuronStatusCode::INVALID_ARGUMENT,
                        MakeNeuronString("MinMaxOpBuilder, unknown op: ", op_type));
  }

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input1));  // input 1
  input_indices.push_back(operand_indices.at(input2));  // input 2

  const OperandType output_operand_type(operand_types.at(input1).type, shaper[output]);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(op_code, input_indices,
                                                 {output}, {output_operand_type}));

  return NeuronStatus::OK();
}

NeuronStatus AddReshapeOperator(ModelBuilder& model_builder,
                                const onnx::NodeProto& node,
                                const NeuronGraphContext& /*ctx*/,
                                const std::string& input,
                                const std::vector<int32_t>& shape) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto& output = node.output(0);

  const auto input_shape = shaper[input];
  const auto output_shape = shaper[output];

  // For reshape, the output type should be the same as the input type except the shape is different
  auto output_operand_type = operand_types.at(input);
  output_operand_type.SetDimensions(output_shape);

  {
    std::string shape_name = model_builder.GetUniqueName(node.name() + input + "newshape");
    ORT_RETURN_IF_ERROR(op_builder_helpers::AddNeuronReshape(model_builder, input, shape_name, shape, output));
  }

  return NeuronStatus::OK();
}

bool IsQuantizationScaleSupported(const NeuronGraphContext& ctx,
                                  const std::string& scale_name,
                                  const OpSupportCheckParams& params,
                                  const std::string& op_type,
                                  bool is_quant_matmul,
                                  bool is_conv_matmul_u8s8_weight) {
  const auto* scale = ctx.GetInitializer(scale_name);
  if (!scale) {
    LOGS_NEURON(VERBOSE) << "The scale of " << op_type << " must be a constant initializer";
    return false;
  }

  const auto& scale_tensor = *scale;
  int64_t scales_dim = scale_tensor.dims().empty() ? 1 : scale_tensor.dims(0);
  if (!is_conv_matmul_u8s8_weight) {
    if (scales_dim != 1) {
      LOGS_NEURON(VERBOSE) << op_type << " does not support per-channel quantization, "
                           << " for now, only u8s8 QlinearConv supports per-channel quantization on API 29+";
      return false;
    }
  } else if (scales_dim != 1) {
    // For u8s8 Qlinear[Conv/MatMul], we support
    // 1. Per-tensor, the weight will be transformed to uint8 later
    // 2. Per-channel, only from Android API level 29
    if (is_quant_matmul) {
      LOGS_NEURON(VERBOSE) << "QLinearMatMul does not support per-channel quantization";
      return false;
    }
  }

  return true;
}

bool IsQuantizationZeroPointSupported(const NeuronGraphContext& ctx,
                                      const std::string& zero_point_name_opt,
                                      const std::string& op_type,
                                      bool is_quant_matmul,
                                      bool is_conv_matmul_u8s8_weight) {
  // zero point is optional
  if (zero_point_name_opt.empty()) return true;

  const auto* zero_point = ctx.GetInitializer(zero_point_name_opt);
  if (!zero_point) {
    LOGS_NEURON(VERBOSE) << "The zero point of " << op_type << " must be a constant initializer";
    return false;
  }

  return true;
}

// ---------------------------------------------------------------------------
// GetBinaryOpQuantizationScaleAndZeroPoint
// For binary quantized ops: inputs[0]=A, inputs[1]=B, outputs[0]=Y.
// Scale/ZP names for each are the initializer inputs immediately following.
// ---------------------------------------------------------------------------
NeuronStatus GetBinaryOpQuantizationScaleAndZeroPoint(
    const NeuronGraphContext& ctx, const onnx::NodeProto& node,
    float& a_scale, float& b_scale, float& y_scale,
    int32_t& a_zero_point, int32_t& b_zero_point, int32_t& y_zero_point) {
  // QLinearAdd layout: A, a_scale, a_zp, B, b_scale, b_zp, y_scale, y_zp
  auto inp = [&](int i) { return i < node.input_size() ? node.input(i) : std::string{}; };
  using onnxruntime::neuron::GetQuantizationScaleAndZeroPoint;
  NEURON_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(ctx, inp(1), inp(2), a_scale, a_zero_point));
  NEURON_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(ctx, inp(4), inp(5), b_scale, b_zero_point));
  NEURON_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(ctx, inp(6), inp(7), y_scale, y_zero_point));
  return NeuronStatus::OK();
}

// ---------------------------------------------------------------------------
// GetConvMatMulOpQuantizationScaleAndZeroPoint
// For QLinearConv/QLinearMatMul: scale/zp names are node inputs at fixed slots.
// ---------------------------------------------------------------------------
NeuronStatus GetConvMatMulOpQuantizationScaleAndZeroPoint(
    const ModelBuilder& model_builder,
    const NeuronGraphContext& ctx,
    const onnx::NodeProto& node,
    float& a_scale, float& w_scale, float& y_scale,
    int32_t& a_zero_point, int32_t& w_zero_point, int32_t& y_zero_point,
    std::optional<std::vector<float>>& w_scales,
    bool& is_per_tensor_u8s8) {
  is_per_tensor_u8s8 = false;
  using onnxruntime::neuron::GetQuantizationScaleAndZeroPoint;

  // QLinearConv layout: x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp [, B]
  const auto input = [&](int i) -> std::string {
    return (i < node.input_size()) ? node.input(i) : std::string{};
  };

  NEURON_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(ctx, input(1), input(2), a_scale, a_zero_point));
  NEURON_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(ctx, input(4), input(5), w_scale, w_zero_point));
  NEURON_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(ctx, input(6), input(7), y_scale, y_zero_point));

  // Check for per-channel weight scale
  const auto& weight_name = input(3);
  if (!weight_name.empty()) {
    int32_t weight_type = ctx.GetElementType(weight_name);
    if (weight_type == onnx::TensorProto::UINT8)
      return NeuronStatus::OK();  // u8u8 — done

    // Check scale tensor dimension for per-channel vs per-tensor
    const std::string& w_scale_name = input(4);
    const auto* scale_tensor = ctx.GetInitializer(w_scale_name);
    if (scale_tensor && !scale_tensor->dims().empty() && scale_tensor->dims(0) > 1) {
      // Per-channel u8s8
      w_scale = 0.0f; w_zero_point = 0;
      std::vector<float> sv;
      if (NeuronGraphContext::GetFloatData(*scale_tensor, sv).IsOK())
        w_scales = std::move(sv);
    } else {
      // Per-tensor u8s8
      w_zero_point = 128;
      is_per_tensor_u8s8 = true;
    }
  }
  (void)model_builder;
  return NeuronStatus::OK();
}

// ---------------------------------------------------------------------------
// IsOpInRequiredLayout
// ---------------------------------------------------------------------------
NeuronStatus IsOpInRequiredLayout(bool use_nchw, const onnx::NodeProto& node,
                                  const NeuronGraphContext& /*ctx*/) {
  // In the plugin EP NCHW is always the preferred layout (GetPreferredLayout returns NCHW).
  // Nodes should never arrive in NHWC domain; if they do, it's a bug.
  if (use_nchw && IsNodeLayoutNHWC(node)) {
    return NeuronStatus{false, MakeNeuronString(
        "Expected NCHW but node [", node.name(), "] is in NHWC domain")};
  }
  return NeuronStatus::OK();
}

// ---------------------------------------------------------------------------
// IsQuantizedIOSupported
// ---------------------------------------------------------------------------
bool IsQuantizedIOSupported(const NeuronGraphContext& ctx, const onnx::NodeProto& node,
                            const std::vector<size_t>& indices,
                            const OpSupportCheckParams& /*params*/,
                            bool is_input) {
  // For QDQ-wrapped ops in the plugin EP, the target node's own inputs are
  // FLOAT32 (DQ outputs) and its output is FLOAT32 (Q input).  The actual
  // quantized types live in the DQ/Q sibling nodes captured in ctx.qdq_groups.
  // Consult those params first; fall back to ONNX tensor type for non-QDQ ops.
  const QDQNodeGroup* qdq = nullptr;
  if (node.output_size() > 0 && !node.output(0).empty())
    qdq = ctx.GetQDQGroup(node.output(0));

  for (size_t idx : indices) {
    const int n = is_input ? node.input_size() : node.output_size();
    if (static_cast<int>(idx) >= n) {
      LOGS_NEURON(VERBOSE) << node.op_type() << " " << (is_input ? "input" : "output")
                           << " index " << idx << " out of range " << n;
      return false;
    }
    const std::string& name = is_input ? node.input(static_cast<int>(idx))
                                       : node.output(static_cast<int>(idx));
    if (name.empty()) continue;

    int32_t elem_type;
    if (is_input && qdq && idx < qdq->input_quant_params.size() &&
        qdq->input_quant_params[idx].has_value()) {
      elem_type = qdq->input_quant_params[idx]->quant_type;
    } else if (!is_input && qdq && qdq->output_quant_params.has_value()) {
      elem_type = qdq->output_quant_params->quant_type;
    } else {
      elem_type = ctx.GetElementType(name);
    }

    if (elem_type != onnx::TensorProto::UINT8 && elem_type != onnx::TensorProto::INT8) {
      LOGS_NEURON(VERBOSE) << node.op_type() << " unsupported quant type "
                           << GetTensorProtoDataTypeStr(elem_type) << " at index " << idx;
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// HasRequiredScaleAndZeroPoint
// ---------------------------------------------------------------------------
bool HasRequiredScaleAndZeroPoint(const NeuronGraphContext& ctx,
                                  const std::string& op_desc,
                                  const std::string& scale_name,
                                  const std::string& zp_name,
                                  float required_scale, int32_t required_zp) {
  using onnxruntime::neuron::GetQuantizationScaleAndZeroPoint;
  float scale = 0.0f; int32_t zp = 0;
  auto st = GetQuantizationScaleAndZeroPoint(ctx, scale_name, zp_name, scale, zp);
  if (!st.IsOK()) {
    LOGS_NEURON(VERBOSE) << op_desc << ": GetQuantizationScaleAndZeroPoint failed: " << st.ErrorMessage();
    return false;
  }
  if (scale != required_scale) {
    LOGS_NEURON(VERBOSE) << op_desc << ": scale must be " << required_scale << ", got " << scale;
    return false;
  }
  if (zp != required_zp) {
    LOGS_NEURON(VERBOSE) << op_desc << ": zero_point must be " << required_zp << ", got " << zp;
    return false;
  }
  return true;
}

NeuronStatus PerformBroadcasting(const Shape& shape1, const Shape& shape2, Shape& output_shape) {
  const size_t rank = std::max(shape1.size(), shape2.size());
  output_shape.resize(rank);
  for (size_t i = 0; i < rank; ++i) {
    uint32_t d1 = (i < rank - shape1.size()) ? 1u : shape1[i - (rank - shape1.size())];
    uint32_t d2 = (i < rank - shape2.size()) ? 1u : shape2[i - (rank - shape2.size())];
    if (d1 == d2)      { output_shape[i] = d1; }
    else if (d1 == 1)  { output_shape[i] = d2; }
    else if (d2 == 1)  { output_shape[i] = d1; }
    else {
      return NeuronStatus{false, MakeNeuronString(
          "PerformBroadcasting: incompatible dims ", d1, " vs ", d2, " at axis ", i)};
    }
  }
  return NeuronStatus::OK();
}

}  // namespace onnxruntime::neuron::op_builder_helpers
