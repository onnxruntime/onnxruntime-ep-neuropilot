// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "op_builder.h"

namespace onnxruntime {
namespace neuron {

struct OpBuilderRegistrations {
  std::vector<std::unique_ptr<IOpBuilder>> builders;
  std::unordered_map<std::string, const IOpBuilder*> op_builder_map;
};

// Get the lookup table with IOpBuilder delegates for different onnx operators
// Note, the lookup table should have same number of entries as the result of CreateOpBuilders() in op_builder.h
const std::unordered_map<std::string, const IOpBuilder*>& GetOpBuilders();

void CreateGridSampleOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateExpandOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateBatchNormalizationOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateLayerNormalizationOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateInstanceNormalizationOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateCastOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateClipOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateConcatOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateDepthToSpaceOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateDequantizeLinearOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateEluOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateFlattenOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateGatherOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateIdentityOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateLeakyReluOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateLRNOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreatePadOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateQuantizeLinearOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateReluOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateGeluOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateReshapeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateResizeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateSliceOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateSoftMaxOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateSplitOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateSqueezeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateTransposeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateUnsqueezeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateArgMaxOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);

void CreateBinaryOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateConvOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateGemmOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateMatMulOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateMatMulNBitsOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);  // MatMulNBits
void CreateSkipSimplifiedLayerNormOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreatePoolOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateMinMaxOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateUnaryOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);
void CreateReductionOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);

}  // namespace neuron
}  // namespace onnxruntime
