// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <chrono>
#include <gsl/span>
#include <thread>

#include "neuron_wrapper_model.h"

#include "../neuron_types_internal.h"
#include "../builders/helper.h"
#include "neuron_implementation.h"

#ifdef USE_NEURON_SHARED_MEM
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace onnxruntime {
namespace neuron {
namespace wrapper {

#pragma region Model

Model::Model(const NeuronApi& neuron_handle) : neuron_(neuron_handle) {}

Model::~Model() {
  neuron_.NeuronCompilation_free(compilation_);
  neuron_.NeuronModel_free(model_);
}

void Model::AddInput(const std::string& name, const neuron::wrapper::OperandType& operand_type) {
  input_names_.push_back(name);
  operand_types_.emplace(name, operand_type);
}

void Model::AddOutput(const std::string& onnx_output_name,
                      const std::string& neuron_output_name,
                      const neuron::wrapper::OperandType& operand_type) {
  LOGS_NEURON(VERBOSE) << "Model::AddOutput output name " << onnx_output_name
                        << " shape " << Shape2String(operand_type.dimensions);

  output_names_.push_back(onnx_output_name);
  onnx_to_neuron_output_map_.emplace(onnx_output_name, neuron_output_name);
  operand_types_.emplace(neuron_output_name, operand_type);
}

bool Model::IsScalarOutput(const std::string& output_name) const {
  return Contains(scalar_outputs_, output_name);
}

void Model::AddScalarOutput(const std::string& output_name) {
  scalar_outputs_.insert(output_name);
}

const std::vector<std::string>& Model::GetInputs() const {
  return input_names_;
}

const std::vector<std::string>& Model::GetOutputs() const {
  return output_names_;
}

const neuron::wrapper::OperandType& Model::GetInputType(const std::string& name) const {
  return operand_types_.at(name);
}

neuron::wrapper::OperandType Model::GetOutputType(const std::string& name,
                                                  const Execution& /* execution */) const {
  // Note: Before we validate if it's required to get Shaper from execution (if encounter dynamic shapes,
  // shape can get updated during execution), we commented the usage here for now.
  /* neuron::wrapper::OperandType type(
      output_type.type, execution.GetShaper()[neuron_output_name], output_type.operandType.scale,
                                                                  output_type.operandType.zeroPoint); */
  return operand_types_.at(name);
}

void Model::SetInputMap(std::unordered_map<std::string, size_t>&& input_map) {
  input_map_ = std::move(input_map);
}

void Model::SetOutputMap(std::unordered_map<std::string, size_t>&& output_map) {
  output_map_ = std::move(output_map);
}

size_t Model::GetMappedInputIdx(const std::string& name) const {
  return input_map_.at(name);
}

size_t Model::GetMappedOutputIdx(const std::string& name) const {
  return output_map_.at(name);
}

bool Model::SupportsDynamicOutputShape() const {
  // Neuron does not support dynamic output shape
  return false;
}

NeuronStatus Model::PrepareForExecution(std::unique_ptr<Execution>& execution) {
  ORT_RETURN_IF_NOT(nullptr != compilation_,
                    "Error in PrepareForExecution, compilation_ is null");

  NeuronExecution* neuron_execution;
  NEURON_RETURN_STATUS_ON_ERROR(
      neuron_.NeuronExecution_create(compilation_, &neuron_execution));

  execution = std::make_unique<Execution>(*neuron_execution /*, shaper_*/, neuron_);
  return NeuronStatus::OK();
}

NeuronStatus Model::PrepareForSharedExecution(std::shared_ptr<Execution>& execution) {
  ORT_RETURN_IF_NOT(nullptr != compilation_,
                    "Error in PrepareForExecution, compilation_ is null");

  if (shared_execution_ == nullptr) {
    NeuronExecution* neuron_execution;
    NEURON_RETURN_STATUS_ON_ERROR(
        neuron_.NeuronExecution_create(compilation_, &neuron_execution));

    shared_execution_ = std::make_shared<Execution>(*neuron_execution /*, shaper_*/, neuron_);
  }
  execution = shared_execution_;
  return NeuronStatus::OK();
}

#pragma region Model::NNMemory

#ifdef USE_NEURON_SHARED_MEM
Model::NNMemory::NNMemory(const NeuronApi& neuron_handle, const char* name, size_t size) : neuron_(neuron_handle) {
  if (name && size > 0) {
    byte_size_ = size;
    fd_ = neuron_.ASharedMemory_create(name, size);
    data_ptr_ = reinterpret_cast<uint8_t*>(
        mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));
    NEURON_THROW_ON_ERROR(neuron_.NeuronMemory_createFromFd(size, PROT_READ | PROT_WRITE,
                                                            fd_, 0, &nn_memory_handle_));
  }
}

Model::NNMemory::~NNMemory() {
  if (nn_memory_handle_) {
    neuron_.NeuronMemory_free(nn_memory_handle_);
  }
  if (data_ptr_) {
    munmap(data_ptr_, byte_size_);
  }

  if (fd_ >= 0) close(fd_);
}
#else
Model::NNMemory::NNMemory(const NeuronApi& /*neuron_handle*/, const char* name, size_t size) {
  if (name && size > 0) {
    data_.resize(size);
  }
}
#endif

#pragma endregion

#pragma endregion

#pragma region Execution

Execution::Execution(NeuronExecution& execution /*, const Shaper& shaper */,
                     const NeuronApi& neuron_handle)
    : neuron_(neuron_handle),
      execution_(&execution) {
}

Execution::~Execution() {
  neuron_.NeuronExecution_free(execution_);
}

NeuronStatus Execution::SetInputBuffers(const std::vector<Execution::InputBuffer>& inputs) {
  for (size_t i = 0; i < inputs.size(); i++) {
    ORT_RETURN_IF_ERROR(SetInputBuffer(static_cast<int32_t>(i), inputs[i]));
  }

  return NeuronStatus::OK();
}

NeuronStatus Execution::SetOutputBuffers(const std::vector<OutputBuffer>& outputs) {
  for (size_t i = 0; i < outputs.size(); i++) {
    ORT_RETURN_IF_ERROR(SetOutputBuffer(static_cast<int32_t>(i), outputs[i]));
  }

  return NeuronStatus::OK();
}

NeuronStatus Execution::SetInputBuffer(const int32_t index, const InputBuffer& input) {
  // TODO: Use AHWB to reduce copy
  NEURON_RETURN_STATUS_ON_ERROR(neuron_.NeuronExecution_setInput(
      execution_, index, &input.type.operandType, input.buffer, input.type.GetOperandBlobByteSize()));

  return NeuronStatus::OK();
}

NeuronStatus Execution::SetOutputBuffer(const int32_t index, const OutputBuffer& output) {
  LOGS_NEURON(VERBOSE) << "Model::SetOutputBuffer, output shape "
                        << Shape2String(output.type.dimensions);

  // TODO: Use AHWB to reduce copy
  NEURON_RETURN_STATUS_ON_ERROR(neuron_.NeuronExecution_setOutput(
      execution_, index, &output.type.operandType, output.buffer, output.buffer_byte_size));

  return NeuronStatus::OK();
}

NeuronStatus Execution::SetInputFast(int32_t index, const void* buffer, size_t byte_size) {
  // Pass nullptr for operandType — reuse the compiled type (no per-call copy).
  NEURON_RETURN_STATUS_ON_ERROR(neuron_.NeuronExecution_setInput(
      execution_, index, nullptr, buffer, byte_size));
  return NeuronStatus::OK();
}

NeuronStatus Execution::SetOutputFast(int32_t index, void* buffer, size_t byte_size) {
  NEURON_RETURN_STATUS_ON_ERROR(neuron_.NeuronExecution_setOutput(
      execution_, index, nullptr, buffer, byte_size));
  return NeuronStatus::OK();
}

// The Neuron API splits its failure codes into TRANSIENT and PERSISTENT pairs.
// TODO: try to understand what this means.
static bool IsTransientNeuronError(int code) {
  return code == NEURON_MISSED_DEADLINE_TRANSIENT ||
         code == NEURON_RESOURCE_EXHAUSTED_TRANSIENT;
}

NeuronStatus Execution::Predict() {
  constexpr int kMaxAttempts = 3;
  int result = NEURON_NO_ERROR;
  for (int attempt = 1; attempt <= kMaxAttempts; ++attempt) {
    result = neuron_.NeuronExecution_compute(execution_);
    if (!IsTransientNeuronError(result)) break;
    LOGS_NEURON(VERBOSE) << "NeuronExecution_compute returned transient error "
                         << result << " (attempt " << attempt << " of "
                         << kMaxAttempts << "); retrying";

    if (attempt < kMaxAttempts)
      std::this_thread::sleep_for(std::chrono::milliseconds(5 * attempt));
  }
  NEURON_RETURN_STATUS_ON_ERROR(result);
  return NeuronStatus::OK();
}

NeuronStatus Execution::Predict(const std::vector<int32_t>& dynamic_outputs, std::vector<Shape>& dynamic_output_shapes) {
  // {
  //   NeuronEvent* event = nullptr;
  //   auto ret = neuron_.NeuronExecution_startComputeWithDependencies(execution_, nullptr, 0, 0, &event);
  //   auto free_event = gsl::finally([&]() { neuron_.NeuronEvent_free(event); });
  //   RETURN_STATUS_ON_ERROR(ret);
  //   RETURN_STATUS_ON_ERROR(neuron_.NeuronEvent_wait(event));
  // }
  NEURON_RETURN_STATUS_ON_ERROR(neuron_.NeuronExecution_compute(execution_));

  dynamic_output_shapes.clear();
  dynamic_output_shapes.reserve(dynamic_outputs.size());
  for (const int32_t i : dynamic_outputs) {
    uint32_t output_rank = 0;
    NEURON_RETURN_STATUS_ON_ERROR(neuron_.NeuronExecution_getOutputOperandRank(execution_, i, &output_rank));

    std::vector<uint32_t> output_shape(output_rank);
    NEURON_RETURN_STATUS_ON_ERROR(neuron_.NeuronExecution_getOutputOperandDimensions(execution_, i, output_shape.data()));

    dynamic_output_shapes.push_back(output_shape);
  }

  return NeuronStatus::OK();
}

#pragma endregion

}  // namespace wrapper
}  // namespace neuron
}  // namespace onnxruntime
