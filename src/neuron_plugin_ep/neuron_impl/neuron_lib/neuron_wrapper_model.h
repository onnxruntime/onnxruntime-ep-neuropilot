// neuron_wrapper_model.h
// (Plugin EP refactoring: removed core/ ORT headers.
//  OrtMutex replaced with std::mutex.
//  onnxruntime::Status replaced with NeuronStatus.
//  InlinedVector replaced with std::vector.)

#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../neuron_types_internal.h"
#include "neuron_wrapper_operand.h"

struct NeuronApi;
namespace onnxruntime {
namespace neuron {

// Forward declaration; ModelBuilder defined in builders/neuron_model_builder.h
class ModelBuilder;

namespace wrapper {

#if defined(__ANDROID__)
#define USE_NEURON_SHARED_MEM 1
#endif

class Execution;

class Model {
  friend class neuron::ModelBuilder;

 public:
  // Memory for persist data such as initializers and intermediate results.
#ifdef USE_NEURON_SHARED_MEM
  class NNMemory {
   public:
    NNMemory(const NeuronApi& neuron, const char* name, size_t size);
    ~NNMemory();
    NeuronMemory* GetHandle() { return nn_memory_handle_; }
    uint8_t* GetDataPtr() { return data_ptr_; }
    size_t GetByteSize() { return byte_size_; }
   private:
    const NeuronApi& neuron_;
    int fd_{-1};
    size_t byte_size_{0};
    uint8_t* data_ptr_{nullptr};
    NeuronMemory* nn_memory_handle_{nullptr};
  };
#else
  class NNMemory {
   public:
    NNMemory(const NeuronApi& neuron_handle, const char* name, size_t size);
    ~NNMemory() = default;
    uint8_t* GetDataPtr() { return data_.data(); }
    size_t GetByteSize() { return data_.size(); }
   private:
    std::vector<uint8_t> data_;
  };
#endif

 public:
  explicit Model(const NeuronApi& neuron_handle);
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  const std::vector<std::string>& GetInputs() const;
  const std::vector<std::string>& GetOutputs() const;

  // The GroupQueryAttention lowering needs position-dependent vectors (rotary
  // rows, KV-write selectors, the causal mask) that nothing on the NPU can
  // derive from seqlens_k (no GATHER / CAST / int32 compute on MDLA).  The
  // builder registers them as extra Neuron model inputs; ComputeImpl fills
  // them on the host each inference from the seqlens_k value.
  struct SynthInput {
    enum Kind { kCosRow, kSinRow, kKeep, kOneHot, kMask };
    std::string name;
    Kind kind;
    size_t byte_size;
  };
  const std::vector<SynthInput>& GetSynthInputs() const { return synth_inputs_; }
  // fp32 rotary tables, rows [0, rows), row length `half`, sliced from the
  // model's cos_cache/sin_cache by the builder.
  const std::vector<float>& GetRotaryCos() const { return rotary_cos_; }
  const std::vector<float>& GetRotarySin() const { return rotary_sin_; }
  uint32_t GetRotaryHalf() const { return rotary_half_; }
  size_t GetSynthMaxSeq() const {
    return rotary_half_ ? rotary_cos_.size() / rotary_half_ : 0;
  }
  const std::string& GetSeqlensInputName() const { return seqlens_input_name_; }

  const neuron::wrapper::OperandType& GetInputType(const std::string& name) const;
  neuron::wrapper::OperandType GetOutputType(const std::string& name,
                                              const Execution& execution) const;

  void SetInputMap(std::unordered_map<std::string, size_t>&& input_map);
  void SetOutputMap(std::unordered_map<std::string, size_t>&& output_map);

  size_t GetMappedInputIdx(const std::string& name) const;
  size_t GetMappedOutputIdx(const std::string& name) const;

  bool SupportsDynamicOutputShape() const;
  size_t GetDynamicOutputBufferSize() const { return dynamic_output_buffer_size_; }
  void SetDynamicOutputBufferSize(size_t size) { dynamic_output_buffer_size_ = size; }

  std::mutex& GetMutex() { return mutex_; }

  bool IsScalarOutput(const std::string& output_name) const;

  // Rank of the output in the ONNX graph.  Operands with leading unit dims
  // squeezed away are re-padded to this rank when the output
  // shape is reported to ORT.  0 means "same as the operand".
  void SetOrtOutputRank(const std::string& output_name, size_t rank) {
    ort_output_ranks_[output_name] = rank;
  }
  size_t GetOrtOutputRank(const std::string& output_name) const {
    auto it = ort_output_ranks_.find(output_name);
    return it != ort_output_ranks_.end() ? it->second : 0;
  }

  NeuronStatus PrepareForExecution(std::unique_ptr<Execution>& execution);
  NeuronStatus PrepareForSharedExecution(std::shared_ptr<Execution>& execution);

 private:
  const NeuronApi& neuron_;
  int32_t neuron_effective_feature_level_{0};
  NeuronModel* model_{nullptr};
  NeuronCompilation* compilation_{nullptr};
  std::shared_ptr<Execution> shared_execution_{nullptr};

  size_t dynamic_output_buffer_size_{1024};

  std::unique_ptr<NNMemory> mem_initializers_;
  std::vector<std::unique_ptr<NNMemory>> mem_persist_buffers_;

  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::unordered_map<std::string, neuron::wrapper::OperandType> operand_types_;
  std::unordered_set<std::string> scalar_outputs_;

  std::unordered_map<std::string, size_t> input_map_;
  std::unordered_map<std::string, size_t> output_map_;

  // Synth-input metadata (see SynthInput above); set by ModelBuilder.
  std::vector<SynthInput> synth_inputs_;
  std::vector<float> rotary_cos_, rotary_sin_;
  uint32_t rotary_half_{0};
  std::string seqlens_input_name_;
  std::unordered_map<std::string, std::string> onnx_to_neuron_output_map_;
  std::unordered_map<std::string, size_t> ort_output_ranks_;

  std::mutex mutex_;

  void AddInput(const std::string& name, const neuron::wrapper::OperandType& operand_type);
  void AddOutput(const std::string& onnx_output_name,
                 const std::string& neuron_output_name,
                 const neuron::wrapper::OperandType& operand_type);
  void AddScalarOutput(const std::string& output_name);
};

class Execution {
 public:
  struct InputBuffer {
    const std::string& name;
    const void* buffer{nullptr};
    neuron::wrapper::OperandType type;
  };

  struct OutputBuffer {
    void* buffer{nullptr};
    neuron::wrapper::OperandType type;
    size_t buffer_byte_size;
  };

  explicit Execution(NeuronExecution& execution, const NeuronApi& neuron_handle);
  ~Execution();
  Execution(const Execution&) = delete;
  Execution& operator=(const Execution&) = delete;

  NeuronStatus SetInputBuffers(const std::vector<InputBuffer>& inputs);
  NeuronStatus SetOutputBuffers(const std::vector<OutputBuffer>& outputs);
  NeuronStatus Predict(const std::vector<int32_t>& dynamic_outputs,
                       std::vector<Shape>& dynamic_output_shapes);

  // Fast path: pass nullptr for operandType (use compiled type) and pre-computed sizes.
  NeuronStatus SetInputFast(int32_t index, const void* buffer, size_t byte_size);
  NeuronStatus SetOutputFast(int32_t index, void* buffer, size_t byte_size);
  NeuronStatus Predict();

 private:
  NeuronStatus SetInputBuffer(const int32_t index, const InputBuffer& input);
  NeuronStatus SetOutputBuffer(const int32_t index, const OutputBuffer& output);

  const NeuronApi& neuron_;
  NeuronExecution* execution_;
};

}  // namespace wrapper
}  // namespace neuron
}  // namespace onnxruntime
