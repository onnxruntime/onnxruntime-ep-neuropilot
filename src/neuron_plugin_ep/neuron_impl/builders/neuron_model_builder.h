// neuron_model_builder.h
// Plugin EP version – takes NeuronGraphContext instead of GraphViewer.
// All ORT internal headers removed.

#pragma once

#include <onnx/onnx_pb.h>
#include <gsl/span>

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../neuron_graph_context.h"
#include "../neuron_lib/neuron_wrapper_device.h"
#include "../neuron_lib/neuron_wrapper_model.h"
#include "../neuron_lib/neuron_wrapper_operand.h"
#include "shaper.h"

struct NeuronApi;

namespace onnxruntime {
namespace neuron {

class IOpBuilder;

static constexpr const char* kExtensionGeneralOpration =
    "com.mediatek.general_operation";

enum {
  ADAPTER_EXTENSION_GENERAL_OPERAND_ARGSTRING = 0x0100,
  ADAPTER_EXTENSION_GENERAL_OPERATION_TYPE = 0x0000,
};

struct OemOperandValue {
  uint8_t* type;
  uint8_t typeLen;
  uint8_t* data;
  uint32_t dataLen;
};

class ModelBuilder {
 public:
  enum DataLayout {
    NO_CHANGE = 0,
    L_0231 = 1,
    L_1230 = 2,
  };

  ModelBuilder(const NeuronGraphContext& ctx,
               const NeuronApi& neuron_handle,
               NeuronRuntimeVersion neuron_version,
               gsl::span<const neuron::wrapper::Device> neuron_target_devices,
               neuron::wrapper::TargetDeviceOption target_device_option,
               const std::string& neuron_optimization_string);

  // asks the target devices whether they can run every operation in the model
  NeuronStatus ValidateDeviceSupport();

  NeuronStatus Compile(std::unique_ptr<neuron::wrapper::Model>& model);
  NeuronStatus RestoredFromAdapterFormattedDla(std::unique_ptr<neuron::wrapper::Model>& model);

  // Add a Neuron operation.
  NeuronStatus AddOperation(int op,
                             const std::vector<uint32_t>& input_indices,
                             const std::vector<std::string>& output_names,
                             const std::vector<neuron::wrapper::OperandType>& output_types);

  // Register an operand alias for an absorbed DQ or Q node.
  // Called by AddOperations() instead of emitting a NEURON_DEQUANTIZE /
  // NEURON_QUANTIZE operation when the node is part of a QDQ group.
  NeuronStatus AddQDQOperandAlias(const onnx::NodeProto& dq_or_q_node);

  // Extension operation helpers.
  NeuronStatus GetExtensionGeneralOperationType(NeuronOperationType* type);
  NeuronStatus AddCustomNameStringOperand(const std::string& custom_op_name,
                                           const std::string& operand_name,
                                           uint32_t& index);
  NeuronStatus AddScalarExtensionOperand(const std::string& operand_name,
                                          const uint8_t* value,
                                          uint32_t length,
                                          uint32_t& operand_index);
  NeuronStatus AddDLaOperand(const onnx::NodeProto& node, const NeuronGraphContext& ctx,
                              uint32_t& index);
  NeuronStatus AddDlaOperation(const onnx::NodeProto& node, const NeuronGraphContext& ctx);

  // Find fuseable activation (Relu/Relu1/Relu6) following this node.
  int32_t FindActivation(const onnx::NodeProto& node, const NeuronGraphContext& ctx);

  // Scalar operand helpers.
  NeuronStatus AddOperandFromScalar(bool value, uint32_t& index);
  NeuronStatus AddOperandFromScalar(float value, uint32_t& index);
  NeuronStatus AddOperandFromScalar(int32_t value, uint32_t& index);

  // Tensor operand helpers.
  NeuronStatus AddOperandFromPersistMemoryBuffer(
      const std::string& name, const void* buffer,
      const neuron::wrapper::OperandType& operand_type);

  NeuronStatus AddOperandFromPersistMemoryBufferWithNewLayout(
      const std::string& name, const void* buffer,
      const neuron::wrapper::OperandType& operand_type,
      ModelBuilder::DataLayout new_layout);

  // Add a constant initializer tensor from the graph context.
  NeuronStatus AddOperandFromConstantInitializer(
      const onnx::NodeProto& node,
      const NeuronGraphContext& ctx,
      int input_index,
      bool dequantize_to_float32,
      ModelBuilder::DataLayout new_layout,
      const std::string& new_operand_name);

  // Registers a Neuron model input the EP fills on the host each inference
  NeuronStatus AddSynthModelInput(const std::string& name,
                                   const neuron::wrapper::OperandType& operand_type,
                                   neuron::wrapper::Model::SynthInput::Kind kind,
                                   uint32_t& index);
  // fp32 rotary tables (rows [0, max_seq), row length half) used to fill the
  // cos/sin row inputs, and the name of the graph input carrying seqlens_k.
  void SetSynthRotaryTables(std::vector<float> cos_rows, std::vector<float> sin_rows,
                            uint32_t half);
  void SetSynthSeqlensName(const std::string& name);

  void AddInitializerToSkip(const std::string& tensor_name);
  // Excludes a graph input from Neuron model registration.
  void AddInputToSkip(const std::string& tensor_name);
  void RegisterOperand(const std::string& name, uint32_t index,
                       const neuron::wrapper::OperandType& operand_type);
  std::string GetUniqueName(const std::string& base_name);

  void SetUseNCHW(bool use_nchw) { use_nchw_ = use_nchw; }
  bool UseNCHW() const { return use_nchw_; }
  NeuronRuntimeVersion GetNeuronVersion() const;

  // Layout preference: returns true if NHWC.
  bool UseNHWC() const { return !use_nchw_; }

  void SetUseFp16(bool use_fp16) { use_fp16_ = use_fp16; }
  void SetStoreDla(bool store_dla) { store_dla_ = store_dla; }
  // Directory holding the context .onnx; used to resolve a sidecar DLA file
  // referenced by an EPContext node with embed_mode=0.
  void SetEpContextDir(std::filesystem::path dir) { ep_context_dir_ = std::move(dir); }
  void SetDlaPath(const std::filesystem::path& dla_path) { dla_path_ = dla_path; }
  void SetExecutePreference(neuron::wrapper::ExecutePreference pref) { exe_pref_ = pref; }

  // Accessors.
  Shaper& GetShaper() { return shaper_; }
  const std::unordered_map<std::string, uint32_t>& GetOperandIndices() const { return operand_indices_; }
  const std::unordered_map<std::string, neuron::wrapper::OperandType>& GetOperandTypes() const { return operand_types_; }
  const std::unordered_set<std::string>& GetFusedActivations() const { return fused_activations_; }

  const NeuronGraphContext& GetContext() const { return ctx_; }

  // Return the initializer map from the graph context.
  const std::unordered_map<std::string, const onnx::TensorProto*>& GetInitializerTensors() const {
    return ctx_.initializers;
  }

  // Initializer access via graph context.
  const onnx::TensorProto* GetConstantInitializer(const std::string& name) const;

  bool HasDevice(const std::string& device_name) const;

  // Composite operations.
  NeuronStatus AddReshape(const std::string& input, const Shape& new_shape, const std::string& output);
  NeuronStatus AddExpandDims(const std::string& input, int32_t index_to_insert, const std::string& output);
  NeuronStatus AddCast(const std::string& input, wrapper::OperandType target_type, const std::string& output);
  NeuronStatus AddMul(const std::string& input, float value, const std::string& output);
  NeuronStatus AddQuantizeLinear(const std::string& input,
                                  const wrapper::OnnxDataTypeInfo& output_type_info,
                                  const std::string& output);
  NeuronStatus AddDequantizeLinear(const std::string& input,
                                    const wrapper::OnnxDataTypeInfo& input_type_info,
                                    const std::string& output);

 private:
  const NeuronApi& neuron_;
  const NeuronRuntimeVersion neuron_version_;
  const NeuronGraphContext& ctx_;
  std::unique_ptr<neuron::wrapper::Model> neuron_model_;

  uint32_t name_token_{0};

  bool use_nchw_{false};
  bool use_fp16_{false};
  bool store_dla_{false};
  std::filesystem::path dla_path_{};
  std::filesystem::path ep_context_dir_{};
  neuron::wrapper::ExecutePreference exe_pref_{
      neuron::wrapper::ExecutePreference::PREFER_FAST_SINGLE_ANSWER};

  Shaper shaper_;

  std::unordered_map<std::string, uint32_t> operand_indices_;
  std::unordered_map<std::string, neuron::wrapper::OperandType> operand_types_;

  std::unordered_set<std::string> operands_;
  std::unordered_set<std::string> fused_activations_;
  std::unordered_set<std::string> skipped_initializers_;
  std::unordered_set<std::string> skipped_inputs_;

  // Which node claimed each skip, recorded during PreprocessInitializers.
  // A skip is only honoured when *every* node that consumes the tensor claimed
  // it; see PruneUnsafeInitializerSkips.
  std::unordered_map<std::string, std::unordered_set<const onnx::NodeProto*>>
      initializer_skip_claims_;
  // Node currently being offered to AddInitializersToSkip (null outside of
  // PreprocessInitializers).
  const onnx::NodeProto* skip_claim_node_{nullptr};

  // Activation node map: output_name → fuse_code.
  std::unordered_map<std::string, int32_t> activation_output_map_;

  // Quantized input tracking.
  std::unordered_map<std::string, std::vector<std::string>> all_quantized_op_inputs_;

  std::vector<uint32_t> input_index_vec_;
  std::vector<uint32_t> output_index_vec_;

  std::unordered_set<std::string> unique_names_;

  gsl::span<const wrapper::Device> neuron_target_devices_;
  const neuron::wrapper::TargetDeviceOption target_device_option_;
  size_t num_neuron_ops_ = 0;
  uint32_t next_index_ = 0;

  // false when no target devices were resolved, so Compile() stays device-agnostic
  bool device_support_validated_ = false;

  std::string neuron_optimization_string_;

  NeuronStatus ComposeNeuronModel();
  void PreprocessInitializers();
  void PruneUnsafeInitializerSkips();
  void PreprocessActivations();
  NeuronStatus RegisterInitializers();
  NeuronStatus RegisterModelInputs(bool add_new_operand);
  NeuronStatus AddOperations();
  NeuronStatus RegisterModelOutputs();

  NeuronStatus SetOperandValue(uint32_t index, neuron::wrapper::Model::NNMemory* memory,
                                size_t size, size_t offset);
  NeuronStatus AddNewNeuronOperand(const neuron::wrapper::OperandType& type, uint32_t& index);
  NeuronStatus AddNewOperand(const std::string& name,
                              const neuron::wrapper::OperandType& operand_type,
                              uint32_t& index);

  static const IOpBuilder* GetOpBuilder(const std::string& op_type);
};

}  // namespace neuron
}  // namespace onnxruntime
