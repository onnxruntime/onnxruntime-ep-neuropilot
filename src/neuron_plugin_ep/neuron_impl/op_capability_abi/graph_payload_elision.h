// graph_payload_elision.h
//
// Temporarily strips bulk tensor payloads out of a GraphProto so it can be
// serialized for the op capability library without copying the model's weights.
// Done in place and undone by the destructor; protobuf swaps make each strip
// and restore O(1) regardless of payload size.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <onnx/onnx_pb.h>

#include "neuron_op_capability_abi.h"

namespace onnxruntime {
namespace neuron {
namespace op_capability {

// ONNX's reserved external_data key for the data's location.
constexpr const char kElidedLocationKey[] = "location";

// Bytes `tensor` stores inline, across raw_data and the typed repeated fields;
// 0 for external tensors. string_data is not counted (never elided).
size_t InlinePayloadBytes(const onnx::TensorProto& tensor);

// True if `tensor` was marked by ElidedInitializerPayloads: its metadata is
// meaningful but its values are absent.
inline bool IsElidedPayload(const onnx::TensorProto& tensor) {
  if (tensor.data_location() != onnx::TensorProto::EXTERNAL) return false;
  for (const auto& entry : tensor.external_data()) {
    if (entry.key() == kElidedLocationKey && entry.value() == NEURON_CAP_ELIDED_LOCATION) {
      return true;
    }
  }
  return false;
}

// Scoped, in-place removal of every large tensor payload reachable from a
// GraphProto: initializers, sparse initializers, tensor-valued attributes, and
// subgraphs. While an instance is alive, `graph` is only fit for serialization;
// the destructor restores it exactly. Not thread-safe with respect to `graph`.
class ElidedInitializerPayloads {
 public:
  explicit ElidedInitializerPayloads(
      onnx::GraphProto& graph,
      size_t max_inline_bytes = NEURON_CAP_MAX_INLINE_INITIALIZER_BYTES);

  ~ElidedInitializerPayloads();

  ElidedInitializerPayloads(const ElidedInitializerPayloads&) = delete;
  ElidedInitializerPayloads& operator=(const ElidedInitializerPayloads&) = delete;

  // How many tensors were stripped, and how many bytes they held.
  size_t elided_count() const { return saved_.size(); }
  size_t elided_bytes() const { return elided_bytes_; }

 private:
  // A stripped tensor's payload, parked until the destructor restores it.
  struct Saved {
    onnx::TensorProto* tensor = nullptr;

    bool had_data_location = false;
    onnx::TensorProto::DataLocation data_location = onnx::TensorProto::DEFAULT;

    bool had_raw_data = false;
    std::string raw_data;

    google::protobuf::RepeatedField<float> float_data;
    google::protobuf::RepeatedField<int32_t> int32_data;
    google::protobuf::RepeatedField<int64_t> int64_data;
    google::protobuf::RepeatedField<double> double_data;
    google::protobuf::RepeatedField<uint64_t> uint64_data;
  };

  void VisitGraph(onnx::GraphProto& graph);
  void VisitTensor(onnx::TensorProto& tensor);

  const size_t max_inline_bytes_;
  size_t elided_bytes_ = 0;

  // unique_ptr keeps the `tensor` pointers in Saved stable as the vector grows.
  std::vector<std::unique_ptr<Saved>> saved_;
};

}  // namespace op_capability
}  // namespace neuron
}  // namespace onnxruntime
