// graph_payload_elision.cc

#include "graph_payload_elision.h"

#include <utility>

namespace onnxruntime {
namespace neuron {
namespace op_capability {

size_t InlinePayloadBytes(const onnx::TensorProto& tensor) {
  if (tensor.data_location() == onnx::TensorProto::EXTERNAL) return 0;

  size_t bytes = tensor.raw_data().size();
  bytes += static_cast<size_t>(tensor.float_data_size()) * sizeof(float);
  bytes += static_cast<size_t>(tensor.int32_data_size()) * sizeof(int32_t);
  bytes += static_cast<size_t>(tensor.int64_data_size()) * sizeof(int64_t);
  bytes += static_cast<size_t>(tensor.double_data_size()) * sizeof(double);
  bytes += static_cast<size_t>(tensor.uint64_data_size()) * sizeof(uint64_t);
  return bytes;
}

// ---------------------------------------------------------------------------
// ElidedInitializerPayloads
// ---------------------------------------------------------------------------

ElidedInitializerPayloads::ElidedInitializerPayloads(onnx::GraphProto& graph,
                                                     size_t max_inline_bytes)
    : max_inline_bytes_(max_inline_bytes) {
  VisitGraph(graph);
}

ElidedInitializerPayloads::~ElidedInitializerPayloads() {
  // Restore in reverse order of elision.
  for (auto it = saved_.rbegin(); it != saved_.rend(); ++it) {
    Saved& s = **it;
    onnx::TensorProto& t = *s.tensor;

    // We appended exactly one external_data entry; remove it.
    t.mutable_external_data()->RemoveLast();

    if (s.had_data_location) {
      t.set_data_location(s.data_location);
    } else {
      t.clear_data_location();
    }

    if (s.had_raw_data) {
      t.mutable_raw_data()->swap(s.raw_data);
    }
    t.mutable_float_data()->Swap(&s.float_data);
    t.mutable_int32_data()->Swap(&s.int32_data);
    t.mutable_int64_data()->Swap(&s.int64_data);
    t.mutable_double_data()->Swap(&s.double_data);
    t.mutable_uint64_data()->Swap(&s.uint64_data);
  }
}

void ElidedInitializerPayloads::VisitTensor(onnx::TensorProto& tensor) {
  // Genuinely external tensors must keep their location.
  if (tensor.data_location() == onnx::TensorProto::EXTERNAL) return;

  const size_t bytes = InlinePayloadBytes(tensor);
  if (bytes <= max_inline_bytes_) return;

  auto saved = std::make_unique<Saved>();
  saved->tensor = &tensor;
  saved->had_data_location = tensor.has_data_location();
  saved->data_location = tensor.data_location();
  saved->had_raw_data = tensor.has_raw_data();

  // All swaps below are pointer swaps; nothing copies the payload.
  if (saved->had_raw_data) {
    saved->raw_data.swap(*tensor.mutable_raw_data());
    // Also clear the presence bit.
    tensor.clear_raw_data();
  }
  tensor.mutable_float_data()->Swap(&saved->float_data);
  tensor.mutable_int32_data()->Swap(&saved->int32_data);
  tensor.mutable_int64_data()->Swap(&saved->int64_data);
  tensor.mutable_double_data()->Swap(&saved->double_data);
  tensor.mutable_uint64_data()->Swap(&saved->uint64_data);

  tensor.set_data_location(onnx::TensorProto::EXTERNAL);
  auto* entry = tensor.add_external_data();
  entry->set_key(kElidedLocationKey);
  entry->set_value(NEURON_CAP_ELIDED_LOCATION);

  elided_bytes_ += bytes;
  saved_.push_back(std::move(saved));
}

void ElidedInitializerPayloads::VisitGraph(onnx::GraphProto& graph) {
  for (auto& init : *graph.mutable_initializer()) {
    VisitTensor(init);
  }

  for (auto& sparse : *graph.mutable_sparse_initializer()) {
    VisitTensor(*sparse.mutable_values());
    VisitTensor(*sparse.mutable_indices());
  }

  for (auto& node : *graph.mutable_node()) {
    for (auto& attr : *node.mutable_attribute()) {
      // Constant nodes carry tensors as attributes.
      if (attr.has_t()) VisitTensor(*attr.mutable_t());
      for (auto& t : *attr.mutable_tensors()) VisitTensor(t);

      if (attr.has_sparse_tensor()) {
        VisitTensor(*attr.mutable_sparse_tensor()->mutable_values());
        VisitTensor(*attr.mutable_sparse_tensor()->mutable_indices());
      }
      for (auto& st : *attr.mutable_sparse_tensors()) {
        VisitTensor(*st.mutable_values());
        VisitTensor(*st.mutable_indices());
      }

      // Subgraph bodies (If/Loop/Scan) hold their own initializers.
      if (attr.has_g()) VisitGraph(*attr.mutable_g());
      for (auto& g : *attr.mutable_graphs()) VisitGraph(g);
    }
  }
}

}  // namespace op_capability
}  // namespace neuron
}  // namespace onnxruntime
