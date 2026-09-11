// graph_payload_elision_test.cc
//
// Checks that elision bounds the serialized graph size and that the graph is
// restored byte-for-byte afterwards.

#include "op_capability_abi/graph_payload_elision.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using onnxruntime::neuron::op_capability::ElidedInitializerPayloads;
using onnxruntime::neuron::op_capability::InlinePayloadBytes;
using onnxruntime::neuron::op_capability::IsElidedPayload;

int g_failures = 0;

void Check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++g_failures;
  }
}

constexpr size_t kBigBytes = 1u << 20;  // 1 MiB, comfortably over the threshold

onnx::TensorProto* AddRawInitializer(onnx::GraphProto& graph, const std::string& name,
                                     size_t bytes) {
  auto* t = graph.add_initializer();
  t->set_name(name);
  t->set_data_type(onnx::TensorProto::FLOAT);
  t->add_dims(static_cast<int64_t>(bytes / sizeof(float)));
  t->set_raw_data(std::string(bytes, '\x7f'));
  return t;
}

// ---------------------------------------------------------------------------

void TestSerializedSizeIsBounded() {
  onnx::GraphProto graph;
  for (int i = 0; i < 8; ++i) {
    AddRawInitializer(graph, "w" + std::to_string(i), kBigBytes);
  }

  std::string before;
  Check(graph.SerializeToString(&before), "serialize before elision");
  Check(before.size() > 8 * kBigBytes - 4096, "unelided graph should carry the weights");

  std::string after;
  {
    const ElidedInitializerPayloads elided(graph);
    Check(graph.SerializeToString(&after), "serialize with elision");
    Check(elided.elided_count() == 8, "all eight initializers elided");
    Check(elided.elided_bytes() == 8 * kBigBytes, "elided byte count");
  }

  // Structure only: names, dims, types and the markers.
  Check(after.size() < 4096, "elided graph must not scale with the weights");
}

void TestRestoresExactly() {
  onnx::GraphProto graph;
  AddRawInitializer(graph, "raw_big", kBigBytes);

  // A typed-field tensor, to cover the non-raw_data path.
  auto* typed = graph.add_initializer();
  typed->set_name("typed_big");
  typed->set_data_type(onnx::TensorProto::INT64);
  typed->add_dims(4096);
  for (int i = 0; i < 4096; ++i) typed->add_int64_data(i);

  // A tensor that genuinely lives in an external file must be left alone.
  auto* external = graph.add_initializer();
  external->set_name("external");
  external->set_data_type(onnx::TensorProto::FLOAT);
  external->add_dims(1024);
  external->set_data_location(onnx::TensorProto::EXTERNAL);
  auto* loc = external->add_external_data();
  loc->set_key("location");
  loc->set_value("weights.bin");

  std::string before;
  Check(graph.SerializeToString(&before), "serialize before");

  {
    const ElidedInitializerPayloads elided(graph);
    Check(elided.elided_count() == 2, "external tensor must not be elided");
    Check(graph.initializer(2).external_data(0).value() == "weights.bin",
          "external tensor's location preserved during elision");
  }

  std::string after;
  Check(graph.SerializeToString(&after), "serialize after");
  Check(before == after, "graph must be restored byte-for-byte");
}

void TestSmallTensorsUntouched() {
  onnx::GraphProto graph;
  auto* shape = graph.add_initializer();
  shape->set_name("shape");
  shape->set_data_type(onnx::TensorProto::INT64);
  shape->add_dims(4);
  for (int64_t d : {1, 3, 224, 224}) shape->add_int64_data(d);

  {
    const ElidedInitializerPayloads elided(graph);
    Check(elided.elided_count() == 0, "small tensor must not be elided");
    Check(!IsElidedPayload(graph.initializer(0)), "small tensor unmarked");
    Check(graph.initializer(0).int64_data_size() == 4, "small tensor keeps its values");
  }
}

// An elided tensor keeps its name, data type and dims.
void TestMetadataSurvivesAndIsMarked() {
  onnx::GraphProto graph;
  AddRawInitializer(graph, "w", kBigBytes);

  {
    const ElidedInitializerPayloads elided(graph);
    const auto& t = graph.initializer(0);
    Check(IsElidedPayload(t), "elided tensor is marked");
    Check(t.name() == "w", "name survives");
    Check(t.data_type() == onnx::TensorProto::FLOAT, "data_type survives");
    Check(t.dims_size() == 1 && t.dims(0) == kBigBytes / sizeof(float), "dims survive");
    Check(t.raw_data().empty(), "payload is gone");
    Check(InlinePayloadBytes(t) == 0, "elided tensor reports no inline payload");
  }
}

void TestAttributesAndSubgraphs() {
  onnx::GraphProto graph;

  auto* constant = graph.add_node();
  constant->set_op_type("Constant");
  auto* value = constant->add_attribute();
  value->set_name("value");
  value->set_type(onnx::AttributeProto::TENSOR);
  auto* t = value->mutable_t();
  t->set_data_type(onnx::TensorProto::FLOAT);
  t->add_dims(static_cast<int64_t>(kBigBytes / sizeof(float)));
  t->set_raw_data(std::string(kBigBytes, '\x01'));

  auto* loop = graph.add_node();
  loop->set_op_type("Loop");
  auto* body = loop->add_attribute();
  body->set_name("body");
  body->set_type(onnx::AttributeProto::GRAPH);
  AddRawInitializer(*body->mutable_g(), "body_w", kBigBytes);

  std::string before;
  Check(graph.SerializeToString(&before), "serialize before");

  {
    const ElidedInitializerPayloads elided(graph);
    Check(elided.elided_count() == 2, "attribute and subgraph tensors elided");

    std::string after;
    Check(graph.SerializeToString(&after), "serialize with elision");
    Check(after.size() < 4096, "attribute/subgraph payloads excluded");
  }

  std::string restored;
  Check(graph.SerializeToString(&restored), "serialize after");
  Check(before == restored, "attribute and subgraph tensors restored");
}

}  // namespace

int main() {
  TestSerializedSizeIsBounded();
  TestRestoresExactly();
  TestSmallTensorsUntouched();
  TestMetadataSurvivesAndIsMarked();
  TestAttributesAndSubgraphs();

  if (g_failures != 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("All graph payload elision checks passed.\n");
  return 0;
}
