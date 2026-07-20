// shaper.h
// Plugin EP version – uses NeuronGraphContext instead of GraphViewer.

#pragma once

#include <string>
#include <unordered_map>

#include "../neuron_graph_context.h"
#include "../neuron_lib/neuron_wrapper_operand.h"

namespace onnxruntime {
namespace neuron {

class Shaper {
 public:
  explicit Shaper(const NeuronGraphContext& ctx) : ctx_(&ctx) {}

  void AddShape(const std::string& name, const Shape& shape) {
    shape_map_[name] = shape;
  }

  inline Shape operator[](const std::string& key) const {
    auto it = shape_map_.find(key);
    if (it != shape_map_.end()) {
      return it->second;
    }
    // Fall back to graph type information.
    return ctx_->GetShape(key);
  }

 private:
  std::unordered_map<std::string, Shape> shape_map_;
  const NeuronGraphContext* ctx_;
};

}  // namespace neuron
}  // namespace onnxruntime
