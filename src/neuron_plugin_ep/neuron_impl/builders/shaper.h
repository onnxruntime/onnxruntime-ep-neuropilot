// shaper.h
// Plugin EP version – uses NeuronGraphContext instead of GraphViewer.

#pragma once

#include <string>
#include <unordered_map>

#include "../neuron_graph_context.h"
#include "../neuron_lib/neuron_wrapper_operand.h"

namespace onnxruntime {
namespace neuron {

// Neuron operations are limited to rank 4, but some modules
// work on rank-5 tensors whose extra dimensions are a leading
// unit batch.
inline Shape SqueezeLeadingUnitDims(Shape shape) {
  while (shape.size() > 4 && shape.front() == 1)
    shape.erase(shape.begin());
  return shape;
}

class Shaper {
 public:
  explicit Shaper(const NeuronGraphContext& ctx) : ctx_(&ctx) {}

  void AddShape(const std::string& name, const Shape& shape) {
    shape_map_[name] = SqueezeLeadingUnitDims(shape);
  }

  inline Shape operator[](const std::string& key) const {
    auto it = shape_map_.find(key);
    if (it != shape_map_.end()) {
      return it->second;
    }
    // Fall back to graph type information.
    return SqueezeLeadingUnitDims(ctx_->GetShape(key));
  }

 private:
  std::unordered_map<std::string, Shape> shape_map_;
  const NeuronGraphContext* ctx_;
};

}  // namespace neuron
}  // namespace onnxruntime
