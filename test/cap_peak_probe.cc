// cap_peak_probe.cc
//
// Reports resident/peak RSS while handing a model's GraphProto to the op
// capability library via CapabilityGraphHandle.
//
//   LD_LIBRARY_PATH=<dir with libneuron_op_capability.so> ./cap_peak_probe m.onnx
//
// Models with ONNX external-data sidecars are not supported.

#include <sys/resource.h>

#include <cstdio>
#include <fstream>
#include <limits>
#include <string>

#include <onnx/onnx_pb.h>

#include "op_capability_abi/op_capability_loader.h"

namespace {

long PeakResidentMiB() {
  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
  return usage.ru_maxrss / 1024;
}

long CurrentResidentMiB() {
  std::ifstream status("/proc/self/status");
  std::string key;
  while (status >> key) {
    if (key == "VmRSS:") {
      long kib = 0;
      status >> kib;
      return kib / 1024;
    }
    status.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
  return -1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <model.onnx>\n", argv[0]);
    return 2;
  }

  onnx::ModelProto model;
  {
    std::ifstream in(argv[1], std::ios::binary);
    if (!in || !model.ParseFromIstream(&in)) {
      std::fprintf(stderr, "failed to parse %s\n", argv[1]);
      return 1;
    }
  }

  const long parsed_rss = CurrentResidentMiB();
  std::printf("model parsed : %ld MiB resident, %ld MiB peak\n", parsed_rss,
              PeakResidentMiB());

  {
    const onnxruntime::neuron::op_capability::CapabilityGraphHandle handle(
        *model.mutable_graph());
    std::printf("handle open  : %ld MiB resident, %ld MiB peak (valid=%d)\n",
                CurrentResidentMiB(), PeakResidentMiB(), handle.valid());
    if (!handle.valid()) {
      std::fprintf(stderr,
                   "capability library did not take the graph; is "
                   "libneuron_op_capability.so on LD_LIBRARY_PATH?\n");
    }
  }

  std::printf("handle closed: %ld MiB resident, %ld MiB peak\n",
              CurrentResidentMiB(), PeakResidentMiB());
  std::printf("graph cost   : %ld MiB above the parsed model\n",
              PeakResidentMiB() - parsed_rss);
  return 0;
}
