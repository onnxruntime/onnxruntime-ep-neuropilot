# onnxruntime-ep-neuropilot

The **MediaTek Neuron plugin execution provider** for ONNX Runtime. Builds
`libonnxruntime_providers_neuropilot.so`, a plugin EP that ORT loads at runtime to offload
supported subgraphs to the Neuron NPU.

The per-operator support-check logic is consumed as a prebuilt closed-source shared
library, `libneuron_op_capability.so`, which this EP `dlopen`s at runtime. The prebuilt
library is available from the GitHub release page.

---

## Prerequisites

- The prebuilt **`libneuron_op_capability.so`**, downloadable from the GitHub release page.
  This EP **requires** it at configure time.
- ORT public headers (the directory containing `onnxruntime_cxx_api.h`,
  e.g. `<ort-src>/include/onnxruntime/core/session`). No `libonnxruntime.so` is needed — the
  EP is `dlopen`ed by ORT, so its symbols resolve at runtime.
- protobuf / ONNX / GSL are fetched automatically.

---

## Build

Download the prebuilt `libneuron_op_capability.so` for your target platform from the GitHub
release page — it is **required** at configure time.

```bash
cmake -S . -B build \
  -DORT_INCLUDE_DIR=/path/to/onnxruntime/include/onnxruntime/core/session \
  -DNEURON_OP_CAPABILITY_LIB_DIR=/dir/containing/libneuron_op_capability.so
cmake --build build -j
```

Key CMake options:

| Option | Meaning |
|---|---|
| `ORT_INCLUDE_DIR` | dir with `onnxruntime_cxx_api.h` (required) |
| `NEURON_OP_CAPABILITY_LIB_DIR` | dir containing the prebuilt `libneuron_op_capability.so` (**required**) |
| `ORT_LIBRARY_DIR` | optional; enables `--no-undefined` link check |
| `ORTNeuronEp_INSTALL` | install targets; also stages the `.so` beside the EP (default ON) |

If the `.so` is missing, configuration aborts with:

```
CMake Error: libneuron_op_capability.so is REQUIRED but was not found.
  Download it from the GitHub release page, then re-configure with:
    -DNEURON_OP_CAPABILITY_LIB_DIR=/dir/containing/libneuron_op_capability.so
```

---

## Deploy & use

Place both libraries in the same directory (or on the loader's search path):

```
libonnxruntime_providers_neuropilot.so
libneuron_op_capability.so
```

Register the EP with ORT (`register_execution_provider_library` / the plugin EP API). On
init the EP `dlopen`s `libneuron_op_capability.so`; if it is absent, EP creation fails with a
clear error.

---

## Repository Layout

```
onnxruntime-ep-neuropilot/
├── CMakeLists.txt                         standalone EP build (FetchContent for protobuf/ONNX/GSL)
├── Dockerfile / docker-build.sh / docker-entrypoint.sh   aarch64 cross-build
├── cmake/                                 toolchain + package config
└── src/neuron_plugin_ep/
    ├── neuron_plugin_ep.cc                DLL entry (CreateEpFactories); eagerly loads the .so
    ├── neuron_ep*.{cc,h}                  OrtEp plugin shell
    └── neuron_impl/
        ├── op_capability_abi/             ← bridge (loader + abstract interface)
        ├── builders/                      op-builders (AddToModelBuilder); NO IsOpSupported logic
        ├── neuron_lib/                    Neuron Adapter wrappers
        ├── neuron_graph_context.{cc,h}    graph/context accessors
        └── ...
```
