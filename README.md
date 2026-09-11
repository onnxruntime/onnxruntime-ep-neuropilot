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

## Target platforms

The same sources build for two targets; the platform-specific code is confined to
`neuron_impl/neuron_lib/`, behind `__ANDROID__`:

| Target | libc | Toolchain file | Extra link |
|---|---|---|---|
| aarch64 Linux | glibc | `cmake/toolchain-aarch64.cmake` | — |
| Android arm64-v8a | bionic | `cmake/toolchain-android.cmake` | `liblog` (added automatically when `ANDROID`) |

`libneuron_op_capability.so` must be built for the **same platform** — it is loaded into the
same process. Mixing them fails at `dlopen` time on the device, not at build time.

## Build

Download the prebuilt `libneuron_op_capability.so` for your target platform from the GitHub
release page — it is **required** at configure time.

```bash
cmake -S . -B build \
  -DORT_INCLUDE_DIR=/path/to/onnxruntime/include/onnxruntime/core/session \
  -DNEURON_OP_CAPABILITY_LIB_DIR=/dir/containing/libneuron_op_capability.so
cmake --build build -j
```

Cross-compiling — add the toolchain file for the target. Codegen needs a
**host** protoc (a target-architecture one cannot run on the build machine); it is resolved
automatically.

```bash
# aarch64 Linux
cmake -S . -B build-aarch64 -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-aarch64.cmake \
  -DORT_INCLUDE_DIR=... -DNEURON_OP_CAPABILITY_LIB_DIR=...

# Android
cmake -S . -B build-android -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-android.cmake \
  -DANDROID_NDK_HOME=$HOME/Android/Sdk/ndk/27.2.12479018 \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-29 \
  -DORT_INCLUDE_DIR=... -DNEURON_OP_CAPABILITY_LIB_DIR=...
```

A `protoc` on `PATH` is used when it is exactly the version of the
protobuf runtime this library links — otherwise a matching host protoc is built from source.
Pin a specific one with `-DPROTOBUF_PROTOC_EXECUTABLE=/path/to/protoc`; a version mismatch
will result in an error.

Key CMake options:

| Option | Meaning |
|---|---|
| `ORT_INCLUDE_DIR` | dir with `onnxruntime_cxx_api.h` (required) |
| `NEURON_OP_CAPABILITY_LIB_DIR` | dir containing the prebuilt `libneuron_op_capability.so` (**required**) |
| `ORT_LIBRARY_DIR` | optional; enables `--no-undefined` link check |
| `ORTNeuronEp_INSTALL` | install targets; also stages the `.so` beside the EP (default ON) |
| `ANDROID_NDK_HOME` / `ANDROID_ABI` / `ANDROID_PLATFORM` | Android only; read by `cmake/toolchain-android.cmake` (defaults `arm64-v8a` / `android-29`) |

If the `.so` is missing, configuration aborts with:

```
CMake Error: libneuron_op_capability.so is REQUIRED but was not found.
  Download it from the GitHub release page, then re-configure with:
    -DNEURON_OP_CAPABILITY_LIB_DIR=/dir/containing/libneuron_op_capability.so
```

---

## Deploy & use

Place both libraries — built for the same platform — in the same directory (or on the
loader's search path):

```
libonnxruntime_providers_neuropilot.so
libneuron_op_capability.so
```

Register the EP with ORT (`register_execution_provider_library` / the plugin EP API). On
init the EP `dlopen`s `libneuron_op_capability.so`; if it is absent, EP creation fails with a
clear error.

On Android, push both plus an Android `libonnxruntime.so` to the device and set
`LD_LIBRARY_PATH` to that directory; the EP resolves `libneuron_op_capability.so` **by name**
through the normal loader search path. Real NPU execution also needs the on-device MediaTek
Neuron runtime (`libneuronusdk_adapter.mtk.so`).

The NDK builds with `-g`, so the Android `.so` files are tens of MB. Strip before shipping:

```bash
$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip \
  --strip-unneeded build-android/libonnxruntime_providers_neuropilot.so
```

---

## Repository Layout

```
onnxruntime-ep-neuropilot/
├── CMakeLists.txt                         standalone EP build (FetchContent for protobuf/ONNX/GSL)
├── cmake/
│   ├── toolchain-aarch64.cmake            aarch64 Linux cross-build (glibc, gcc)
│   ├── toolchain-android.cmake            Android cross-build (bionic, NDK clang)
│   └── ORTNeuronEpConfig*.cmake.in        package config
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
