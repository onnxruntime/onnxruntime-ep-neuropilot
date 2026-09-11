# onnxruntime-ep-neuron (Plugin EP) — Standalone build
#
# Does NOT require ORT_SOURCE_DIR.
# Does NOT use the path shim (symlink hack).
# Uses FetchContent for protobuf, ONNX, and GSL.
# Sources are globbed from src/neuron_plugin_ep/**/*.cc.
# Target name: onnxruntime_providers_neuron
#
# Required CMake variables:
#   ORT_INCLUDE_DIR  — directory containing onnxruntime_cxx_api.h etc.
#   ORT_LIBRARY_DIR  — directory containing libonnxruntime.so
#
# Optional:
#   ORT_HOME         — ORT install root (lib/ and include/ derived automatically)
#   PROTOC_BIN       — path to a HOST-architecture protoc binary
#                      (Yocto: ${STAGING_BINDIR_NATIVE}/protoc)
#   ORTNeuronEp_OBJECT_CACHE — enable ccache/sccache (default ON)
#   ORTNeuronEp_INSTALL      — install targets (default ON)
#
# Example (native build):
#   cmake -DORT_INCLUDE_DIR=... -DORT_LIBRARY_DIR=... -S . -B build
#
# Example (Yocto cross-compile):
#   cmake -DORT_INCLUDE_DIR=${STAGING_INCDIR}/onnxruntime \
#         -DORT_LIBRARY_DIR=${STAGING_LIBDIR}             \
#         -DPROTOC_BIN=${STAGING_BINDIR_NATIVE}/protoc    \
#         -S . -B build
#
# Example (aarch64 Linux cross-compile):
#   cmake -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-aarch64.cmake \
#         -DORT_INCLUDE_DIR=... -DNEURON_OP_CAPABILITY_LIB_DIR=... \
#         -S . -B build-aarch64
#
# Example (Android cross-compile, NDK):
#   cmake -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-android.cmake \
#         -DANDROID_NDK_HOME=/path/to/android-ndk               \
#         -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-29 \
#         -DORT_INCLUDE_DIR=... -DNEURON_OP_CAPABILITY_LIB_DIR=... \
#         -S . -B build-android
#   The same sources serve both targets; the Android-only code paths are behind
#   __ANDROID__ in src/neuron_plugin_ep/neuron_impl/neuron_lib/.

cmake_minimum_required(VERSION 3.25)
cmake_policy(SET CMP0091 NEW)
cmake_policy(SET CMP0141 NEW)
project(ORTNeuronEp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  message(WARNING "Build type not set, falling back to RelWithDebInfo.")
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
      "Choose build configuration. [Debug | Release | RelWithDebInfo]" FORCE)
endif()

option(ORTNeuronEp_INSTALL           "Install ORTNeuronEp binaries and files" ON)
option(ORTNeuronEp_OBJECT_CACHE      "Enable object caching via sccache/ccache" ON)
# WIP: Python helper package (onnxruntime_ep_neuron). Disabled until the
# pip packaging and Yocto recipe wiring are completed.
option(ORTNeuronEp_BUILD_PYTHON      "Install Python helper package for the plugin EP" OFF)

# Directory containing the prebuilt libneuron_op_capability.so.
# The per-op capability logic ships as a prebuilt closed-source library;
# this EP requires that .so to be present (dlopen'd at runtime).
set(NEURON_OP_CAPABILITY_LIB_DIR "" CACHE PATH
    "Directory containing the prebuilt libneuron_op_capability.so (REQUIRED)")

# ---------------------------------------------------------------------------
# Compiler cache (ccache / sccache)
# ---------------------------------------------------------------------------
if(ORTNeuronEp_OBJECT_CACHE)
  find_program(CCACHE_PROGRAM  ccache)
  find_program(SCCACHE_PROGRAM sccache)
  if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER   "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  elseif(SCCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER   "${SCCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}")
  endif()
endif()

include(FetchContent)
include("${CMAKE_SOURCE_DIR}/cmake/onnxruntime_library_utils.cmake")

# Minimum supported ONNX Runtime release.  The patch level cannot be seen from
# the headers (ORT_API_VERSION only tracks the minor), so it is enforced on the
# version we download and documented for user-supplied ORT trees.
set(ORTNeuronEp_MIN_ORT_VERSION "1.24.1")
string(REGEX REPLACE "^[0-9]+\\.([0-9]+)\\..*$" "\\1" ORTNeuronEp_MIN_ORT_API_VERSION
       "${ORTNeuronEp_MIN_ORT_VERSION}")

# ---------------------------------------------------------------------------
# ORT include / library paths
# ---------------------------------------------------------------------------
set(ORTNeuronEp_DEFAULT_ORT_VERSION "1.26.0" CACHE STRING
    "ORT version downloaded when ORT_HOME is given (must be >= ORTNeuronEp_MIN_ORT_VERSION)")
# ORT_LIBRARY_DIR is optional: the plugin EP is loaded at runtime by ORT
# (via dlopen), so libonnxruntime.so is already in memory and its symbols
# are available without link-time linkage.  Providing ORT_LIBRARY_DIR lets
# the linker enforce --no-undefined; omitting it uses --allow-shlib-undefined.
if(DEFINED ORT_INCLUDE_DIR)
  if(NOT IS_DIRECTORY "${ORT_INCLUDE_DIR}")
    message(FATAL_ERROR "ORT_INCLUDE_DIR='${ORT_INCLUDE_DIR}' is not a directory.")
  endif()
  if(DEFINED ORT_LIBRARY_DIR AND NOT IS_DIRECTORY "${ORT_LIBRARY_DIR}")
    message(FATAL_ERROR "ORT_LIBRARY_DIR='${ORT_LIBRARY_DIR}' is not a directory.")
  endif()
  message(STATUS "Using explicit ORT include dir (standalone build)")
elseif(DEFINED ORT_HOME AND ORT_HOME)
  if(ORTNeuronEp_DEFAULT_ORT_VERSION VERSION_LESS ORTNeuronEp_MIN_ORT_VERSION)
    message(FATAL_ERROR "ORTNeuronEp_DEFAULT_ORT_VERSION (${ORTNeuronEp_DEFAULT_ORT_VERSION}) "
                        "is older than the minimum supported ORT ${ORTNeuronEp_MIN_ORT_VERSION}.")
  endif()
  set_onnxruntime_paths(
    ORT_HOME            "${ORT_HOME}"
    DEFAULT_ORT_VERSION "${ORTNeuronEp_DEFAULT_ORT_VERSION}"
    ORT_INCLUDE_DIR_VAR ORT_INCLUDE_DIR
    ORT_LIBRARY_DIR_VAR ORT_LIBRARY_DIR
  )
else()
  message(FATAL_ERROR "Set ORT_INCLUDE_DIR (required) and optionally ORT_LIBRARY_DIR.")
endif()

# Reject headers older than the minimum ORT release.
set(_ort_c_api_header "${ORT_INCLUDE_DIR}/onnxruntime_c_api.h")
if(NOT EXISTS "${_ort_c_api_header}")
  message(FATAL_ERROR "onnxruntime_c_api.h not found in ORT_INCLUDE_DIR='${ORT_INCLUDE_DIR}'.")
endif()
file(STRINGS "${_ort_c_api_header}" _ort_api_line REGEX "^#define ORT_API_VERSION [0-9]+")
string(REGEX REPLACE "^#define ORT_API_VERSION ([0-9]+).*" "\\1" ORT_HEADER_API_VERSION "${_ort_api_line}")
if(NOT ORT_HEADER_API_VERSION OR ORT_HEADER_API_VERSION LESS ORTNeuronEp_MIN_ORT_API_VERSION)
  message(FATAL_ERROR "ORT headers in '${ORT_INCLUDE_DIR}' have ORT_API_VERSION='${ORT_HEADER_API_VERSION}'; "
                      "ONNX Runtime >= ${ORTNeuronEp_MIN_ORT_VERSION} "
                      "(API version ${ORTNeuronEp_MIN_ORT_API_VERSION}) is required.")
endif()
message(STATUS "ORT_INCLUDE_DIR : ${ORT_INCLUDE_DIR}")
message(STATUS "ORT_LIBRARY_DIR : ${ORT_LIBRARY_DIR}")

# Locate libonnxruntime.so (optional — only needed for --no-undefined)
set(ORT_LIB "")
if(DEFINED ORT_LIBRARY_DIR AND ORT_LIBRARY_DIR)
  if(WIN32)
    set(ORT_LIB "${ORT_LIBRARY_DIR}/onnxruntime.lib")
  else()
    find_library(ORT_LIB
      NAMES onnxruntime
      HINTS "${ORT_LIBRARY_DIR}"
      NO_DEFAULT_PATH
      NO_CMAKE_FIND_ROOT_PATH
    )
    if(NOT ORT_LIB)
      message(WARNING "libonnxruntime not found under ORT_LIBRARY_DIR='${ORT_LIBRARY_DIR}' — building with --allow-shlib-undefined")
    endif()
  endif()
endif()
message(STATUS "ORT_LIB         : ${ORT_LIB} (empty = runtime-resolved)")

# ---------------------------------------------------------------------------
# FetchContent dependencies (protobuf + ONNX + GSL)
# ---------------------------------------------------------------------------

# --- GSL ---
set(GSL_INSTALL OFF CACHE INTERNAL "")
FetchContent_Declare(gsl
  GIT_REPOSITORY https://github.com/microsoft/GSL.git
  GIT_TAG        v4.0.0
)
FetchContent_MakeAvailable(gsl)

# --- protobuf ---
if(WIN32)
  set(BUILD_SHARED_LIBS OFF)
endif()
set(protobuf_BUILD_TESTS      OFF CACHE INTERNAL "")
set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE INTERNAL "")
set(PROTOBUF_EXPECTED_VERSION "3.21.12")   # == the v21.12 tag below
FetchContent_Declare(protobuf
  GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
  GIT_TAG        v21.12
)
FetchContent_MakeAvailable(protobuf)

# Cross-compilation fix: inject an IMPORTED protobuf::protoc target pointing
# to the host binary.  ONNX's cmake uses $<TARGET_FILE:protobuf::protoc> in
# add_custom_command, so the target must exist even though we don't build an
# aarch64 protoc (protobuf_BUILD_PROTOC_BINARIES=OFF in toolchain).
#
# protoc is a build-time code generator, not a link-time dependency, but the
# generated sources must compile against the runtime we link: a protoc from
# protobuf >= 26 emits .pb.h files that #include
# "google/protobuf/runtime_version.h", a header that does not exist in 21.12.
# So a host protoc is usable only if its version matches exactly; otherwise one
# is built for the HOST from the same protobuf tree FetchContent checked out.
if(CMAKE_CROSSCOMPILING AND NOT TARGET protobuf::protoc)
  # -DPROTOBUF_PROTOC_EXECUTABLE pins one explicitly, and a mismatch is then an
  # error instead of a silent source build. Otherwise take it from PATH: the
  # toolchain sets CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER, but be explicit that
  # this has to run on the BUILD machine. NO_CACHE keeps the search result out
  # of the cache, where it would look like an explicit pin on the next configure.
  set(_protoc_pinned "${PROTOBUF_PROTOC_EXECUTABLE}")
  if(NOT _protoc_pinned)
    find_program(PROTOBUF_PROTOC_EXECUTABLE protoc NO_CMAKE_FIND_ROOT_PATH NO_CACHE)
  endif()

  set(_protoc_id "no usable protoc found")
  set(_protoc_version "")
  if(PROTOBUF_PROTOC_EXECUTABLE)
    execute_process(
      COMMAND "${PROTOBUF_PROTOC_EXECUTABLE}" --version
      OUTPUT_VARIABLE _protoc_out
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
    if(_protoc_out MATCHES "libprotoc ([0-9]+(\\.[0-9]+)+)")
      set(_protoc_version "${CMAKE_MATCH_1}")
      set(_protoc_id
          "libprotoc ${_protoc_version} at ${PROTOBUF_PROTOC_EXECUTABLE}")
    endif()
  endif()

  if(_protoc_version VERSION_EQUAL PROTOBUF_EXPECTED_VERSION)
    message(STATUS "Cross-compile: using host ${_protoc_id}")
  elseif(_protoc_pinned)
    message(FATAL_ERROR
      "PROTOBUF_PROTOC_EXECUTABLE is ${_protoc_id}: expected libprotoc "
      "${PROTOBUF_EXPECTED_VERSION} to match the linked protobuf runtime.\n"
      "  Unset it to build a matching host protoc from source.")
  else()
    message(STATUS "Cross-compile: building host protoc "
      "${PROTOBUF_EXPECTED_VERSION} from source (${_protoc_id})")

    include(ExternalProject)
    find_program(HOST_C_COMPILER   NAMES gcc cc  NO_CMAKE_FIND_ROOT_PATH REQUIRED)
    find_program(HOST_CXX_COMPILER NAMES g++ c++ NO_CMAKE_FIND_ROOT_PATH REQUIRED)

    set(PROTOBUF_PROTOC_EXECUTABLE "${CMAKE_BINARY_DIR}/host-protoc/protoc")
    ExternalProject_Add(host_protoc
      SOURCE_DIR       "${protobuf_SOURCE_DIR}"
      BINARY_DIR       "${CMAKE_BINARY_DIR}/host-protoc"
      DOWNLOAD_COMMAND ""
      INSTALL_COMMAND  ""   # only the binary is needed, used from the build dir
      CMAKE_ARGS
        # Deliberately NOT passing CMAKE_TOOLCHAIN_FILE: this is a host build.
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_C_COMPILER=${HOST_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${HOST_CXX_COMPILER}
        -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
        -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
        -Dprotobuf_BUILD_PROTOC_BINARIES=ON
        -Dprotobuf_BUILD_TESTS=OFF
        -Dprotobuf_WITH_ZLIB=OFF
      BUILD_BYPRODUCTS    "${PROTOBUF_PROTOC_EXECUTABLE}"
      USES_TERMINAL_BUILD TRUE)
  endif()

  add_executable(protobuf::protoc IMPORTED GLOBAL)
  set_target_properties(protobuf::protoc PROPERTIES
    IMPORTED_LOCATION "${PROTOBUF_PROTOC_EXECUTABLE}")
endif()

# --- ONNX ---
# Same commit as onnxruntime-ep-tensorrt (v1.18.0).
# v1.18.0 reuses the parent FetchContent protobuf via find_package rather than
# downloading its own, which avoids the cross-compilation protoc conflict seen
# with v1.21.0 (which unconditionally fetches protobuf 31.1 and uses
# $<TARGET_FILE:protobuf::protoc>).
set(ONNX_ML                1   CACHE INTERNAL "")
set(ONNX_BUILD_TESTS       OFF CACHE INTERNAL "")
set(ONNX_BUILD_PYTHON_SRCS OFF CACHE INTERNAL "")
FetchContent_Declare(onnx
  GIT_REPOSITORY https://github.com/onnx/onnx.git
  GIT_TAG        e709452ef2bbc1d113faf678c24e6d3467696e83  # v1.18.0
)
FetchContent_MakeAvailable(onnx)

# ONNX's codegen custom commands reference protobuf::protoc by path, so they
# need an explicit ordering dependency on the host protoc ExternalProject.
if(TARGET host_protoc)
  foreach(_gen IN ITEMS gen_onnx_proto gen_onnx_operators_proto gen_onnx_data_proto)
    add_dependencies(${_gen} host_protoc)
  endforeach()
endif()

FetchContent_GetProperties(onnx SOURCE_DIR ONNX_SRC_DIR)
message(STATUS "ONNX sources      : ${ONNX_SRC_DIR}")

# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------
set(NEURON_IMPL_DIR "${PROJECT_SOURCE_DIR}/src/neuron_plugin_ep")

file(GLOB_RECURSE NEURON_EP_SOURCES "${NEURON_IMPL_DIR}/*.cc")

# Prefer real neuron_implementation.cc over the stub.
set(NEURON_STUB "${NEURON_IMPL_DIR}/neuron_impl/neuron_lib/neuron_implementation_stub.cc")
set(NEURON_REAL "${NEURON_IMPL_DIR}/neuron_impl/neuron_lib/neuron_implementation.cc")
if(EXISTS "${NEURON_REAL}" AND EXISTS "${NEURON_STUB}")
  list(REMOVE_ITEM NEURON_EP_SOURCES "${NEURON_STUB}")
  message(STATUS "neuron_lib: using real neuron_implementation.cc (stub excluded)")
endif()

# op_capability: the per-op capability logic ships as a prebuilt closed-source
# libneuron_op_capability.so that this EP dlopen()s at runtime.
#
# This repo only carries the runtime bridge under neuron_impl/op_capability_abi/;
# its .cc files are picked up by the NEURON_EP_SOURCES glob above.
#
# Nothing here builds or links that .so; it is only required to be
# present (see the find_library check further below).

if(NOT NEURON_EP_SOURCES)
  message(WARNING "No .cc files found under src/neuron_plugin_ep/")
endif()

# ---------------------------------------------------------------------------
# Shared library target
# ---------------------------------------------------------------------------
add_library(onnxruntime_providers_neuron SHARED
  ${NEURON_EP_SOURCES}
)

target_compile_definitions(onnxruntime_providers_neuron PRIVATE
  ONNX_NAMESPACE=onnx
  ONNX_ML=1
  NOMINMAX
  # Oldest ORT API version this EP loads against (from ORTNeuronEp_MIN_ORT_VERSION).
  ORT_NEURON_EP_MIN_ORT_API_VERSION=${ORTNeuronEp_MIN_ORT_API_VERSION}
  # Release version of this EP, reported to ORT via OrtEpFactory::GetVersion.
  # Defined by CMake via project(... VERSION ...) above.
  ORT_NEURON_EP_VERSION="${PROJECT_VERSION}"
  # Set to 0 to enable VERBOSE logs (LOGS_NEURON(VERBOSE) calls).
  # Remove or set to 1 for INFO-and-above only in production.
  NEURON_LOG_MIN_SEVERITY=0
)

# ---------------------------------------------------------------------------
# Include directories
# ---------------------------------------------------------------------------
target_include_directories(onnxruntime_providers_neuron PRIVATE
  # Public ORT headers (onnxruntime_cxx_api.h, onnxruntime_ep_c_api.h, etc.)
  "$<BUILD_INTERFACE:${ORT_INCLUDE_DIR}>"

  # EP shell and neuron_impl root
  "$<BUILD_INTERFACE:${NEURON_IMPL_DIR}>"
  "$<BUILD_INTERFACE:${NEURON_IMPL_DIR}/neuron_impl>"

  # ONNX source tree (proto-generated headers live under the build dir via the
  # onnx FetchContent target's INTERFACE_INCLUDE_DIRECTORIES)
  "${ONNX_SRC_DIR}"

  # GSL headers (from FetchContent)
  "$<BUILD_INTERFACE:$<TARGET_PROPERTY:GSL,INTERFACE_INCLUDE_DIRECTORIES>>"

  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ortneuronep>"
)

# ---------------------------------------------------------------------------
# Link libraries
# ---------------------------------------------------------------------------
target_link_libraries(onnxruntime_providers_neuron PRIVATE
  GSL
  protobuf::libprotobuf
  onnx
  ${CMAKE_DL_LIBS}
)
if(ORT_LIB)
  target_link_libraries(onnxruntime_providers_neuron PRIVATE "${ORT_LIB}")
endif()
if(ANDROID)
  # neuron_lib/ logs through __android_log_* on Android (see the ___NeuronAdapter
  # shim and neuron_implementation.cc), which lives in liblog. On Linux those
  # paths compile out under #ifndef __ANDROID__ and there is nothing to link.
  target_link_libraries(onnxruntime_providers_neuron PRIVATE log)
endif()

# ---------------------------------------------------------------------------
# Symbol visibility
# ---------------------------------------------------------------------------
if(NOT WIN32)
  if(ORT_LIB)
    set(_ort_undef_flag "-Wl,--no-undefined")
  else()
    set(_ort_undef_flag "-Wl,--allow-shlib-undefined")
  endif()
  target_link_options(onnxruntime_providers_neuron PRIVATE
    "-Wl,--version-script=${CMAKE_SOURCE_DIR}/src/neuron_plugin_ep/neuron_plugin_ep.lds"
    "${_ort_undef_flag}"
    "-Wl,--gc-sections"
    "-Wl,-z,noexecstack"
  )
else()
  target_link_options(onnxruntime_providers_neuron PRIVATE
    "-DEF:${CMAKE_SOURCE_DIR}/src/neuron_plugin_ep/neuron_plugin_ep.def")
endif()

set_target_properties(onnxruntime_providers_neuron PROPERTIES
  # Output file is libonnxruntime_providers_neuropilot.so (the CMake target and
  # all C++ identifiers keep the shorter "neuron" name).
  OUTPUT_NAME               onnxruntime_providers_neuropilot
  CXX_VISIBILITY_PRESET     hidden
  VISIBILITY_INLINES_HIDDEN ON
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(onnxruntime_providers_neuron PRIVATE
    -Wall -Wextra -Wno-unused-parameter
  )
endif()

# ---------------------------------------------------------------------------
# Tests (host-only; sources live in test/ to stay out of the EP's GLOB_RECURSE)
# ---------------------------------------------------------------------------
include(CTest)

if(BUILD_TESTING AND NOT CMAKE_CROSSCOMPILING)
  add_executable(graph_payload_elision_test
    "${PROJECT_SOURCE_DIR}/test/graph_payload_elision_test.cc"
    "${NEURON_IMPL_DIR}/neuron_impl/op_capability_abi/graph_payload_elision.cc"
  )

  target_compile_definitions(graph_payload_elision_test PRIVATE
    ONNX_NAMESPACE=onnx
    ONNX_ML=1
  )

  target_include_directories(graph_payload_elision_test PRIVATE
    "${NEURON_IMPL_DIR}/neuron_impl"
    "${ONNX_SRC_DIR}"
  )

  target_link_libraries(graph_payload_elision_test PRIVATE
    protobuf::libprotobuf
    onnx
  )

  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(graph_payload_elision_test PRIVATE -Wall -Wextra)
  endif()

  add_test(NAME graph_payload_elision COMMAND graph_payload_elision_test)
endif()

# ---------------------------------------------------------------------------
# Required library: libneuron_op_capability.so
#
# The per-op IsOpSupported logic ships as a prebuilt closed-source library.
# This EP does NOT build or link it — it dlopen()s it at runtime
# (see neuron_impl/op_capability_abi/op_capability_loader.cc).
# It is a hard requirement: configuration fails if the prebuilt .so cannot be found.
#
# Provide its location with:
#   -DNEURON_OP_CAPABILITY_LIB_DIR=/dir/containing/libneuron_op_capability.so
# ---------------------------------------------------------------------------
find_library(NEURON_OP_CAPABILITY_LIB
  NAMES neuron_op_capability
  HINTS "${NEURON_OP_CAPABILITY_LIB_DIR}"
  NO_DEFAULT_PATH
  NO_CMAKE_FIND_ROOT_PATH
)
if(NOT NEURON_OP_CAPABILITY_LIB)
  message(FATAL_ERROR
    "libneuron_op_capability.so is REQUIRED but was not found.\n"
    "  Download it from the release page, then re-configure with:\n"
    "    -DNEURON_OP_CAPABILITY_LIB_DIR=/dir/containing/libneuron_op_capability.so")
endif()
message(STATUS "  neuron_op_capability : ${NEURON_OP_CAPABILITY_LIB} (required, dlopen'd at runtime)")

# Stage the prebuilt .so beside the EP on install so deployment has both.
if(ORTNeuronEp_INSTALL)
  install(FILES "${NEURON_OP_CAPABILITY_LIB}"
          DESTINATION "${CMAKE_INSTALL_LIBDIR}")
endif()

# ---------------------------------------------------------------------------
# CMake package config
# ---------------------------------------------------------------------------
configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/ORTNeuronEpConfigVersion.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/ORTNeuronEpConfigVersion.cmake"
  @ONLY
)
configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/ORTNeuronEpConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/ORTNeuronEpConfig.cmake"
  @ONLY
)

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
if(ORTNeuronEp_INSTALL)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)

  install(
    DIRECTORY   src/neuron_plugin_ep/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/ortneuronep"
    FILES_MATCHING PATTERN "*.h"
  )
  install(
    FILES
      "${CMAKE_CURRENT_BINARY_DIR}/ORTNeuronEpConfig.cmake"
      "${CMAKE_CURRENT_BINARY_DIR}/ORTNeuronEpConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ortneuronep"
  )
  install(TARGETS onnxruntime_providers_neuron
          EXPORT ORTNeuronEpTargets
          LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
  install(TARGETS GSL
          EXPORT ORTNeuronEpTargets)
  install(EXPORT ORTNeuronEpTargets
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ortneuronep"
    NAMESPACE   ORTNeuronEp::
  )
endif()

# ---------------------------------------------------------------------------
# Python helper package (WIP — ORTNeuronEp_BUILD_PYTHON=OFF by default)
# When enabled, installs python/onnxruntime_ep_neuron/ to the Python
# site-packages directory so users can do:
#   import onnxruntime_ep_neuron as neuron_ep
#   ort.register_execution_provider_library("NeuronEP", neuron_ep.get_library_path())
# ---------------------------------------------------------------------------
if(ORTNeuronEp_BUILD_PYTHON)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
            "import sysconfig; print(sysconfig.get_path('platlib'))"
    OUTPUT_VARIABLE _PY_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  install(
    DIRECTORY   "${PROJECT_SOURCE_DIR}/python/onnxruntime_ep_neuron"
    DESTINATION "${_PY_SITE_PACKAGES}"
  )
  message(STATUS "  Python site-pkgs  : ${_PY_SITE_PACKAGES}")
endif()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "=== onnxruntime_providers_neuron (Plugin EP – standalone) ===")
if(ANDROID)
  message(STATUS "  Target platform   : Android ${ANDROID_ABI} / ${ANDROID_PLATFORM}")
else()
  message(STATUS "  Target platform   : ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
endif()
message(STATUS "  ORT include dir   : ${ORT_INCLUDE_DIR}")
message(STATUS "  ORT library       : ${ORT_LIB}")
message(STATUS "  neuron_impl dir   : ${NEURON_IMPL_DIR}")
message(STATUS "  protoc            : ${PROTOC_BIN}")
message(STATUS "  ONNX sources      : ${ONNX_SRC_DIR}")
message(STATUS "  Build type        : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Python package    : ${ORTNeuronEp_BUILD_PYTHON} (WIP)")
message(STATUS "")
