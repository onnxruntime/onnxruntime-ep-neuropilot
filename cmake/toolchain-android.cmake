# Android (NDK) cross-compile toolchain.
#
# Counterpart of cmake/toolchain-aarch64.cmake: that one targets aarch64 Linux
# (glibc, aarch64-linux-gnu-g++), this one targets Android (bionic, NDK clang++).
# It is a thin wrapper around the NDK's own android.toolchain.cmake plus the same
# host-protoc handling, so both targets configure identically:
#
#   cmake -S . -B build-android -G Ninja \
#     -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-android.cmake \
#     -DANDROID_NDK_HOME=/path/to/android-ndk \
#     ...
#
# Variables (command line or environment; command line wins):
#   ANDROID_NDK_HOME  NDK root. Also read from the environment as
#                     ANDROID_NDK_HOME / ANDROID_NDK_ROOT / ANDROID_NDK.
#   ANDROID_ABI       target ABI      (default arm64-v8a)
#   ANDROID_PLATFORM  min API level   (default android-29)
#
# The Neuron NPU is arm64-only, so arm64-v8a is the only ABI that makes sense
# for a real device; the variable exists so the sources can still be
# compile-tested for another ABI.

# --- locate the NDK ---------------------------------------------------------
if(NOT ANDROID_NDK_HOME)
  foreach(_ndk_env ANDROID_NDK_HOME ANDROID_NDK_ROOT ANDROID_NDK)
    if(DEFINED ENV{${_ndk_env}}
       AND EXISTS "$ENV{${_ndk_env}}/build/cmake/android.toolchain.cmake")
      set(ANDROID_NDK_HOME "$ENV{${_ndk_env}}")
      break()
    endif()
  endforeach()
endif()

if(NOT ANDROID_NDK_HOME OR NOT EXISTS "${ANDROID_NDK_HOME}/build/cmake/android.toolchain.cmake")
  message(FATAL_ERROR
    "Android NDK not found.\n"
    "  Pass -DANDROID_NDK_HOME=/path/to/android-ndk (or set ANDROID_NDK_HOME in\n"
    "  the environment); the directory must contain build/cmake/android.toolchain.cmake.")
endif()

# --- defaults the NDK toolchain reads --------------------------------------
if(NOT ANDROID_ABI)
  set(ANDROID_ABI "arm64-v8a")
endif()
if(NOT ANDROID_PLATFORM)
  set(ANDROID_PLATFORM "android-29")
endif()

# CMake re-reads this toolchain file inside every try_compile() sub-project, and
# those do NOT inherit the outer -D command line. Without this, ANDROID_NDK_HOME
# is empty there and the compiler ABI check fails with "Android NDK not found".
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES
  ANDROID_NDK_HOME ANDROID_ABI ANDROID_PLATFORM)

# android.toolchain.cmake refers to the NDK root as ANDROID_NDK.
set(ANDROID_NDK "${ANDROID_NDK_HOME}")
include("${ANDROID_NDK_HOME}/build/cmake/android.toolchain.cmake")

# --- protoc, exactly as in toolchain-aarch64.cmake --------------------------
# Never build a target-architecture protoc: it cannot run on the x86 build host.
# A version-matched HOST protoc is provided separately by the top-level
# CMakeLists.txt (see its CMAKE_CROSSCOMPILING block).
set(protobuf_BUILD_PROTOC_BINARIES OFF CACHE BOOL "" FORCE)
