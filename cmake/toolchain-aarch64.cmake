set(CMAKE_SYSTEM_NAME      Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# Programs (cmake, git, protoc) are always the HOST binaries.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# Libraries and headers come from the cross sysroot / explicit paths.
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Prevent protobuf FetchContent from building an aarch64 protoc that can't
# run on the x86 host.  A version-matched HOST protoc is built separately by
# the top-level CMakeLists.txt (see its CMAKE_CROSSCOMPILING block).
set(protobuf_BUILD_PROTOC_BINARIES OFF CACHE BOOL "" FORCE)

# Host zlib (/usr/include + /usr/lib/libz.so) must never leak into the
# aarch64 build: it injects `-isystem /usr/include`, which makes the cross
# compiler use host glibc headers (__WORDSIZE=32 -> 32-bit uintptr_t).
set(protobuf_WITH_ZLIB OFF CACHE BOOL "" FORCE)
