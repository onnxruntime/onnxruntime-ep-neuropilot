// neuron_types_internal.h
//
// Plugin EP internal type definitions and macros that replace the ORT
// internal headers previously needed by neuron_impl:
//
//   core/common/common.h          → NeuronStatus, ORT_RETURN_IF_ERROR, etc.
//   core/common/inlined_containers.h → replaced by std::vector / std::unordered_map
//   core/common/logging/logging.h → NEURON_LOG macro
//   core/common/make_string.h     → MakeNeuronString
//
// This header is the ONLY "internal" header that neuron_impl may use in
// addition to public ORT headers and ONNX protobuf headers.

#pragma once

#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ---------------------------------------------------------------------------
// MakeNeuronString – variadic string builder (replaces ORT MakeString)
// ---------------------------------------------------------------------------
namespace neuron_detail {
inline void BuildString(std::ostringstream& /*ss*/) {}
template <typename T, typename... Rest>
inline void BuildString(std::ostringstream& ss, const T& val, const Rest&... rest) {
  ss << val;
  BuildString(ss, rest...);
}
}  // namespace neuron_detail

template <typename... Args>
inline std::string MakeNeuronString(const Args&... args) {
  std::ostringstream ss;
  neuron_detail::BuildString(ss, args...);
  return ss.str();
}

// ---------------------------------------------------------------------------
// NeuronStatusCode – error-code enum used by NeuronStatus constructors
// ---------------------------------------------------------------------------
enum class NeuronStatusCode {
  OK = 0,
  FAIL,
  INVALID_ARGUMENT,
  NOT_IMPLEMENTED,
  OUT_OF_MEMORY,
};

// ---------------------------------------------------------------------------
// NeuronStatus – lightweight Status replacement (replaces onnxruntime::Status)
// ---------------------------------------------------------------------------
struct NeuronStatus {
  bool ok_{true};
  std::string msg_;

  NeuronStatus() = default;

  explicit NeuronStatus(bool ok, std::string msg = {})
      : ok_(ok), msg_(std::move(msg)) {}

  // Construct from error code (any code != OK is a failure).
  NeuronStatus(NeuronStatusCode code, std::string msg = {})
      : ok_(code == NeuronStatusCode::OK), msg_(std::move(msg)) {}

  static NeuronStatus OK() { return NeuronStatus{true, {}}; }

  bool IsOK() const { return ok_; }
  const std::string& ErrorMessage() const { return msg_; }

  // Allow implicit check: if (status) means IsOK()
  explicit operator bool() const { return ok_; }
};

// Convenience alias so neuron_impl code can write `using Status = NeuronStatus`
// inside a namespace if desired.
using NeuronStatus = NeuronStatus;

// ---------------------------------------------------------------------------
// Macros (replicate ORT internal macros without pulling in core/common/common.h)
// ---------------------------------------------------------------------------

// Propagate a NeuronStatus error upward.
#define NEURON_RETURN_IF_ERROR(expr)          \
  do {                                         \
    auto _ns = (expr);                         \
    if (!_ns.IsOK()) return _ns;               \
  } while (0)

// Return an error NeuronStatus if condition is false.
#define NEURON_RETURN_IF_NOT(cond, ...)        \
  do {                                         \
    if (!(cond)) {                             \
      return NeuronStatus{false,               \
          MakeNeuronString(__VA_ARGS__)};       \
    }                                          \
  } while (0)

// Throw if condition is false (replaces ORT_ENFORCE for fatal checks).
#define NEURON_ENFORCE(cond, ...)              \
  do {                                         \
    if (!(cond)) {                             \
      throw std::runtime_error(               \
          MakeNeuronString(__VA_ARGS__));       \
    }                                          \
  } while (0)

// Build a NeuronStatus error (replaces ORT_MAKE_STATUS).
#define NEURON_MAKE_STATUS(...) NeuronStatus{false, MakeNeuronString(__VA_ARGS__)}

// Adapt macros that existing code uses with ORT naming so they map here.
// This allows neuron_impl files to compile without changing every call site.
#ifndef ORT_RETURN_IF_ERROR
#define ORT_RETURN_IF_ERROR(expr)  NEURON_RETURN_IF_ERROR(expr)
#endif
#ifndef ORT_RETURN_IF_NOT
#define ORT_RETURN_IF_NOT(cond, ...) NEURON_RETURN_IF_NOT(cond, __VA_ARGS__)
#endif
#ifndef ORT_ENFORCE
#define ORT_ENFORCE(cond, ...) NEURON_ENFORCE(cond, __VA_ARGS__)
#endif
#ifndef ORT_MAKE_STATUS
#define ORT_MAKE_STATUS(cat, code, ...) NEURON_MAKE_STATUS(__VA_ARGS__)
#endif
#ifndef ORT_THROW_IF_ERROR
#define ORT_THROW_IF_ERROR(expr) \
  do { auto _s = (expr); if (!_s.IsOK()) throw std::runtime_error(_s.ErrorMessage()); } while (0)
#endif
#ifndef ORT_NOT_IMPLEMENTED
#define ORT_NOT_IMPLEMENTED(...) \
  throw std::runtime_error(MakeNeuronString("Not implemented: ", __VA_ARGS__))
#endif

// Alias so existing code that uses `Status` compiles unchanged.
// Usage: add `using Status = NeuronStatus;` in the relevant namespace,
// or change type names in neuron_impl sources.

// ---------------------------------------------------------------------------
// Shape – vector<uint32_t> (replaces InlinedVector<uint32_t> from ORT core)
// ---------------------------------------------------------------------------
// Shape is defined in neuron_graph_context.h to avoid circular includes.
// Declare the using alias here so helper.h and friends can find it.

// ---------------------------------------------------------------------------
// Logging – simple stderr shim (replaces core/common/logging/logging.h)
// ---------------------------------------------------------------------------
// When ORT logging infra is not available inside the neuron_impl layer, route
// all logs to stderr.  In a production build, swap this to a real logger.

#ifndef NEURON_LOG_SEVERITY_VERBOSE
#define NEURON_LOG_SEVERITY_VERBOSE 0
#define NEURON_LOG_SEVERITY_INFO    1
#define NEURON_LOG_SEVERITY_WARNING 2
#define NEURON_LOG_SEVERITY_ERROR   3
#endif

// Compile-time log level threshold – set to 1 (INFO) by default.
#ifndef NEURON_LOG_MIN_SEVERITY
#define NEURON_LOG_MIN_SEVERITY NEURON_LOG_SEVERITY_INFO
#endif

// NeuronLogStream: stream-style logging helper.
struct NeuronLogStream {
  int severity_;
  bool active_;
  std::ostringstream ss_;

  NeuronLogStream(int severity, const char* tag)
      : severity_(severity),
        active_(severity >= NEURON_LOG_MIN_SEVERITY) {
    if (active_) ss_ << "[" << tag << "] ";
  }

  template <typename T>
  NeuronLogStream& operator<<(const T& val) {
    if (active_) ss_ << val;
    return *this;
  }

  ~NeuronLogStream() {
    if (active_) {
      const char* sev = (severity_ >= NEURON_LOG_SEVERITY_ERROR)   ? "ERROR"
                      : (severity_ >= NEURON_LOG_SEVERITY_WARNING) ? "WARN "
                      : (severity_ >= NEURON_LOG_SEVERITY_INFO)    ? "INFO "
                                                                    : "VERB ";
      std::fprintf(stderr, "[NeuronEP][%s] %s\n", sev, ss_.str().c_str());
    }
  }
};

// Primary log macro – mirrors ORT's LOGS_CATEGORY usage pattern.
// Severity names mirror ORT: VERBOSE, INFO, WARNING, ERROR.
#define LOGS_NEURON(severity) \
  NeuronLogStream(NEURON_LOG_SEVERITY_##severity, "neuron")

// Alias used in existing helper.h (LOGS_NEURON already defined above).

// ---------------------------------------------------------------------------
// Misc helpers
// ---------------------------------------------------------------------------

// Replaces ORT Contains() helper used in several places.
template <typename C, typename V>
inline bool Contains(const C& container, const V& value) {
  return container.find(value) != container.end();
}

// narrow<T> – checked narrowing cast (replaces gsl::narrow / ORT narrow).
template <typename To, typename From>
inline To narrow(From value) {
  auto result = static_cast<To>(value);
  if (static_cast<From>(result) != value) {
    throw std::runtime_error("narrow: value out of range");
  }
  return result;
}
