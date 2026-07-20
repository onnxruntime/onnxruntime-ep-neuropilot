// neuron_ep_utils.h
// Shared utility macros and helpers for the Neuron plugin EP shell.

#pragma once

#include <sstream>
#include <string>

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT
// onnxruntime_ep_c_api.h is intentionally NOT included here.
// onnxruntime_c_api.h (pulled in by onnxruntime_cxx_api.h above) already
// includes it at its end, and that header has no include guard — double
// inclusion causes redefinition errors.

// ---------------------------------------------------------------------------
// Error-handling macros
// ---------------------------------------------------------------------------

// Propagate a non-OK OrtStatus* returned by `fn` upward.
#define RETURN_IF_ERROR(fn)     \
  do {                          \
    Ort::Status _s{(fn)};       \
    if (!_s.IsOK()) {           \
      return _s.release();      \
    }                           \
  } while (0)

// Return an ORT_EP_FAIL status if `cond` is true.
#define RETURN_IF(cond, ort_api, msg)                     \
  do {                                                    \
    if ((cond)) {                                         \
      return (ort_api).CreateStatus(ORT_EP_FAIL, (msg));  \
    }                                                     \
  } while (0)

// Take ownership of an OrtStatus* and silently discard it.
#define IGNORE_ORTSTATUS(expr)       \
  do {                               \
    OrtStatus* _s = (expr);          \
    Ort::Status _ignored{_s};        \
  } while (false)

// ---------------------------------------------------------------------------
// Logging macro (requires `ort_api` and `logger_` in scope)
// ---------------------------------------------------------------------------

#define NEURON_LOG(level, ...)                                                        \
  do {                                                                                \
    std::ostringstream _ss;                                                           \
    _ss << __VA_ARGS__;                                                               \
    IGNORE_ORTSTATUS(ort_api.Logger_LogMessage(&logger_,                              \
                                               ORT_LOGGING_LEVEL_##level,             \
                                               _ss.str().c_str(),                     \
                                               __FILE__, __LINE__, __FUNCTION__));    \
  } while (false)

// ---------------------------------------------------------------------------
// Core structs
// ---------------------------------------------------------------------------

// Bundles the three ORT API pointers every EP file needs.
struct ApiPtrs {
  const OrtApi&            ort_api;
  const OrtEpApi&          ep_api;
  const OrtModelEditorApi& model_editor_api;
};

// ---------------------------------------------------------------------------
// Session-config helper
// ---------------------------------------------------------------------------

inline OrtStatus* GetSessionConfigEntryOrDefault(const OrtSessionOptions& session_options,
                                                  const char* config_key,
                                                  const std::string& default_val,
                                                  std::string& config_val) {
  try {
    Ort::ConstSessionOptions opts{&session_options};
    config_val = opts.GetConfigEntryOrDefault(config_key, default_val);
  } catch (const Ort::Exception& ex) {
    Ort::Status s(ex);
    return s.release();
  }
  return nullptr;
}
