// neuron_ep_version.h
// Gating of optional ORT API features by the negotiated ORT API version.
//
// The EP is compiled against one set of ORT headers (ORT_API_VERSION) but may be
// loaded by an older runtime, down to ORT_NEURON_EP_MIN_ORT_API_VERSION.  Two
// guards are needed before a newer feature is used:
//   - compile time: the headers we build with must declare it (ORT_API_VERSION);
//   - run time: the runtime must provide it, i.e. the version negotiated in
//     CreateEpFactories (ApiPtrs::ort_api_version) must be at least N.
// A callback set on an older runtime would be ignored by ORT (it reads only the
// OrtEp members up to ort_version_supported), but any OrtApi/OrtEpApi function
// the callback calls would be read past the end of the runtime's smaller table.

#pragma once

#include <cstdint>
#include <utility>

// Runs func(ep) only if the headers declare the feature (compile time,
// ORT_API_VERSION >= MinVersion) and the negotiated runtime version is >=
// MinVersion.  The EP pointer is passed to the lambda as auto* so accesses to
// members that older headers lack stay template-dependent; with if constexpr the
// discarded branch is never instantiated, so older headers still compile.
template <uint32_t MinVersion, typename EpT, typename Func>
void OrtVersionGate(uint32_t negotiated, EpT* ep, Func&& func) {
  if constexpr (ORT_API_VERSION >= MinVersion) {
    if (negotiated >= MinVersion) std::forward<Func>(func)(ep);
  }
}
