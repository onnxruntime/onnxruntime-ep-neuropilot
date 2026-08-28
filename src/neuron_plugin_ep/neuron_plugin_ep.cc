// neuron_plugin_ep.cc
// DLL / shared-library entry point.
//
// ORT calls CreateEpFactories() immediately after dlopen().
// ReleaseEpFactory() is called when the ORT environment is destroyed.
//
// Only these two C symbols are exported (see .lds / .def files).

#define ORT_API_MANUAL_INIT   // suppress automatic global init inside a DLL
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "neuron_ep_factory.h"
#include "neuron_impl/op_capability_abi/op_capability_loader.h"

// -fvisibility=hidden (from CXX_VISIBILITY_PRESET) hides all symbols at
// compile time; the .lds version script cannot override that.  Explicitly
// mark the two entry points as default visibility so they survive into the
// dynamic symbol table.  Everything else stays hidden.
#define EXPORT_SYMBOL __attribute__((visibility("default")))

// oldest ORT we can run against; the plugin EP surface we use landed in 1.23
static constexpr uint32_t kMinSupportedOrtApiVersion = 23;

extern "C" {

// ---------------------------------------------------------------------------
// CreateEpFactories
// Called by ORT once after loading this shared library.
//   registration_name – unused; the EP name is NeuronEpFactory::kEpName.
//   ort_api_base      – use this to obtain the versioned ORT C API.
//   default_logger    – process-level logger.
//   factories / max_factories / num_factories – output array.
// ---------------------------------------------------------------------------
EXPORT_SYMBOL OrtStatus* CreateEpFactories(
    const char*          /*registration_name*/,
    const OrtApiBase*    ort_api_base,
    const OrtLogger*     default_logger,
    OrtEpFactory**       factories,
    size_t               max_factories,
    size_t*              num_factories) {

  // 1. Negotiate the ORT API version.  GetApi() returns null for a version the
  //    runtime does not implement, so walk down to the newest one it does.
  const OrtApi* ort_api        = nullptr;
  uint32_t      ort_api_version = 0;
  for (uint32_t v = ORT_API_VERSION; v >= kMinSupportedOrtApiVersion; --v) {
    if ((ort_api = ort_api_base->GetApi(v)) != nullptr) {
      ort_api_version = v;
      break;
    }
  }

  // no OrtApi means no way to build an OrtStatus; ORT reports the load failure
  if (ort_api == nullptr) return nullptr;

  const OrtEpApi*          ep_api           = ort_api->GetEpApi();
  const OrtModelEditorApi* model_editor_api = ort_api->GetModelEditorApi();

  // 2. Manually initialise the C++ wrapper (required in DLLs).
  Ort::InitApi(ort_api);

  // 3. Sanity-check output buffer size.
  if (max_factories < 1) {
    return ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                  "Not enough space to return EP factory. Need at least 1.");
  }

  // 4. Eagerly load libneuron_op_capability.so. This is a hard dependency —
  //    without it the EP cannot determine which operators the hardware supports.
  //    Fail here rather than silently accepting all ops at capability-check time.
  {
    const auto& loader = onnxruntime::neuron::op_capability::OpCapabilityLoader::Instance();
    if (!loader.IsLoaded()) {
      return ort_api->CreateStatus(ORT_EP_FAIL,
                                    "NeuronEP: libneuron_op_capability.so could not be loaded. "
                                    "Ensure it is installed alongside libonnxruntime_providers_neuropilot.so.");
    }
  }

  // 5. Create the factory.  The EP name is NeuronEpFactory::kEpName, not the
  //    registration name.
  auto factory = std::make_unique<NeuronEpFactory>(
      ApiPtrs{*ort_api, *ep_api, *model_editor_api, ort_api_version},
      *default_logger,
      ort_api_version);

  factories[0]  = factory.release();
  *num_factories = 1;
  return nullptr;
}

// ---------------------------------------------------------------------------
// ReleaseEpFactory
// Called by ORT when the environment is shut down.
// ---------------------------------------------------------------------------
EXPORT_SYMBOL OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  delete static_cast<NeuronEpFactory*>(factory);
  return nullptr;
}

}  // extern "C"
