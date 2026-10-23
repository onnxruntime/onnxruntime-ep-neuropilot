/* neuron_plugin_ep.lds
   Linux linker version script.
   Exports ONLY CreateEpFactories, ReleaseEpFactory and the dmabuf
   registration pair (neuron_dmabuf.h).
   Everything else (C++ symbols, STL internals) stays hidden,
   which prevents symbol collisions when multiple plugin EPs are loaded. */

VERS_1.0.0 {
  global:
    CreateEpFactories;
    ReleaseEpFactory;
    NeuronEp_RegisterDmabuf;
    NeuronEp_UnregisterDmabuf;
  local:
    *;
};
