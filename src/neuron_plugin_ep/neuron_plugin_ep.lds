/* neuron_plugin_ep.lds
   Linux linker version script.
   Exports ONLY CreateEpFactories and ReleaseEpFactory.
   Everything else (C++ symbols, STL internals) stays hidden,
   which prevents symbol collisions when multiple plugin EPs are loaded. */

VERS_1.0.0 {
  global:
    CreateEpFactories;
    ReleaseEpFactory;
  local:
    *;
};
