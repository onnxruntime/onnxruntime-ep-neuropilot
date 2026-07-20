"""Python helper utilities for the Neuron ONNX Runtime plugin EP package.

STATUS: Work-in-progress. Not yet wired into the Yocto recipe or pip packaging.

This follows the ONNX Runtime plugin EP packaging pattern (same as telum/tensorrt EPs):
  - get_ep_name()       — canonical EP name string for ORT APIs
  - get_ep_names()      — list of EP names exposed by this plugin
  - get_library_path()  — absolute path to libonnxruntime_providers_neuron.so

Usage from Python:
    import onnxruntime as ort
    import onnxruntime_ep_neuron as neuron_ep

    ort.register_execution_provider_library(
        "NeuronEP", neuron_ep.get_library_path()
    )
    devices = [d for d in ort.get_ep_devices()
               if d.ep_name == neuron_ep.get_ep_name()]

    opts = ort.SessionOptions()
    opts.add_provider_for_devices(devices, {})
    sess = ort.InferenceSession("model.onnx", sess_options=opts)

    ort.unregister_execution_provider_library("NeuronEP")
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# EP identity — must match NeuronEpFactory::GetNameImpl() in neuron_ep_factory.cc
# ---------------------------------------------------------------------------
_EP_NAME = "NeuronPluginExecutionProvider"

# Library base name (without lib prefix or platform suffix).
_LIB_BASE = "onnxruntime_providers_neuropilot"


def get_ep_name() -> str:
    """Return the canonical EP name used with ORT APIs."""
    return _EP_NAME


def get_ep_names() -> List[str]:
    """Return the list of EP names exposed by this plugin package."""
    return [_EP_NAME]


def _platform_lib_filename(base: str) -> str:
    if sys.platform.startswith("win"):
        return f"{base}.dll"
    if sys.platform == "darwin":
        return f"lib{base}.dylib"
    return f"lib{base}.so"


def get_library_path() -> str:
    """Return the absolute path to the Neuron plugin EP shared library.

    Search order:
      1. Next to this __init__.py (pip-installed / bundled layout).
      2. Standard system library paths (Yocto / distro install at /usr/lib).

    Raises FileNotFoundError if the library cannot be found.
    """
    lib_name = _platform_lib_filename(_LIB_BASE)
    pkg_dir = Path(__file__).resolve().parent

    # 1. Bundled alongside this package (pip wheel layout)
    bundled = pkg_dir / lib_name
    if bundled.is_file():
        return str(bundled)

    # 2. System library directories (Yocto installs to /usr/lib)
    system_candidates = [
        Path("/usr/lib") / lib_name,
        Path("/usr/local/lib") / lib_name,
        Path("/lib") / lib_name,
    ]
    for p in system_candidates:
        if p.is_file():
            return str(p)

    raise FileNotFoundError(
        f"Neuron plugin EP shared library '{lib_name}' not found.\n"
        f"Searched:\n"
        f"  {bundled} (pip-bundled)\n"
        + "\n".join(f"  {p}" for p in system_candidates)
    )
