"""Compatibility shim for cocotb-test 0.2.6 with cocotb 2.x.

The locked OpenEye test runner still calls the removed ``lib_name`` helper.
Keep the workaround outside the pinned OpenEye checkout and return the absolute
shared-library path accepted by the cocotb 2.x VPI loader.
"""

import os
import sys
import types

from cocotb_tools import config
import find_libpython


if not hasattr(config, "lib_name"):
    # cocotb-test 0.2.6 uses the removed helper and also adds ``-M``. With
    # cocotb 2.x, passing the absolute VPI library directly matches the
    # maintained cocotb Makefile and avoids vvp's legacy ``.vpi`` lookup.
    from cocotb_test import simulator as _simulator

    config.lib_name = lambda interface, simulator: str(
        config.lib_name_path(interface, simulator)
    )

    def _icarus_run_command(self):
        return (
            [
                "vvp",
                "-m",
                str(config.lib_name_path("vpi", "icarus")),
            ]
            + self.simulation_args
            + [self.sim_file]
            + self.plus_args
        )

    _simulator.Icarus.run_command = _icarus_run_command

# Make the GPI bootstrap used by cocotb 2.x explicit. cocotb-test's direct
# vvp invocation does not inherit the Makefile's COCOTB_ENV setup.
os.environ.setdefault(
    "GPI_USERS",
    ";".join((find_libpython.find_libpython(), config.pygpi_entry_point())),
)

# The constant convolution fixture creates layers procedurally and never uses
# ONNX import. OpenEye imports this optional helper unconditionally, while the
# pinned environment intentionally does not include torch/onnxruntime.
if "open_eye.onnx2model" not in sys.modules:
    _onnx_stub = types.ModuleType("open_eye.onnx2model")
    _onnx_stub.get_model = lambda model: model
    sys.modules["open_eye.onnx2model"] = _onnx_stub
