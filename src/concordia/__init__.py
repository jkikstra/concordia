"""Concordia package and workflow
"""

from importlib.metadata import version as _version

from .utils import RegionMapping, VariableDefinitions


try:
    __version__ = _version("concordia")
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"
