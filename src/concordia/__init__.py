"""Concordia package and workflow
"""

from importlib.metadata import version as _version

from .report import add_sticky_toc, embed_image
from .utils import RegionMapping, VariableDefinitions, combine_countries


try:
    __version__ = _version("concordia")
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"
