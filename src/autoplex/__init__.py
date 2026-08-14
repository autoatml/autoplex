"""
AutoPLEX -- Automated machine-learned Potential Landscape explorer -- flows.

This package is designed to perform a workflow-based
automated machine-learned interatomic potentials (MLIP) fit for
DFT-labelled data.

"""

from autoplex._version import __version__
from autoplex.settings import (
    MLIPHypers,
    RssConfig,
)

MLIP_HYPERS = MLIPHypers()
RSS_CONFIG = RssConfig()
