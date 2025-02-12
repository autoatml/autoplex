"""
AutoPLEX -- Automated machine-learned Potential Landscape explorer -- flows.

This package is designed to perform a workflow-based
automated machine-learned interatomic potentials (MLIP) fit for
DFT-labelled data.

"""

from autoplex._version import __version__
from autoplex.settings import (
    GAPSettings,
    JACESettings,
    M3GNETSettings,
    MACESettings,
    MLIPHypers,
    NEPSettings,
    NEQUIPSettings,
    RssConfig,
)

MLIP_HYPERS = MLIPHypers()
RSS_CONFIG = RssConfig()
GAP_HYPERS = GAPSettings()
JACE_HYPERS = JACESettings()
M3GNET_HYPERS = M3GNETSettings()
MACE_HYPERS = MACESettings()
NEQUIP_HYPERS = NEQUIPSettings()
NEP_HYPERS = NEPSettings()
