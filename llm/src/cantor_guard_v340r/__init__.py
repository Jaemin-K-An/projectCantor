"""V3.4.0R -- protocol rectification over the frozen V3.4.0 sensor and actuator.

Nothing here refits w, v, W, the depth or the rho family. What changes is the
CONTROL LAW's budget discipline (a real statewise cap) and the experimental
design (attacked-state calibration, no-controller and non-Cantor baselines).
"""
from .controllers import CappedCantorController, LinearThresholdController  # noqa: F401
