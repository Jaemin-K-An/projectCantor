"""V3.4.0 -- sensor/actuator decoupled Cantor-certified P0 safety control.

The V3.3.5c lineage forced one residual direction to serve as both the
behavioural state SENSOR and the causal ACTUATOR.  V3.4.0 separates them:
``w`` senses, ``v`` actuates, and ``kappa = <w_hat, v>`` is their coupling.
"""
from .sensor_distance import SensorHyperplane  # noqa: F401
