"""Public nonrecursive sensor-actuator baseline module for V3.4.0R."""
from .controllers import LinearThresholdController

LinearSensorActuator = LinearThresholdController

__all__ = ["LinearSensorActuator", "LinearThresholdController"]
