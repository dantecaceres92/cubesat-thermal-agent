"""CubeSat orbital thermal analysis engine (preliminary design, nodal model)."""
from .analysis import beta_sweep, check_margins, radiator_area, radiator_heater, size_heater
from .model import Case, Result, ThermalModel
from .orbit import (critical_beta_deg, earth_view_factors, eclipse_fraction, period_s,
                    sso_beta_range)

__all__ = ["ThermalModel", "Case", "Result", "check_margins", "size_heater", "beta_sweep",
           "radiator_area", "radiator_heater", "period_s", "eclipse_fraction",
           "critical_beta_deg", "earth_view_factors", "sso_beta_range"]
__version__ = "0.1.0"
