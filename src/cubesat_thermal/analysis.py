"""Engineering checks built on the solver: margins, heater sizing, beta sweeps, hand calculations."""
from __future__ import annotations

import dataclasses

import numpy as np

from .model import SIGMA, T_SPACE, K0, Result, ThermalModel


# ---------------------------------------------------------------------- margins
def check_margins(model: ThermalModel, result: Result) -> list[dict]:
    """PASS: prediction +/- uncertainty inside operating limits.
    WATCH: prediction inside limits but closer than the uncertainty.
    FAIL: prediction outside operating limits."""
    u = model.uncertainty_c
    rows = []
    for node, lim in model.limits.items():
        lo, hi = lim["op"]
        tmin, tmax = result.tmin(node), result.tmax(node)
        m_lo, m_hi = tmin - lo, hi - tmax
        if tmin < lo or tmax > hi:
            status = "FAIL"
        elif m_lo < u or m_hi < u:
            status = "WATCH"
        else:
            status = "PASS"
        rows.append({"node": node, "tmin_c": round(tmin, 1), "tmax_c": round(tmax, 1),
                     "op_limits_c": [lo, hi], "margin_cold_c": round(m_lo, 1),
                     "margin_hot_c": round(m_hi, 1), "status": status})
    return rows


# ---------------------------------------------------------------------- heater sizing
def size_heater(model: ThermalModel, case: str = "cold", node: str = "Battery",
                target_c: float | None = None, tol_w: float = 0.01,
                duty_margin: float = 0.02, max_steps: int = 100) -> dict:
    """Constant heater power that keeps `node` at or above target (default: op min + uncertainty),
    then installed power = required / max duty (70 %), verified with the thermostat. The
    thermostat holds the node inside its deadband, above the sizing target, so it loses more
    heat than the constant-power estimate: the installed power is raised in 0.1 W steps until
    the thermostat duty is at most max_duty - duty_margin."""
    lo = model.limits[node]["op"][0]
    target = lo + model.uncertainty_c if target_c is None else target_c
    base = model.cases[case].dissipation.get(node, 0.0)

    def tmin_with(p):
        m = model.copy_with(dissipation=(case, node, base + p), heater_power=0.0)
        return m.run(case).tmin(node)

    if tmin_with(0.0) >= target:
        return {"node": node, "case": case, "required_w": 0.0, "installed_w": 0.0,
                "note": "no heater needed", "target_c": target}
    a, b = 0.0, 1.0
    while tmin_with(b) < target:
        a, b = b, b * 2
        if b > 200:
            raise RuntimeError("heater power above 200 W, check the model")
    while b - a > tol_w:
        mid = 0.5 * (a + b)
        a, b = (mid, b) if tmin_with(mid) < target else (a, mid)
    required = b
    max_duty = model.heaters[0].max_duty if model.heaters else 0.7
    installed = float(np.ceil(required / max_duty * 10) / 10)   # round up to 0.1 W
    duty_limit = max_duty - duty_margin
    for _ in range(max_steps):
        check = model.copy_with(heater_power=installed).run(case)
        duty = float(next(iter(check.energy["heater_duty"].values()), 0.0))
        if duty <= duty_limit:
            break
        installed = round(installed + 0.1, 1)
    else:
        raise RuntimeError(f"thermostat duty still above {duty_limit:.2f} at {installed} W")
    return {"node": node, "case": case, "target_c": target, "required_w": round(required, 2),
            "installed_w": installed, "duty_with_thermostat": round(duty, 3),
            "duty_limit": round(duty_limit, 3), "status": check.status,
            "converged": check.converged,
            "periodicity_pct": round(check.energy["periodicity_pct"], 4),
            "window_periodicity_pct": (None if check.window_periodicity_pct is None
                                       else round(check.window_periodicity_pct, 4)),
            "tmin_with_thermostat_c": round(check.tmin(node), 2),
            # duty is the window mean in quasi-periodic mode, so use it rather than one orbit
            "heater_energy_wh_per_orbit": round(duty * installed * check.period_s / 3600, 3)}


# ---------------------------------------------------------------------- beta sweep
def beta_sweep(model: ThermalModel, case: str, betas) -> list[dict]:
    rows = []
    for b in betas:
        c = dataclasses.replace(model.cases[case], beta_deg=float(b))
        r = model.run(c)
        rows.append({"beta_deg": float(b),
                     **{f"{n}_min": round(r.tmin(n), 2) for n in model.limits},
                     **{f"{n}_max": round(r.tmax(n), 2) for n in model.limits}})
    return rows


# ---------------------------------------------------------------------- hand calculations (GSFC)
def radiator_area(q_w: float, t_k: float, eps: float, f_earth: float, earth_flux: float) -> float:
    """Radiator area for a box dissipating q_w at temperature t_k with an Earth view factor."""
    return q_w / (SIGMA * eps * (t_k**4 - T_SPACE**4) - f_earth * eps * earth_flux)


def radiator_heater(q_w: float, area: float, t_k: float, eps: float, f_earth: float,
                    earth_flux: float) -> float:
    """Heater power that keeps a radiator of given area at t_k."""
    return SIGMA * eps * area * (t_k**4 - T_SPACE**4) - q_w - f_earth * eps * earth_flux * area
