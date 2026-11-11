"""MCP server for the CubeSat thermal engine (MCP Python SDK v2, MCPServer).

Thin layer: every number comes from the engine in this package. The server validates inputs,
calls the engine, and formats results. It never computes temperatures itself.

Run over stdio:
    uv run python -m cubesat_thermal.server

Register in Claude Code:
    claude mcp add cubesat-thermal -- uv run --directory <repo> python -m cubesat_thermal.server

Environment variables (read here: the v2 SDK no longer reads MCP_* settings itself):
    CUBESAT_THERMAL_MODEL   path to model.yaml (default: <repo>/model.yaml)
    CUBESAT_THERMAL_OUT     output folder for plots and reports (default: <repo>/results/mcp)
"""
from __future__ import annotations

import dataclasses
import os
import threading
from datetime import date
from pathlib import Path
from typing import Any, Literal

import numpy as np
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

from . import __version__
from .analysis import beta_sweep as _beta_sweep
from .analysis import check_margins as _check_margins
from .analysis import size_heater as _size_heater
from .model import Case, Result, ThermalModel
from .orbit import (critical_beta_deg, earth_view_factors, eclipse_fraction, period_s,
                    sso_beta_range)
from .plotting import plot_beta_sweep, plot_temperatures

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.environ.get("CUBESAT_THERMAL_MODEL", ROOT / "model.yaml"))
OUT_DIR = Path(os.environ.get("CUBESAT_THERMAL_OUT", ROOT / "results" / "mcp"))

BASELINE = "baseline"
MAX_SWEEP_POINTS = 61                 # ~0.5 s per case run
CASE_OVERRIDE_KEYS = {"solar", "albedo", "olr", "optics"}
OPTICS_MODES = ("eol_not_converting", "bol_converting")
COATING_PROPS = ("alpha_bol", "alpha_eol", "eps", "eta")

KNOWN_LIMITS = [
    "No internal radiation (owner's decision, 2026-11-10)",
    "Isothermal nodes",
    "Albedo uses a cos(zeta) approximation with no terminator correction",
    "Constant dissipation per case",
    "Preliminary-design accuracy, about +/-5-10 degC",
]
REFERENCES = [
    "NASA, Passive Thermal Control Engineering Guidebook, rev. 3.1, 2022: orbits, worst cases, "
    "margins, contact conductances, thermostats",
    "H. Peabody, Thermal Design for Spaceflight, NASA GSFC course, 2022: environment values, "
    "hot/cold biasing, 5 degC and 70 % margin practice, radiator and heater hand calculations",
    "Per-value sources: comments in model.yaml (resource model://cubesat-thermal/model.yaml)",
]

mcp = MCPServer(
    "cubesat-thermal",
    instructions=(
        "Orbital thermal analysis of a 3U CubeSat (nodal model, preliminary design). "
        "Temperatures are in degC. Every result carries its assumptions and the orbit "
        "energy-balance closure: check the closure before trusting temperatures. "
        "Design changes are made with modify_design, which stores a named variant; the "
        "baseline model is never changed."
    ),
    version=__version__,
)

# ---------------------------------------------------------------------- state
_lock = threading.RLock()             # sync tools run in worker threads
_base = ThermalModel.from_yaml(MODEL_PATH)
_variants: dict[str, ThermalModel] = {BASELINE: _base}
_changes: dict[str, list[str]] = {BASELINE: []}
_cache: dict[tuple, Result] = {}


# ---------------------------------------------------------------------- input validation
def _model(variant: str | None) -> tuple[str, ThermalModel]:
    name = variant or BASELINE
    if name not in _variants:
        raise ToolError(f"Unknown variant {name!r}. Available: {sorted(_variants)}")
    return name, _variants[name]


def _check_case(model: ThermalModel, case: str) -> None:
    if case not in model.cases:
        raise ToolError(f"Unknown case {case!r}. Available: {sorted(model.cases)}")


def _check_node(model: ThermalModel, node: str) -> None:
    if node not in model.nodes:
        raise ToolError(f"Unknown node {node!r}. Available: {model.nodes}")


def _check_beta(beta_deg: float) -> None:
    if not -90.0 <= beta_deg <= 90.0:
        raise ToolError(f"beta_deg must be within [-90, 90] deg, got {beta_deg}")


def _resolve_case(model: ThermalModel, case: str, beta_deg: float | None,
                  overrides: dict[str, Any] | None) -> Case:
    _check_case(model, case)
    changes: dict[str, Any] = {}
    if beta_deg is not None:
        _check_beta(beta_deg)
        changes["beta_deg"] = float(beta_deg)
    for key, value in (overrides or {}).items():
        if key not in CASE_OVERRIDE_KEYS:
            raise ToolError(f"Unknown override {key!r}. Allowed: {sorted(CASE_OVERRIDE_KEYS)} "
                            "(use beta_deg for the beta angle, modify_design for dissipation)")
        if key == "optics":
            if value not in OPTICS_MODES:
                raise ToolError(f"optics must be one of {OPTICS_MODES}, got {value!r}")
            changes[key] = value
            continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise ToolError(f"Override {key!r} must be a number, got {value!r}") from None
        if key == "albedo" and not 0.0 <= v <= 1.0:
            raise ToolError(f"albedo must be within [0, 1], got {v}")
        if key in ("solar", "olr") and v < 0.0:
            raise ToolError(f"{key} must be >= 0 W/m2, got {v}")
        changes[key] = v
    return dataclasses.replace(model.cases[case], **changes)


def _run(variant: str | None, case: str, beta_deg: float | None = None,
         overrides: dict[str, Any] | None = None) -> tuple[str, ThermalModel, Result]:
    with _lock:
        vname, model = _model(variant)
        c = _resolve_case(model, case, beta_deg, overrides)
        key = (vname, dataclasses.astuple(c).__repr__())
        if key not in _cache:
            _cache[key] = model.run(c)
        return vname, model, _cache[key]


# ---------------------------------------------------------------------- formatting
def _r(x: float, nd: int = 2) -> float:
    return round(float(x), nd)


def _assumptions(vname: str, model: ThermalModel, case: Case) -> dict[str, Any]:
    return {
        "model": model.name,
        "variant": vname,
        "variant_changes": _changes[vname],
        "case": {"name": case.name, "beta_deg": case.beta_deg, "solar_W_m2": case.solar,
                 "albedo": case.albedo, "olr_W_m2": case.olr, "optics": case.optics,
                 "dissipation_W": case.dissipation},
        "altitude_km": model.altitude_m / 1e3,
        "analysis_uncertainty_c": model.uncertainty_c,
        "heaters": [{"name": h.name, "node": h.node, "power_w": h.power_w,
                     "thermostat_c": [h.on_c, h.off_c]} for h in model.heaters],
        "known_limits": KNOWN_LIMITS,
    }


def _closure(r: Result) -> dict[str, Any]:
    e = r.energy
    return {"closure_pct": round(e["closure_pct"], 4), "in_J": _r(e["in_J"], 1),
            "out_J": _r(e["out_J"], 1), "dU_J": _r(e["dU_J"], 1),
            "orbits_to_converge": r.orbits, "converged": r.converged}


def _out(name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR / name


def _slug(*parts: Any) -> str:
    s = "_".join(str(p) for p in parts if p is not None)
    return "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in s)


# ---------------------------------------------------------------------- tools
@mcp.tool()
def describe_model(variant: str | None = None) -> dict[str, Any]:
    """Describe the thermal model: nodes, conductors, coatings, design cases, limits, heaters
    and sources. Use variant to describe a design variant created with modify_design."""
    vname, m = _model(variant)
    nodes = {}
    for n in m.nodes:
        d = m.cfg["nodes"][n]
        nodes[n] = {"C_J_per_K": d["C"], "external": "face" in d}
        if "face" in d:
            nodes[n].update(face=d["face"], area_m2=d["area"], coatings=d["coatings"])
    return {
        "name": m.name,
        "variant": vname,
        "variant_changes": _changes[vname],
        "variants_available": sorted(_variants),
        "orbit": m.cfg["orbit"],
        "nodes": nodes,
        "conductors_W_per_K": [{"a": a, "b": b, "G": g} for a, b, g in m.cfg["conductors"]],
        "coatings": m.coatings,
        "cases": {k: dataclasses.asdict(c) for k, c in m.cases.items()},
        "limits_c": m.limits,
        "analysis_uncertainty_c": m.uncertainty_c,
        "heaters": [dataclasses.asdict(h) for h in m.heaters],
        "internal_radiation": False,
        "known_limits": KNOWN_LIMITS,
        "sources": REFERENCES,
    }


@mcp.tool()
def orbit_info(altitude_km: float, beta_deg: float,
               ltdn_hours: float | None = None) -> dict[str, Any]:
    """Circular-orbit geometry: period, eclipse fraction, critical beta, flat-plate Earth view
    factors, and the yearly beta range of a sun-synchronous orbit with the given local time of
    descending node (default: the model's LTDN)."""
    if not 160.0 <= altitude_km <= 2000.0:
        raise ToolError(f"altitude_km must be within [160, 2000] km (LEO), got {altitude_km}")
    _check_beta(beta_deg)
    ltdn = float(_base.cfg["orbit"].get("ltdn_hours", 10.5)) if ltdn_hours is None else ltdn_hours
    if not 0.0 <= ltdn < 24.0:
        raise ToolError(f"ltdn_hours must be within [0, 24) h, got {ltdn}")
    h = altitude_km * 1e3
    P = period_s(h)
    f_ecl = eclipse_fraction(h, beta_deg)
    lo, hi = sso_beta_range(h, ltdn)
    return {
        "altitude_km": altitude_km, "beta_deg": beta_deg,
        "period_min": _r(P / 60), "eclipse_fraction": round(f_ecl, 4),
        "eclipse_min": _r(f_ecl * P / 60), "critical_beta_deg": _r(critical_beta_deg(h)),
        "earth_view_factors": {k: round(v, 4) for k, v in earth_view_factors(h).items()},
        "sso_beta_range_deg": {"ltdn_hours": ltdn, "min": _r(lo, 1), "max": _r(hi, 1)},
        "assumptions": ["Circular orbit, spherical Earth, cylindrical shadow",
                        "SSO inclination from J2 only; low-precision solar ephemeris"],
    }


@mcp.tool()
def run_case(case: str, beta_deg: float | None = None,
             overrides: dict[str, Any] | None = None,
             variant: str | None = None) -> dict[str, Any]:
    """Run a design case (e.g. 'hot', 'cold') to a periodic orbit and report min/max
    temperature per node [degC], orbits to converge and energy-balance closure.
    beta_deg replaces the case beta angle. overrides may set solar [W/m2], albedo [-],
    olr [W/m2] or optics ('eol_not_converting' | 'bol_converting')."""
    vname, m, r = _run(variant, case, beta_deg, overrides)
    return {
        "temperatures_c": {n: {"min": _r(r.tmin(n)), "max": _r(r.tmax(n))} for n in r.nodes},
        "period_min": _r(r.period_s / 60),
        "eclipse_fraction": round(float(r.eclipse.mean()), 4),
        "heater_duty": {k: round(v, 3) for k, v in r.energy["heater_duty"].items()},
        "energy_balance": _closure(r),
        "assumptions": _assumptions(vname, m, r.case),
    }


@mcp.tool()
def energy_balance(case: str, variant: str | None = None) -> dict[str, Any]:
    """Orbit-integrated energy balance per external face: absorbed solar, albedo and Earth IR,
    emitted to space (orbit averages in W), plus dissipation, heater energy and closure %."""
    vname, m, r = _run(variant, case)
    e, P = r.energy, r.period_s
    faces = {f: {"solar_W": _r(v["solar"] / P, 3), "albedo_W": _r(v["albedo"] / P, 3),
                 "ir_W": _r(v["ir"] / P, 3), "absorbed_W": _r(e["absorbed_avg_W"][f], 3),
                 "emitted_W": _r(e["emitted_avg_W"][f], 3)}
             for f, v in e["absorbed_J"].items()}
    return {
        "faces_orbit_average": faces,
        "dissipated_W": _r(e["dissipated_J"] / P, 3),
        "heater_W": _r(e["heater_J"] / P, 3),
        "total_in_W": _r(e["in_J"] / P, 3),
        "total_out_W": _r(e["out_J"] / P, 3),
        "energy_balance": _closure(r),
        "assumptions": _assumptions(vname, m, r.case),
    }


@mcp.tool()
def check_margins(case: str, variant: str | None = None) -> dict[str, Any]:
    """Check predicted temperatures against operating limits with the analysis uncertainty.
    PASS: prediction +/- uncertainty inside limits. WATCH: inside limits but closer than the
    uncertainty. FAIL: outside operating limits."""
    vname, m, r = _run(variant, case)
    rows = _check_margins(m, r)
    worst = "FAIL" if any(x["status"] == "FAIL" for x in rows) else (
        "WATCH" if any(x["status"] == "WATCH" for x in rows) else "PASS")
    return {"overall": worst, "margins": rows, "energy_balance": _closure(r),
            "assumptions": _assumptions(vname, m, r.case)}


@mcp.tool()
def beta_sweep(case: str, start: float = 0.0, stop: float = 90.0, step: float = 10.0,
               variant: str | None = None) -> dict[str, Any]:
    """Run a case over a range of beta angles [deg] (stop included) and plot orbit min/max
    temperature of the limited nodes. Returns the table and the PNG path."""
    vname, m = _model(variant)
    _check_case(m, case)
    _check_beta(start)
    _check_beta(stop)
    if step <= 0:
        raise ToolError(f"step must be > 0 deg, got {step}")
    if stop < start:
        raise ToolError(f"stop ({stop}) must be >= start ({start})")
    betas = np.arange(start, stop + step * 1e-6, step)
    if len(betas) > MAX_SWEEP_POINTS:
        raise ToolError(f"{len(betas)} beta points requested; the limit is {MAX_SWEEP_POINTS}. "
                        "Use a larger step or a narrower range.")
    with _lock:
        rows = _beta_sweep(m, case, betas)
        nodes = [n for n in ("Battery", "Avionics", "Payload") if n in m.limits]
        path = _out(_slug("beta_sweep", case, vname, f"{start:g}-{stop:g}") + ".png")
        plot_beta_sweep(rows, nodes=nodes, path=path)
    return {"rows": rows, "plot_png": str(path),
            "assumptions": _assumptions(vname, m, m.cases[case]),
            "note": "Each point is a full periodic-orbit run; closure is checked per run_case."}


@mcp.tool()
def size_heater(node: str, case: str = "cold", battery_mount_g: float | None = None,
                target_c: float | None = None, variant: str | None = None) -> dict[str, Any]:
    """Size a thermostatic heater for a node: the constant power that keeps it at or above the
    target (default: operating minimum + uncertainty), installed power = required / max duty
    (70 %), verified with the thermostat. battery_mount_g [W/K] replaces the node-to-Frame
    conductance for this sizing only (e.g. 0.05 for an isolated battery mount)."""
    vname, m = _model(variant)
    _check_node(m, node)
    _check_case(m, case)
    if node not in m.limits:
        raise ToolError(f"Node {node!r} has no operating limits. Nodes with limits: "
                        f"{sorted(m.limits)}")
    if not m.heaters:
        raise ToolError("The model has no heater defined in model.yaml")
    if any(h.node != node for h in m.heaters):
        raise ToolError(f"The model heater is mounted on {m.heaters[0].node!r}, not {node!r}")
    note = None
    if battery_mount_g is not None:
        if battery_mount_g <= 0:
            raise ToolError(f"battery_mount_g must be > 0 W/K, got {battery_mount_g}")
        m = m.copy_with(conductor=(node, "Frame", battery_mount_g))
        note = f"{node}-Frame conductance set to {battery_mount_g} W/K for this sizing"
    with _lock:
        try:
            out = _size_heater(m, case=case, node=node, target_c=target_c)
        except RuntimeError as exc:
            raise ToolError(f"Heater sizing failed: {exc}") from exc
        r = m.copy_with(heater_power=out["installed_w"]).run(case)
    assumptions = _assumptions(vname, m, r.case)
    if note:
        assumptions["variant_changes"] = [*assumptions["variant_changes"], note]
    return {**out, "energy_balance": _closure(r), "assumptions": assumptions}


class ConductorChange(BaseModel):
    node_a: str
    node_b: str
    g_w_per_k: float = Field(gt=0, description="Linear conductance [W/K]")


class CoatingChange(BaseModel):
    coating: str
    prop: Literal["alpha_bol", "alpha_eol", "eps", "eta"]
    value: float = Field(ge=0, le=1)


class DissipationChange(BaseModel):
    case: str
    node: str
    watts: float = Field(ge=0)


@mcp.tool()
def modify_design(name: str, conductor: ConductorChange | None = None,
                  coating: CoatingChange | None = None,
                  dissipation: DissipationChange | None = None,
                  heater_power_w: float | None = None,
                  base: str | None = None) -> dict[str, Any]:
    """Create a named design variant from the baseline (or from another variant via base) by
    changing one or more of: a conductor (an existing pair is replaced, a new pair is added),
    a coating property, a case dissipation, or the heater power. The baseline is never changed.
    Use the variant name in the other tools."""
    if not name or name == BASELINE:
        raise ToolError(f"Variant name must be non-empty and not {BASELINE!r}")
    if not any(x is not None for x in (conductor, coating, dissipation, heater_power_w)):
        raise ToolError("Give at least one change: conductor, coating, dissipation or "
                        "heater_power_w")
    with _lock:
        bname, m = _model(base)
        changes, log = {}, []
        if conductor is not None:
            _check_node(m, conductor.node_a)
            _check_node(m, conductor.node_b)
            if conductor.node_a == conductor.node_b:
                raise ToolError("A conductor needs two different nodes")
            changes["conductor"] = (conductor.node_a, conductor.node_b, conductor.g_w_per_k)
            log.append(f"conductor {conductor.node_a}-{conductor.node_b} = "
                       f"{conductor.g_w_per_k} W/K")
        if coating is not None:
            if coating.coating not in m.coatings:
                raise ToolError(f"Unknown coating {coating.coating!r}. Available: "
                                f"{sorted(m.coatings)}")
            if coating.prop not in m.coatings[coating.coating]:
                raise ToolError(f"Coating {coating.coating!r} has no {coating.prop!r}; it has "
                                f"{sorted(m.coatings[coating.coating])}")
            changes["coating"] = (coating.coating, coating.prop, coating.value)
            log.append(f"coating {coating.coating}.{coating.prop} = {coating.value}")
        if dissipation is not None:
            _check_case(m, dissipation.case)
            _check_node(m, dissipation.node)
            changes["dissipation"] = (dissipation.case, dissipation.node, dissipation.watts)
            log.append(f"dissipation {dissipation.case}/{dissipation.node} = "
                       f"{dissipation.watts} W")
        if heater_power_w is not None:
            if heater_power_w < 0:
                raise ToolError(f"heater_power_w must be >= 0, got {heater_power_w}")
            if not m.heaters:
                raise ToolError("The model has no heater defined in model.yaml")
            changes["heater_power"] = heater_power_w
            log.append(f"heater power = {heater_power_w} W")
        new = m.copy_with(**changes)
        replaced = name in _variants
        _variants[name] = new
        _changes[name] = [*_changes[bname], *log]
        for key in [k for k in _cache if k[0] == name]:
            del _cache[key]
    return {"variant": name, "base": bname, "changes": _changes[name],
            "replaced_existing": replaced, "variants_available": sorted(_variants)}


@mcp.tool()
def plot(case: str, nodes: list[str] | None = None,
         variant: str | None = None) -> dict[str, Any]:
    """Plot node temperatures over one periodic orbit with eclipse shading, operating limits
    and heater state. Returns the PNG path."""
    vname, m = _model(variant)
    for n in nodes or []:
        _check_node(m, n)
    vname, m, r = _run(variant, case)
    with _lock:
        path = _out(_slug("temperatures", case, vname) + ".png")
        plot_temperatures(m, r, nodes=nodes or None, path=path)
    return {"plot_png": str(path), "energy_balance": _closure(r),
            "assumptions": _assumptions(vname, m, r.case)}


@mcp.tool()
def generate_report(cases: list[str], variant: str | None = None) -> dict[str, Any]:
    """Write a Markdown thermal report for the given cases with assumptions, temperatures,
    margins, energy-balance closure, known limits and sources. Returns the file path and text."""
    if not cases:
        raise ToolError("Give at least one case")
    vname, m = _model(variant)
    for c in cases:
        _check_case(m, c)
    L = [f"# Thermal analysis report: {m.name}", "",
         f"Variant: **{vname}**. Engine cubesat-thermal {__version__}. "
         f"Generated {date.today().isoformat()}.", ""]
    L += ["## Assumptions", "",
          f"- Altitude {m.altitude_m / 1e3:g} km, analysis uncertainty "
          f"+/-{m.uncertainty_c:g} degC on operating limits"]
    L += [f"- Variant change: {c}" for c in _changes[vname]]
    L += [f"- {x}" for x in KNOWN_LIMITS] + [""]
    for c in cases:
        _, _, r = _run(variant, c)
        cs, e = r.case, r.energy
        L += [f"## Case `{c}`", "",
              f"beta {cs.beta_deg:g} deg, solar {cs.solar:g} W/m2, albedo {cs.albedo:g}, "
              f"OLR {cs.olr:g} W/m2, optics `{cs.optics}`, dissipation {cs.dissipation} W.", "",
              f"Energy balance: closure **{e['closure_pct']:.4f} %** (in {e['in_J']:.0f} J, "
              f"out {e['out_J']:.0f} J, dU {e['dU_J']:.1f} J), {r.orbits} orbits to converge"
              f"{'' if r.converged else ' (NOT converged)'}.", "",
              "| Node | Tmin [degC] | Tmax [degC] | Op limits [degC] | Cold margin | "
              "Hot margin | Status |",
              "|---|---|---|---|---|---|---|"]
        for row in _check_margins(m, r):
            lo, hi = row["op_limits_c"]
            L.append(f"| {row['node']} | {row['tmin_c']} | {row['tmax_c']} | {lo} / {hi} | "
                     f"{row['margin_cold_c']} | {row['margin_hot_c']} | {row['status']} |")
        if e["heater_duty"]:
            L += ["", "Heater duty: " + ", ".join(f"{k} {v:.1%}"
                                                    for k, v in e["heater_duty"].items())]
        L.append("")
    L += ["## Sources", ""] + [f"- {s}" for s in REFERENCES] + [""]
    text = "\n".join(L)
    path = _out(_slug("report", vname, *cases) + ".md")
    path.write_text(text, encoding="utf-8")
    return {"report_md": str(path), "markdown": text}


# ---------------------------------------------------------------------- resource and prompt
@mcp.resource("model://cubesat-thermal/model.yaml", name="model.yaml",
              description="Baseline thermal model with per-value sources (read-only)",
              mime_type="application/yaml")
def model_yaml() -> str:
    return MODEL_PATH.read_text(encoding="utf-8")


@mcp.prompt(title="Worst-case thermal analysis")
def worst_case_analysis(variant: str = BASELINE) -> str:
    """Guided hot/cold worst-case analysis with margins, beta sensitivity and heater sizing."""
    return f"""Run a worst-case thermal analysis of the CubeSat model (variant: {variant}).

1. Call describe_model to state the model, design cases and limits.
2. Run the hot and cold cases with run_case. Before using any temperature, check that the
   energy-balance closure is below 0.1 % and that the run converged.
3. Call check_margins for both cases and list every WATCH or FAIL node with its margin.
4. Call beta_sweep on the hot case over the SSO beta range from orbit_info (and 0-90 deg)
   to confirm that the design-case beta is really the worst case.
5. If a node fails cold, size its heater with size_heater, and compare against an isolated
   mount (battery_mount_g=0.05) if it is the battery.
6. Close with generate_report for hot and cold and summarise: worst nodes, margins,
   heater power, and the model limits that affect the conclusions.

Report temperatures in degC and always quote the assumptions behind each number."""


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
