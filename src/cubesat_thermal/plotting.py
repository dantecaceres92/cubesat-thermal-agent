"""Standard plots: temperatures over one orbit, energy balance per face, beta sweep."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .model import Result, ThermalModel  # noqa: E402

PALETTE = ["#2a6fdb", "#e4572e", "#17a398", "#8e5cd9", "#f2a541", "#5c6b7a"]


def _shade_eclipse(ax, result: Result):
    t = result.t / 60
    e = result.eclipse.astype(int)
    starts = list(np.where(np.diff(e) == 1)[0] + 1)
    ends = list(np.where(np.diff(e) == -1)[0] + 1)
    if e[0]:
        starts = [0] + starts
    if e[-1]:
        ends = ends + [len(t) - 1]
    for k, (s, f) in enumerate(zip(starts, ends)):
        ax.axvspan(t[s], t[f], color="0.88", zorder=0, label="Eclipse" if k == 0 else None)


def plot_temperatures(model: ThermalModel, result: Result, nodes=None, path=None):
    nodes = nodes or [n for n in ("Battery", "Avionics", "Payload") if n in model.nodes]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    _shade_eclipse(ax, result)
    for k, n in enumerate(nodes):
        c = PALETTE[k % len(PALETTE)]
        ax.plot(result.t / 60, result.T[n], color=c, lw=2, label=n)
        if n in model.limits:
            lo, hi = model.limits[n]["op"]
            ax.axhline(lo, color=c, ls="--", lw=0.8, alpha=0.6)
            ax.axhline(hi, color=c, ls="--", lw=0.8, alpha=0.6)
    if result.heater_on.any():
        ax.fill_between(result.t / 60, ax.get_ylim()[0], ax.get_ylim()[0] + 1.5,
                        where=result.heater_on, color="#e4572e", alpha=0.6, step="mid",
                        label="Heater on")
    c = result.case
    ax.set_title(f"{c.name} case: beta = {c.beta_deg:.1f} deg, S = {c.solar:.0f} W/m2 "
                 f"(dashed: operating limits)", fontsize=10)
    ax.set_xlabel("Time in orbit [min]")
    ax.set_ylabel("Temperature [degC]")
    ax.set_xlim(0, result.period_s / 60)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=3, loc="best")
    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig


def plot_energy_balance(model: ThermalModel, result: Result, path=None):
    e = result.energy
    P = result.period_s
    faces = list(e["absorbed_J"].keys())
    x = np.arange(len(faces))
    sol = [e["absorbed_J"][f]["solar"] / P for f in faces]
    alb = [e["absorbed_J"][f]["albedo"] / P for f in faces]
    ir = [e["absorbed_J"][f]["ir"] / P for f in faces]
    emi = [e["emitted_J"][f] / P for f in faces]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.bar(x - 0.2, sol, 0.38, color="#f2a541", label="Solar")
    ax.bar(x - 0.2, alb, 0.38, bottom=sol, color="#17a398", label="Albedo")
    ax.bar(x - 0.2, ir, 0.38, bottom=np.add(sol, alb), color="#8e5cd9", label="Earth IR")
    ax.bar(x + 0.2, emi, 0.38, color="#5c6b7a", label="Emitted to space")
    ax.set_xticks(x, faces)
    ax.set_ylabel("Orbit-average heat [W]")
    ax.set_title(f"{result.case.name} case energy balance, closure {e['closure_pct']:.3f} %",
                 fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig


def plot_beta_sweep(rows, nodes=("Battery", "Avionics", "Payload"), path=None):
    b = [r["beta_deg"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    for k, n in enumerate(nodes):
        c = PALETTE[k % len(PALETTE)]
        ax.fill_between(b, [r[f"{n}_min"] for r in rows], [r[f"{n}_max"] for r in rows],
                        color=c, alpha=0.25)
        ax.plot(b, [r[f"{n}_max"] for r in rows], color=c, lw=2, label=f"{n} max")
        ax.plot(b, [r[f"{n}_min"] for r in rows], color=c, lw=1, ls=":")
    ax.set_xlabel("Beta angle [deg]")
    ax.set_ylabel("Temperature [degC]")
    ax.set_title("Beta sweep: orbit min (dotted) to max (solid)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig
