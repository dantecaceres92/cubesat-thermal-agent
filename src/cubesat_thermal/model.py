"""Nodal thermal model: network assembly and transient orbital solution.

C_i dT_i/dt = Q_env,i(t) + Q_diss,i + Q_heater,i - eps_i A_i sigma (T_i^4 - T_space^4) - sum_j G_ij (T_i - T_j)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from scipy.integrate import solve_ivp
from scipy.optimize import root

from .environment import FACES, Environment
from .orbit import period_s

SIGMA = 5.670374419e-8
T_SPACE = 4.0          # deep-space sink [K]
K0 = 273.15
N_ENV = 1440           # environment samples per orbit (15 deg/60 = 0.25 deg)


@dataclass
class Case:
    name: str
    beta_deg: float
    solar: float
    albedo: float
    olr: float
    optics: str                       # "eol_not_converting" | "bol_converting"
    dissipation: dict[str, float] = field(default_factory=dict)


@dataclass
class Heater:
    name: str
    node: str
    power_w: float
    on_c: float
    off_c: float
    max_duty: float = 0.70


@dataclass
class Result:
    case: Case
    nodes: list[str]
    period_s: float
    t: np.ndarray                     # [s], last (periodic) orbit
    T: dict[str, np.ndarray]          # [degC]
    eclipse: np.ndarray               # bool per sample
    heater_on: np.ndarray             # bool per sample (first heater)
    energy: dict                      # orbit-integrated energy balance [J] and averages [W]
    orbits: int
    converged: bool

    def tmin(self, node: str) -> float:
        return float(self.T[node].min())

    def tmax(self, node: str) -> float:
        return float(self.T[node].max())

    def summary(self) -> dict:
        return {n: (round(self.tmin(n), 2), round(self.tmax(n), 2)) for n in self.nodes}


class ThermalModel:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.name = cfg.get("name", "model")
        self.altitude_m = float(cfg["orbit"]["altitude_km"]) * 1e3
        self.nodes: list[str] = list(cfg["nodes"].keys())
        self.idx = {n: i for i, n in enumerate(self.nodes)}
        self.C = np.array([float(cfg["nodes"][n]["C"]) for n in self.nodes])
        self.coatings = cfg.get("coatings", {})
        self.ext = {n: d for n, d in cfg["nodes"].items() if "face" in d}
        if cfg.get("internal_radiation", False):
            raise NotImplementedError("internal radiation is not part of model v0.1")
        n = len(self.nodes)
        self.L = np.zeros((n, n))
        for a, b, g in cfg.get("conductors", []):
            i, j = self.idx[a], self.idx[b]
            self.L[i, i] += g; self.L[j, j] += g
            self.L[i, j] -= g; self.L[j, i] -= g
        self.cases = {k: Case(name=k, **v) for k, v in cfg.get("cases", {}).items()}
        self.heaters = [Heater(name=k, **v) for k, v in (cfg.get("heaters") or {}).items()]
        self.limits = cfg.get("limits", {})
        self.uncertainty_c = float(cfg.get("uncertainty_c", 5.0))

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_yaml(cls, path: str | Path) -> "ThermalModel":
        with open(path, encoding="utf-8") as fh:
            return cls(yaml.safe_load(fh))

    def copy_with(self, **changes) -> "ThermalModel":
        """Return a modified copy. Supported keys: conductor=(a, b, G), heater_power=W,
        coating=(name, prop, value), dissipation=(case, node, W)."""
        cfg = copy.deepcopy(self.cfg)
        if "conductor" in changes:
            a, b, g = changes["conductor"]
            for c in cfg["conductors"]:
                if {c[0], c[1]} == {a, b}:
                    c[2] = float(g)
                    break
            else:
                cfg["conductors"].append([a, b, float(g)])
        if "heater_power" in changes:
            for h in cfg["heaters"].values():
                h["power_w"] = float(changes["heater_power"])
        if "coating" in changes:
            name, prop, value = changes["coating"]
            cfg["coatings"][name][prop] = float(value)
        if "dissipation" in changes:
            case, node, w = changes["dissipation"]
            cfg["cases"][case]["dissipation"][node] = float(w)
        return ThermalModel(cfg)

    # ------------------------------------------------------------------ properties
    def face_optics(self, node: str, mode: str) -> tuple[float, float]:
        """Area-weighted effective (alpha, eps) of an external node."""
        alpha = eps = 0.0
        for coat, frac in self.ext[node]["coatings"].items():
            p = self.coatings[coat]
            if mode == "eol_not_converting":
                a = p["alpha_eol"]
            elif mode == "bol_converting":
                a = p["alpha_bol"] - p.get("eta", 0.0)
            else:
                raise ValueError(f"unknown optics mode {mode!r}")
            alpha += frac * a
            eps += frac * p["eps"]
        return alpha, eps

    def environment(self, case: Case) -> Environment:
        return Environment(self.altitude_m, case.beta_deg, case.solar, case.albedo, case.olr)

    def env_table(self, case: Case):
        """Absorbed environmental heat per node on a theta grid: arrays (theta, Q[n, k], parts, eclipse)."""
        env = self.environment(case)
        theta = np.linspace(0.0, 2 * np.pi, N_ENV + 1)
        Q = np.zeros((len(self.nodes), N_ENV + 1))
        parts = {f: np.zeros((3, N_ENV + 1)) for f in self.ext}
        ecl = np.zeros(N_ENV + 1, dtype=bool)
        eps_vec = np.zeros(len(self.nodes))
        area = np.zeros(len(self.nodes))
        optics = {nd: self.face_optics(nd, case.optics) for nd in self.ext}
        for nd, (a, e) in optics.items():
            eps_vec[self.idx[nd]] = e
            area[self.idx[nd]] = self.ext[nd]["area"]
        for k, th in enumerate(theta):
            inc = env.incident(th)
            ecl[k] = env.in_eclipse(th)
            for nd, d in self.ext.items():
                a, e = optics[nd]
                qs, qa, qi = inc[d["face"]]
                A = d["area"]
                parts[nd][:, k] = (a * qs * A, a * qa * A, e * qi * A)
                Q[self.idx[nd], k] = parts[nd][:, k].sum()
        return theta, Q, parts, ecl, eps_vec, area

    # ------------------------------------------------------------------ solution
    def run(self, case: Case | str, max_orbits: int = 40, tol_k: float = 0.01,
            n_out: int = 721, rtol: float = 1e-7, atol: float = 1e-6) -> Result:
        case = self.cases[case] if isinstance(case, str) else case
        P = period_s(self.altitude_m)
        theta, Qtab, parts, ecl_tab, eps_vec, area = self.env_table(case)
        ttab = theta / (2 * np.pi) * P
        n = len(self.nodes)
        q_diss = np.zeros(n)
        for nd, w in case.dissipation.items():
            q_diss[self.idx[nd]] += w
        rad = eps_vec * area * SIGMA
        heaters = [h for h in self.heaters if h.power_w > 0]
        h_idx = [self.idx[h.node] for h in heaters]
        h_pow = np.array([h.power_w for h in heaters])

        def q_env(t):
            tt = t % P
            return np.array([np.interp(tt, ttab, Qtab[i]) for i in range(n)])

        def rhs(t, T, state):
            q = q_env(t) + q_diss - rad * (T**4 - T_SPACE**4) - self.L @ T
            for k, on in enumerate(state):
                if on:
                    q[h_idx[k]] += h_pow[k]
            return q / self.C

        # initial condition: steady state with orbit-averaged loads
        q_avg = np.trapezoid(Qtab, ttab, axis=1) / P
        sol0 = root(lambda T: q_avg + q_diss - rad * (T**4 - T_SPACE**4) - self.L @ T,
                    np.full(n, 280.0), method="hybr")
        T = sol0.x
        state = [T[h_idx[k]] - K0 < h.on_c for k, h in enumerate(heaters)]

        def integrate_orbit(T0, state0, t0, record=False):
            """Integrate one orbit with thermostat events. Returns end state and segments."""
            t, T, st = t0, T0.copy(), list(state0)
            segs = []
            t_end = t0 + P
            while t < t_end - 1e-9:
                events = []
                for k, h in enumerate(heaters):
                    i = h_idx[k]
                    if st[k]:
                        f = (lambda tt, y, *_, i=i, lim=h.off_c + K0: y[i] - lim); f.direction = 1
                    else:
                        f = (lambda tt, y, *_, i=i, lim=h.on_c + K0: y[i] - lim); f.direction = -1
                    f.terminal = True
                    events.append(f)
                s = solve_ivp(rhs, (t, t_end), T, method="BDF", args=(st,), rtol=rtol, atol=atol,
                              max_step=P / 360, events=events or None, dense_output=record)
                if record:
                    segs.append((s.t[0], s.t[-1], s.sol, list(st)))
                t, T = s.t[-1], s.y[:, -1]
                if s.status == 1:
                    for k, te in enumerate(s.t_events):
                        if len(te):
                            st[k] = not st[k]
            return T, st, segs

        converged, orbits = False, 0
        t0 = 0.0
        for orbits in range(1, max_orbits + 1):
            T_new, state, _ = integrate_orbit(T, state, t0)
            t0 += P
            dT = np.max(np.abs(T_new - T))
            T = T_new
            if dT < tol_k:
                converged = True
                break
        T_start, state_start = T.copy(), list(state)
        T_end, _, segs = integrate_orbit(T, state, t0, record=True)

        # sample the recorded orbit
        def sample(tt):
            for a, b, sol, st in segs:
                if a - 1e-9 <= tt <= b + 1e-9:
                    return sol(tt), st
            return segs[-1][2](segs[-1][1]), segs[-1][3]

        t_out = np.linspace(t0, t0 + P, n_out)
        Ys, Hs = zip(*(sample(tt) for tt in t_out))
        Y = np.array(Ys).T
        H = np.array([st[0] if st else False for st in Hs])

        # energy balance over the orbit (fine trapezoid for continuous terms, exact for heaters)
        t_f = np.linspace(t0, t0 + P, 8 * n_out)
        Yf = np.array([sample(tt)[0] for tt in t_f]).T
        tt_rel = t_f - t0
        e_abs = {nd: {
            "solar": float(np.trapezoid(np.interp(tt_rel, ttab, parts[nd][0]), t_f)),
            "albedo": float(np.trapezoid(np.interp(tt_rel, ttab, parts[nd][1]), t_f)),
            "ir": float(np.trapezoid(np.interp(tt_rel, ttab, parts[nd][2]), t_f)),
        } for nd in self.ext}
        e_emit = {nd: float(np.trapezoid(rad[self.idx[nd]] * (Yf[self.idx[nd]] ** 4 - T_SPACE**4), t_f))
                  for nd in self.ext}
        e_diss = float(q_diss.sum() * P)
        on_time = [0.0] * len(heaters)
        for a, b, sol, st in segs:
            for k, on in enumerate(st):
                if on:
                    on_time[k] += b - a
        e_heat = float(sum(on_time[k] * h_pow[k] for k in range(len(heaters))))
        e_in = sum(sum(v.values()) for v in e_abs.values()) + e_diss + e_heat
        e_out = sum(e_emit.values())
        dU = float(self.C @ (T_end - T_start))
        energy = {
            "absorbed_J": e_abs, "emitted_J": e_emit, "dissipated_J": e_diss, "heater_J": e_heat,
            "in_J": e_in, "out_J": e_out, "dU_J": dU,
            "closure_pct": abs(e_in - e_out - dU) / e_in * 100.0,
            "heater_duty": {h.name: on_time[k] / P for k, h in enumerate(heaters)},
            "absorbed_avg_W": {nd: sum(v.values()) / P for nd, v in e_abs.items()},
            "emitted_avg_W": {nd: v / P for nd, v in e_emit.items()},
        }
        ecl = np.interp((t_out - t0), ttab, ecl_tab.astype(float)) > 0.5
        return Result(case=case, nodes=self.nodes, period_s=P, t=t_out - t0,
                      T={nd: Y[self.idx[nd]] - K0 for nd in self.nodes}, eclipse=ecl,
                      heater_on=H, energy=energy, orbits=orbits, converged=converged)
