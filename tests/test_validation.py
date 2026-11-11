"""Validation suite V1-V6 (docs/SPEC.md, section 10)."""
from pathlib import Path

import numpy as np
import pytest

from cubesat_thermal import (ThermalModel, eclipse_fraction, period_s, radiator_area,
                             radiator_heater)
from cubesat_thermal.environment import Environment

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def model():
    return ThermalModel.from_yaml(ROOT / "model.yaml")


def test_v1_gsfc_radiator_area():
    """Peabody 2022 p.124: 40 W box at 35 degC (308 K), eps 0.85, F_earth 0.2, Earth flux 243 W/m2 -> ~0.1 m2."""
    area = radiator_area(40.0, 308.0, 0.85, 0.2, 243.0)
    assert area == pytest.approx(0.1019, rel=0.01)
    assert area == pytest.approx(0.10, abs=0.005)


def test_v2_gsfc_heater():
    """Peabody 2022 p.124: 15 W, 0.1 m2, held at -5 degC (268 K) with Earth flux 208 W/m2.
    The course quotes ~6.7 W from rounded inputs; the exact evaluation of its own equation gives 6.33 W."""
    q = radiator_heater(15.0, 0.1, 268.0, 0.85, 0.2, 208.0)
    assert q == pytest.approx(6.33, rel=0.01)


@pytest.mark.parametrize("beta, minutes", [(0.0, 35.55), (60.0, 20.49)])
def test_v3_eclipse_duration(beta, minutes):
    """Analytic eclipse duration, and the environment's shadow test agrees with it."""
    h = 550e3
    P = period_s(h) / 60
    assert eclipse_fraction(h, beta) * P == pytest.approx(minutes, abs=0.2)
    env = Environment(h, beta, 1367, 0.3, 240)
    theta = np.linspace(0, 2 * np.pi, 20001)[:-1]
    numeric = np.mean([env.in_eclipse(t) for t in theta]) * P
    assert numeric == pytest.approx(minutes, abs=0.2)


def test_v4_sun_normal_plate():
    """Single plate always normal to the Sun, insulated back: T = (alpha S / eps sigma)^0.25 = 125.6 degC."""
    cfg = {
        "orbit": {"altitude_km": 550},
        "coatings": {"cell": {"alpha_bol": 0.86, "alpha_eol": 0.86, "eps": 0.82}},
        "nodes": {"plate": {"C": 50, "face": "+Y", "area": 0.01, "coatings": {"cell": 1.0}}},
        "conductors": [],
        "cases": {"sun": {"beta_deg": 90.0, "solar": 1367, "albedo": 0.0, "olr": 0.0,
                          "optics": "eol_not_converting", "dissipation": {}}},
    }
    r = ThermalModel(cfg).run("sun")
    assert r.tmax("plate") == pytest.approx(125.6, abs=0.5)
    assert r.tmin("plate") == pytest.approx(125.6, abs=0.5)


@pytest.mark.parametrize("case", ["hot", "cold"])
def test_v5_energy_balance_closes(model, case):
    r = model.run(case)
    assert r.converged
    assert r.energy["closure_pct"] < 0.5


def test_v5_energy_balance_with_thermostat(model):
    m = model.copy_with(heater_power=2.0)
    r = m.run("cold")
    assert r.energy["heater_J"] > 0
    assert r.energy["closure_pct"] < 0.5


@pytest.mark.parametrize("case", ["hot", "cold"])
def test_v6_time_step_independence(model, case):
    a = model.run(case)
    b = model.run(case, rtol=1e-9, atol=1e-8, n_out=1441)
    for node in model.nodes:
        assert abs(a.tmin(node) - b.tmin(node)) < 0.1
        assert abs(a.tmax(node) - b.tmax(node)) < 0.1
