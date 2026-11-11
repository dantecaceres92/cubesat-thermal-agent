"""Circular-orbit geometry: period, eclipse, beta angle and Earth view factors."""
from __future__ import annotations

import numpy as np

R_EARTH = 6371.0e3          # mean Earth radius [m]
R_EARTH_EQ = 6378.137e3     # equatorial radius, used for J2 precession [m]
MU_EARTH = 3.986004418e14   # [m^3/s^2]
J2 = 1.08263e-3


def period_s(altitude_m: float) -> float:
    """Orbital period of a circular orbit [s]."""
    r = R_EARTH + altitude_m
    return 2.0 * np.pi * np.sqrt(r**3 / MU_EARTH)


def eclipse_fraction(altitude_m: float, beta_deg: float) -> float:
    """Fraction of the orbit in Earth's cylindrical shadow (analytic)."""
    r = R_EARTH + altitude_m
    beta = np.radians(beta_deg)
    beta_crit = np.arcsin(R_EARTH / r)
    if abs(beta) >= beta_crit:
        return 0.0
    x = np.sqrt(altitude_m**2 + 2.0 * R_EARTH * altitude_m) / (r * np.cos(beta))
    return float(np.degrees(np.arccos(x)) / 180.0)


def critical_beta_deg(altitude_m: float) -> float:
    """Beta above which the orbit has no eclipse [deg]."""
    return float(np.degrees(np.arcsin(R_EARTH / (R_EARTH + altitude_m))))


def earth_view_factors(altitude_m: float) -> dict[str, float]:
    """Flat-plate view factors to Earth for nadir, side (horizontal normal) and zenith faces."""
    H = (R_EARTH + altitude_m) / R_EARTH
    X = np.sqrt(H**2 - 1.0)
    f_nadir = 1.0 / H**2
    f_side = (np.arctan(1.0 / X) - X / H**2) / np.pi
    return {"nadir": float(f_nadir), "side": float(f_side), "zenith": 0.0}


def sso_inclination_deg(altitude_m: float) -> float:
    """Inclination that makes a circular orbit sun-synchronous (J2 only)."""
    a = R_EARTH_EQ + altitude_m
    n = np.sqrt(MU_EARTH / a**3)
    omega_dot = 2.0 * np.pi / (365.2422 * 86400.0)
    cos_i = -omega_dot / (1.5 * n * J2 * (R_EARTH_EQ / a) ** 2)
    return float(np.degrees(np.arccos(cos_i)))


def _sun_ra_dec(day_of_year: float) -> tuple[float, float]:
    """Low-precision solar right ascension and declination [rad]."""
    eps = np.radians(23.44)
    g = np.radians(357.528 + 0.9856003 * day_of_year)
    lam = np.radians(280.46 + 0.9856474 * day_of_year) + np.radians(1.915) * np.sin(g)
    ra = np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam))
    dec = np.arcsin(np.sin(eps) * np.sin(lam))
    return ra, dec


def sso_beta_deg(altitude_m: float, ltdn_hours: float, day_of_year: float) -> float:
    """Beta angle of a sun-synchronous orbit with a given local time of descending node."""
    inc = np.radians(sso_inclination_deg(altitude_m))
    ra, dec = _sun_ra_dec(day_of_year)
    ltan = (ltdn_hours + 12.0) % 24.0
    raan = ra + np.radians((ltan - 12.0) * 15.0)
    s = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    n = np.array([np.sin(inc) * np.sin(raan), -np.sin(inc) * np.cos(raan), np.cos(inc)])
    return float(np.degrees(np.arcsin(n @ s)))


def sso_beta_range(altitude_m: float, ltdn_hours: float) -> tuple[float, float]:
    """Min and max beta over one year [deg]."""
    betas = [sso_beta_deg(altitude_m, ltdn_hours, d) for d in range(365)]
    return float(min(betas)), float(max(betas))
