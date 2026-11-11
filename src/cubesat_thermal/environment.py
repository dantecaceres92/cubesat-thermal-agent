"""Orbital heat loads (direct solar, albedo, Earth IR) on the six faces of a nadir-pointing box.

Frame: e1 = Sun direction projected on the orbit plane, e2 = along-track at theta = 90 deg,
n = orbit normal. theta = 0 is orbit noon (closest to the Sun).
Body axes (LVLH): +Z zenith (radial out), -Z nadir, +X velocity, +Y orbit normal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .orbit import R_EARTH, earth_view_factors

FACES = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
FACE_KIND = {"+X": "side", "-X": "side", "+Y": "side", "-Y": "side", "+Z": "zenith", "-Z": "nadir"}


@dataclass(frozen=True)
class Environment:
    altitude_m: float
    beta_deg: float
    solar: float      # W/m2
    albedo: float     # -
    olr: float        # W/m2

    def sun_vector(self) -> np.ndarray:
        b = np.radians(self.beta_deg)
        return np.array([np.cos(b), 0.0, np.sin(b)])

    def in_eclipse(self, theta: float) -> bool:
        s = self.sun_vector()
        r = np.array([np.cos(theta), np.sin(theta), 0.0])
        rs = r @ s
        perp = np.linalg.norm(r - rs * s) * (R_EARTH + self.altitude_m)
        return bool(rs < 0.0 and perp < R_EARTH)

    def incident(self, theta: float) -> dict[str, tuple[float, float, float]]:
        """Incident flux per face [W/m2] as (direct solar, albedo, Earth IR)."""
        s = self.sun_vector()
        r = np.array([np.cos(theta), np.sin(theta), 0.0])
        v = np.array([-np.sin(theta), np.cos(theta), 0.0])
        n = np.array([0.0, 0.0, 1.0])
        normals = {"+Z": r, "-Z": -r, "+X": v, "-X": -v, "+Y": n, "-Y": -n}
        F = earth_view_factors(self.altitude_m)
        lit = 0.0 if self.in_eclipse(theta) else 1.0
        cos_zeta = max(0.0, float(r @ s))   # solar zenith cosine at the sub-satellite point
        out = {}
        for face in FACES:
            f = F[FACE_KIND[face]]
            q_sun = self.solar * max(0.0, float(normals[face] @ s)) * lit
            q_alb = self.solar * self.albedo * f * cos_zeta
            q_ir = self.olr * f
            out[face] = (q_sun, q_alb, q_ir)
        return out
