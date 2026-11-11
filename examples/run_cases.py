"""Run the baseline hot and cold cases, check margins, size the battery heater and
compare against an isolated battery mount. Writes plots and a summary to results/.

    uv run python examples/run_cases.py
"""
import json
from pathlib import Path

from cubesat_thermal import ThermalModel, beta_sweep, check_margins, size_heater, sso_beta_range
from cubesat_thermal.plotting import plot_beta_sweep, plot_energy_balance, plot_temperatures

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

model = ThermalModel.from_yaml(ROOT / "model.yaml")
summary = {"model": model.name,
           "beta_range_deg": [round(b, 1) for b in sso_beta_range(model.altitude_m, 10.5)]}

for case in ("hot", "cold"):
    r = model.run(case)
    plot_temperatures(model, r, path=OUT / f"temperatures_{case}.png")
    plot_energy_balance(model, r, path=OUT / f"energy_balance_{case}.png")
    summary[case] = {"temperatures_c": r.summary(), "margins": check_margins(model, r),
                     "closure_pct": round(r.energy["closure_pct"], 4), "orbits": r.orbits}
    print(f"\n== {case} case: closure {r.energy['closure_pct']:.4f} %")
    for row in summary[case]["margins"]:
        print(f"  {row['node']:9s} {row['tmin_c']:7.1f} .. {row['tmax_c']:6.1f} degC  {row['status']}")

print("\n== Battery heater, baseline mount (G = 0.25 W/K)")
summary["heater_baseline"] = size_heater(model)
print(" ", summary["heater_baseline"])

isolated = model.copy_with(conductor=("Battery", "Frame", 0.05))
print("== Battery heater, isolated mount (G = 0.05 W/K)")
summary["heater_isolated"] = size_heater(isolated)
print(" ", summary["heater_isolated"])

fixed = isolated.copy_with(heater_power=summary["heater_isolated"]["installed_w"])
for case in ("hot", "cold"):
    r = fixed.run(case)
    plot_temperatures(fixed, r, path=OUT / f"temperatures_{case}_isolated_heater.png")
    summary[f"{case}_isolated_heater"] = {"margins": check_margins(fixed, r),
                                          "heater_duty": r.energy["heater_duty"]}

rows = beta_sweep(model, "hot", range(0, 91, 10))
plot_beta_sweep(rows, path=OUT / "beta_sweep_hot.png")
summary["beta_sweep_hot"] = rows

(OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
print(f"\nResults written to {OUT}")
