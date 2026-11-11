# CLAUDE.md: context for Claude Code

## What this project is
This is an orbital thermal analysis engine for a 3U CubeSat. Phase 2 adds an MCP server so an AI agent can run it in plain language.

The owner is Dante Cáceres, a satellite thermal engineer (4 years: NX/Simcenter Space Systems Thermal, correlation, worst cases) and CFD engineer. He is building this as a portfolio project for recruiters in the space industry. He speaks Spanish, so explain in Spanish. Code, comments, README and docs are in English.

## Commands
- `uv sync`: create or update the environment
- `uv run pytest`: validation suite V1–V6. It must stay green after every change.
- `uv run python examples/run_cases.py`: regenerates `results/` (plots and summary.json)

## Architecture rules
- **Physics lives only in `src/cubesat_thermal/`.** The MCP server is a thin layer: it calls the engine and formats results. It never computes temperatures itself.
- **`model.yaml` is the single source of truth** for nodes, coatings, conductors, cases, limits and heaters. Every value carries its source in a comment.
- **Never modify the base model in place.** Use `ThermalModel.copy_with(...)`, which returns a modified copy.
- **Every result shows its assumptions and the energy-balance closure.** This is the owner's engineering signature: correlate on the energy balance, not only on temperatures.
- Units are SI internally (K, W, J, m). Temperatures are reported in °C.

## Phase 2 – MCP server (next step)
Create `src/cubesat_thermal/server.py` with MCPServer (`mcp>=2` package, `from mcp.server.mcpserver import MCPServer`; MCP Python SDK v2 renamed FastMCP to MCPServer) and expose these tools:

| Tool | Engine call |
|---|---|
| `describe_model()` | nodes, conductors, coatings, cases, limits, sources |
| `orbit_info(altitude_km, beta_deg)` | `period_s`, `eclipse_fraction`, `critical_beta_deg`, `earth_view_factors`, `sso_beta_range` |
| `run_case(case, beta_deg=None, overrides=None)` | `ThermalModel.run` → min/max per node, orbits to converge |
| `energy_balance(case)` | `Result.energy` per face (solar, albedo, IR, emitted) and closure % |
| `check_margins(case)` | `analysis.check_margins` → PASS / WATCH / FAIL |
| `beta_sweep(case, start, stop, step)` | `analysis.beta_sweep` + plot |
| `size_heater(node, case, battery_mount_g=None)` | `analysis.size_heater` |
| `modify_design(name, conductor/coating/dissipation)` | `copy_with`, stored as a named variant |
| `plot(case, nodes, variant=None)` | `plotting.plot_temperatures` → PNG path |
| `generate_report(cases)` | Markdown report with assumptions, results, margins, sources |

Also expose `model.yaml` as a read-only MCP resource and add a prompt `worst_case_analysis`.
Register the server in Claude Code with `claude mcp add cubesat-thermal -- uv run --directory <repo> python -m cubesat_thermal.server`.

## Known limits (v0.1)
- No internal radiation (owner's decision, 2026-11-10)
- Isothermal nodes
- Albedo uses a cos ζ approximation with no terminator correction
- Constant dissipation per case
- Preliminary-design accuracy, about ±5–10 °C
