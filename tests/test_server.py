"""MCP server smoke tests: tool surface, ToolError on invalid inputs, thin-layer consistency.
Skipped when the optional `mcp` extra is not installed (uv sync --extra mcp)."""
import pytest

pytest.importorskip("mcp.server.mcpserver")

from mcp import Client  # noqa: E402

from cubesat_thermal import server  # noqa: E402

TOOLS = {"describe_model", "orbit_info", "run_case", "energy_balance", "check_margins",
         "beta_sweep", "size_heater", "modify_design", "plot", "generate_report"}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _tmp_out(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "OUT_DIR", tmp_path)


def _text(result) -> str:
    return " ".join(getattr(c, "text", "") for c in result.content)


@pytest.mark.anyio
async def test_surface():
    async with Client(server.mcp) as client:
        tools = {t.name for t in (await client.list_tools()).tools}
        assert TOOLS <= tools
        resources = (await client.list_resources()).resources
        assert any(str(r.uri) == "model://cubesat-thermal/model.yaml" for r in resources)
        prompts = {p.name for p in (await client.list_prompts()).prompts}
        assert "worst_case_analysis" in prompts


@pytest.mark.anyio
@pytest.mark.parametrize("tool, args, message", [
    ("run_case", {"case": "nominal"}, "Unknown case"),
    ("run_case", {"case": "hot", "beta_deg": 120}, "beta_deg"),
    ("run_case", {"case": "hot", "overrides": {"albedo": 1.5}}, "albedo"),
    ("run_case", {"case": "hot", "overrides": {"dissipation": 3}}, "Unknown override"),
    ("run_case", {"case": "hot", "variant": "nope"}, "Unknown variant"),
    ("orbit_info", {"altitude_km": 50, "beta_deg": 0}, "altitude_km"),
    ("beta_sweep", {"case": "hot", "start": 0, "stop": 90, "step": 0.5}, "limit"),
    ("beta_sweep", {"case": "hot", "step": -1}, "step"),
    ("size_heater", {"node": "Payload", "case": "cold"}, "heater is mounted"),
    ("plot", {"case": "hot", "nodes": ["Radiator"]}, "Unknown node"),
    ("modify_design", {"name": "v1"}, "at least one change"),
    ("modify_design", {"name": "v1", "coating": {"coating": "anodized_al", "prop": "eta",
                                                 "value": 0.1}}, "has no 'eta'"),
    ("generate_report", {"cases": []}, "at least one case"),
])
async def test_invalid_inputs_are_tool_errors(tool, args, message):
    async with Client(server.mcp) as client:
        result = await client.call_tool(tool, args)
        assert result.is_error
        assert message in _text(result)


@pytest.mark.anyio
async def test_run_case_matches_engine():
    async with Client(server.mcp) as client:
        result = await client.call_tool("run_case", {"case": "hot"})
        assert not result.is_error
        data = result.structured_content
    r = server._base.run("hot")
    assert data["temperatures_c"]["Battery"]["max"] == pytest.approx(r.tmax("Battery"), abs=0.01)
    assert data["energy_balance"]["closure_pct"] < 0.1
    assert "known_limits" in data["assumptions"]


@pytest.mark.anyio
async def test_variant_does_not_touch_baseline():
    async with Client(server.mcp) as client:
        result = await client.call_tool("modify_design", {
            "name": "isolated", "conductor": {"node_a": "Battery", "node_b": "Frame",
                                              "g_w_per_k": 0.05}})
        assert not result.is_error
        desc = (await client.call_tool("describe_model", {})).structured_content
    base_g = [c["G"] for c in desc["conductors_W_per_K"] if {c["a"], c["b"]} == {"Battery", "Frame"}]
    assert base_g == [0.25]
