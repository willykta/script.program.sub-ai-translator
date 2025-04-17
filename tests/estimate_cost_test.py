from core.translation import estimate_cost
from pathlib import Path

def test_estimate_cost_output():
    path = Path(__file__).parent / "data" / "sample.srt"
    result = estimate_cost(str(path), "PL")

    assert isinstance(result, dict)
    assert "chars" in result
    assert "tokens" in result
    assert "usd" in result
    assert "prompts" in result

    assert result["chars"] > 0
    assert result["tokens"] > 0
    assert 0 <= result["usd"] < 1.0
    assert isinstance(result["prompts"], list)
    assert all(isinstance(p, str) for p in result["prompts"])
