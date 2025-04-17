from core.estimation import estimate_cost
from pathlib import Path

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / "data"
SAMPLE_FILE = DATA_DIR / "sample.srt"

def test_estimate_cost_output():
    # Given: a sample subtitle file

    # When: estimating cost
    result = estimate_cost(str(SAMPLE_FILE), "PL")

    # Then: result should be a valid estimation dictionary
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