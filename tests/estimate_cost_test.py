from core.estimation import estimate_cost
from pathlib import Path
import pytest

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

def test_estimate_cost_with_minimal_input(tmp_path):
    # Given: a minimal .srt file with one short line
    test_file = tmp_path / "minimal.srt"
    test_file.write_text("1\n00:00:01,000 --> 00:00:02,000\nHi\n", encoding="utf-8")

    # When: estimating cost
    result = estimate_cost(str(test_file), "PL")

    # Then: result should contain small values, possibly zero USD
    assert result["chars"] > 0
    assert result["tokens"] > 0
    assert result["usd"] >= 0

def test_estimate_cost_with_empty_file(tmp_path):
    # Given: an empty .srt file
    test_file = tmp_path / "empty.srt"
    test_file.write_text("", encoding="utf-8")

    # When: estimating cost
    result = estimate_cost(str(test_file), "PL")

    # Then: result should contain zero values
    assert result["chars"] == 0
    assert result["tokens"] == 0
    assert result["usd"] == 0
    assert result["prompts"] == []

def test_estimate_cost_with_uncommon_language():
    # Given: a valid sample subtitle file

    # When: estimating cost for an uncommon language
    result = estimate_cost(str(SAMPLE_FILE), "Esperanto")

    # Then: estimation should still work
    assert result["chars"] > 0
    assert result["tokens"] > 0
    assert isinstance(result["prompts"], list)

def test_estimate_cost_with_large_file(tmp_path):
    # Given: a large file simulated by repeating many subtitle blocks
    block = "1\n00:00:01,000 --> 00:00:02,000\nTest line\n\n"
    large_content = block * 200  # simulate 200 subtitles
    test_file = tmp_path / "large.srt"
    test_file.write_text(large_content, encoding="utf-8")

    # When: estimating cost
    result = estimate_cost(str(test_file), "PL")

    # Then: results should reflect the larger size
    assert result["chars"] > 0
    assert result["tokens"] > 0
    assert len(result["prompts"]) > 1
