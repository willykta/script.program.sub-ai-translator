import pytest
import re
from pathlib import Path
from unittest.mock import patch
from translator.translator import translate_subtitles
import sys
import os

addon_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(addon_dir, "api"))

from api.mock_openai import call as mock_openai

def fake_call_openai(prompt, model, api_key):
    blocks = re.split(r"\n\d+:\n", prompt)[1:] 
    return "\n".join(f"{i+1}:\n{block.strip()}" for i, block in enumerate(blocks))

@pytest.fixture
def sample_paths():
    base = Path(__file__).parent / "data"
    return {
        "input": base / "sample.srt",
        "expected": base / "expected_output.srt",
        "output": base / "sample.pl.translated.srt"
    }

def test_translate_srt_to_expected_output(sample_paths):
    input_path = sample_paths["input"]
    expected_path = sample_paths["expected"]
    output_path = sample_paths["output"]

    assert input_path.exists(), "Missing sample.srt"
    assert expected_path.exists(), "Missing expected_output.srt"

    # with patch("translator.translator.call_openai", side_effect=fake_call_openai):
    result_path = translate_subtitles(str(input_path), "fake-key", "PL", "gpt-mock", mock_openai)

    assert Path(result_path) == output_path
    assert output_path.exists()

    expected_lines = expected_path.read_text(encoding="utf-8").splitlines()
    result_lines = output_path.read_text(encoding="utf-8").splitlines()

    assert expected_lines == result_lines, "Translated output doesn't match expected result"
    