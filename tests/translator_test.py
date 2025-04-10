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

    result_path = translate_subtitles(str(input_path), "fake-key", "PL", "gpt-mock", mock_openai)

    assert Path(result_path) == output_path
    assert output_path.exists()

    expected_lines = expected_path.read_text(encoding="utf-8").splitlines()
    result_lines = output_path.read_text(encoding="utf-8").splitlines()

    assert expected_lines == result_lines, "Translated output doesn't match expected result"
    
def test_translate_with_progress_reporting(sample_paths):
    input_path = sample_paths["input"]
    output_path = sample_paths["output"]
    progress = []

    def report_progress(idx, total):
        progress.append((idx, total))

    result_path = translate_subtitles(
        str(input_path),
        api_key="fake",
        lang="PL",
        model="mock",
        call_fn=mock_openai,
        report_progress=report_progress
    )

    assert Path(result_path) == output_path
    assert output_path.exists()
    assert len(progress) >= 1
    assert progress[-1][0] == progress[-1][1]


def test_translate_cancel_after_first_batch(sample_paths):
    input_path = sample_paths["input"]
    cancel_after = 1
    call_count = {"value": 0}

    def cancel_on_second():
        return call_count["value"] >= cancel_after

    def call_with_counter(prompt, model, key):
        call_count["value"] += 1
        return fake_call_openai(prompt, model, key)

    with pytest.raises(Exception, match="anulowane przez użytkownika"):
        translate_subtitles(
            str(input_path),
            api_key="fake",
            lang="PL",
            model="mock",
            call_fn=call_with_counter,
            check_cancelled=cancel_on_second
        )
