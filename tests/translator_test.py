import pytest
import re
from pathlib import Path
from core.translation import translate_subtitles
from api.mock import call as mock_openai

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / "data"

INPUT_FILE = DATA_DIR / "sample.srt"
EXPECTED_FILE = DATA_DIR / "expected_output.srt"
OUTPUT_FILE = DATA_DIR / "sample.pl.translated.srt"

def fake_call_openai(prompt, model, api_key):
    blocks = re.split(r"\n\d+:\n", prompt)[1:]
    return "\n".join(f"{i+1}:\n{block.strip()}" for i, block in enumerate(blocks))

@pytest.fixture
def sample_paths():
    return {
        "input": INPUT_FILE,
        "expected": EXPECTED_FILE,
        "output": OUTPUT_FILE
    }

def test_translate_srt_to_expected_output(sample_paths):
    input_path = sample_paths["input"]
    expected_path = sample_paths["expected"]
    output_path = sample_paths["output"]

    # Given input and expected subtitle files exist
    assert input_path.exists()
    assert expected_path.exists()

    # When we translate the subtitles
    result_path = translate_subtitles(str(input_path), "fake-key", "PL", "gpt-mock", mock_openai)

    # Then output matches expected result
    assert Path(result_path) == output_path
    assert output_path.exists()
    assert expected_path.read_text(encoding="utf-8").splitlines() == output_path.read_text(encoding="utf-8").splitlines()

def test_translate_with_progress_reporting(sample_paths):
    input_path = sample_paths["input"]
    output_path = sample_paths["output"]
    progress = []

    # Given a progress reporting callback
    def report_progress(idx, total):
        progress.append((idx, total))

    # When we translate
    result_path = translate_subtitles(
        str(input_path),
        api_key="fake",
        lang="PL",
        model="mock",
        call_fn=mock_openai,
        report_progress=report_progress
    )

    # Then progress is reported and file is saved
    assert Path(result_path) == output_path
    assert output_path.exists()
    assert len(progress) >= 1
    assert progress[-1][0] == progress[-1][1]

def test_translate_cancel_after_first_batch(sample_paths):
    input_path = sample_paths["input"]
    cancel_after = 1
    call_count = {"value": 0}

    # Given a cancellation condition after the first batch
    def cancel_on_second():
        return call_count["value"] >= cancel_after

    def call_with_counter(prompt, model, key):
        call_count["value"] += 1
        return fake_call_openai(prompt, model, key)

    # When we cancel after first batch
    # Then we expect an exception
    with pytest.raises(Exception, match="anulowane przez użytkownika"):
        translate_subtitles(
            str(input_path),
            api_key="fake",
            lang="PL",
            model="mock",
            call_fn=call_with_counter,
            check_cancelled=cancel_on_second
        )
        