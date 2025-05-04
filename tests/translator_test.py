import pytest
import re
from pathlib import Path
from core.translation import translate_subtitles
from api.mock import call as mock_openai
import textwrap

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / "data"

INPUT_FILE = DATA_DIR / "sample.srt"
EXPECTED_FILE = DATA_DIR / "expected_output.srt"
OUTPUT_FILE = DATA_DIR / "sample.pl.translated.srt"

def fake_call_openai(prompt, model, api_key):
    # We extract only the part after 'Tekst:\n'
    try:
        prompt_body = prompt.split("Tekst:\n", 1)[1]
    except IndexError:
        return ""  # fallback

    blocks = re.split(r"\n(\d+):\n", "\n" + prompt_body.strip())
    result = []
    for i in range(1, len(blocks) - 1, 2):  # i=index, i+1=text
        index = blocks[i]
        text = blocks[i + 1].strip()
        result.append(f"{index}:\n{text}")
    return "\n".join(result)



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
    result_path = translate_subtitles(str(input_path), "fake-key", "PL", "gpt-mock", fake_call_openai)

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
    with pytest.raises(Exception, match="Translation interrupted by client"):
        translate_subtitles(
            str(input_path),
            api_key="fake",
            lang="PL",
            model="mock",
            call_fn=call_with_counter,
            check_cancelled=cancel_on_second
        )
        
def test_translate_skips_last_line_correctly(tmp_path):
    # Given: a custom input .srt with 3 linie
    srt = (
        "13\n00:00:01,000 --> 00:00:02,000\nHello\n\n"
        "14\n00:00:03,000 --> 00:00:04,000\nWorld\n\n"
        "15\n00:00:05,000 --> 00:00:06,000\nAgain\n"
    )
    input_file = tmp_path / "three_lines.srt"
    input_file.write_text(srt, encoding="utf-8")

    # And: fake model that skips the last line
    def call_fn(prompt, model, key):
        blocks = re.split(r"\n(\d+):\n", "\n" + prompt.strip())
        result = []
        for i in range(1, len(blocks) - 3, 2):  # omit last
            index = blocks[i]
            text = blocks[i + 1].strip()
            result.append(f"{index}:\n{text}")
        return "\n".join(result)

    # When: we run translation
    result_path = translate_subtitles(
        str(input_file),
        api_key="fake",
        lang="PL",
        model="mock",
        call_fn=call_fn,
    )

    # Then: result file contains only two translated blocks
    translated = Path(result_path).read_text(encoding="utf-8").strip()
    expected = textwrap.dedent("""\
        13
        00:00:01,000 --> 00:00:02,000
        Hello

        14
        00:00:03,000 --> 00:00:04,000
        World""").strip()

    assert translated == expected


def test_translate_skips_middle_line_correctly(tmp_path):
    # Given: a custom input .srt with 3 linie
    srt = (
        "7\n00:00:01,000 --> 00:00:02,000\nHello\n\n"
        "8\n00:00:03,000 --> 00:00:04,000\nWorld\n\n"
        "9\n00:00:05,000 --> 00:00:06,000\nAgain\n"
    )
    input_file = tmp_path / "three_lines.srt"
    input_file.write_text(srt, encoding="utf-8")

    # And: fake model that skips the middle line
    def call_fn(prompt, model, key):
        try:
            prompt_body = prompt.split("Tekst:\n", 1)[1]
        except IndexError:
            return ""

        blocks = re.split(r"\n(\d+):\n", "\n" + prompt_body.strip())
        result = []
        for i in range(1, len(blocks) - 1, 2):
            index = int(blocks[i])
            if index == 1:  
                continue  
            text = blocks[i + 1].strip()
            result.append(f"{index}:\n{text}")
        return "\n".join(result)


    # When: we run translation
    result_path = translate_subtitles(
        str(input_file),
        api_key="fake",
        lang="PL",
        model="mock",
        call_fn=call_fn,
    )

    # Then: result file should include 1 and 3, but not 2
    translated = Path(result_path).read_text(encoding="utf-8").strip()

    expected =textwrap.dedent("""\
    7
    00:00:01,000 --> 00:00:02,000
    Hello

    9
    00:00:05,000 --> 00:00:06,000
    Again""").strip()

    assert translated == expected

