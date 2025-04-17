import pytest
import re
from core.srt import parse_srt, group_blocks, write_srt
from core.prompt import build_prompt, extract_translations
from pathlib import Path

def test_parse_srt_with_valid_blocks(tmp_path):
    # Given: an .srt file with three valid subtitle blocks
    srt_content = (
        "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nWorld\n\n"
        "3\n00:00:05,000 --> 00:00:06,000\nAgain\n"
    )
    file_path = tmp_path / "test.srt"
    file_path.write_text(srt_content, encoding="utf-8")

    # When: parsing the file
    result = parse_srt(str(file_path))

    # Then: result should contain 3 blocks with correct structure
    assert len(result) == 3
    assert result[0]["index"] == 1
    assert result[1]["lines"] == ["World"]

def test_grouping_function():
    # Given: a list of 10 items
    items = list(range(10))

    # When: grouped by 3
    result = group_blocks(items, 3)

    # Then: result should be grouped into 4 groups (3 + 3 + 3 + 1)
    assert result == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

def test_build_prompt_format():
    # Given: a list of text blocks
    texts = ["Hello world", "Another block"]
    lang = "PL"

    # When: building prompt
    prompt = build_prompt(texts, lang)

    # Then: the prompt should contain numbering and instructions
    assert "1:\nHello world" in prompt
    assert "2:\nAnother block" in prompt
    assert "T\u0142umacz na j\u0119zyk PL" in prompt

def test_extract_translations_parses_cleanly():
    # Given: a raw model output string
    raw = "1:\nCze\u015b\u0107\n2:\n\u015awiat"

    # When: extracting translations
    result = extract_translations(raw)

    # Then: both blocks should be cleanly split
    assert result == ["Cze\u015b\u0107", "\u015awiat"]

def test_write_srt_creates_valid_file(tmp_path):
    # Given: subtitle blocks
    blocks = [
        {"index": 1, "start": "00:00:01,000", "end": "00:00:02,000", "lines": ["Hi"]},
        {"index": 2, "start": "00:00:03,000", "end": "00:00:04,000", "lines": ["There"]},
    ]
    out_file = tmp_path / "test.srt"

    # When: writing the file
    write_srt(blocks, str(out_file))

    # Then: the content should match the expected .srt structure
    content = out_file.read_text(encoding="utf-8").strip()
    expected = """1
00:00:01,000 --> 00:00:02,000
Hi

2
00:00:03,000 --> 00:00:04,000
There"""
    assert content == expected
