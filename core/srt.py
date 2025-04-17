import re

SRT_REGEX = r"(\d+)\s+([\d:,]+)\s+-->\s+([\d:,]+)\s+([\s\S]+?)(?=\n\n|\Z)"

def parse_srt(path):
    with open(path, encoding="utf-8-sig") as f:
        content = f.read()
    return [
        {
            "index": int(m[0]),
            "start": m[1].strip(),
            "end": m[2].strip(),
            "lines": m[3].strip().splitlines()
        }
        for m in re.findall(SRT_REGEX, content)
    ]

def write_srt(blocks, path):
    lines = [
        f"{block['index']}\n{block['start']} --> {block['end']}\n" + "\n".join(block["lines"])
        for block in blocks
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(lines))

def group_blocks(blocks, size):
    return [blocks[i:i + size] for i in range(0, len(blocks), size)]
