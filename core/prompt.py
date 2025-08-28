
import re

def build_prompt(indexed_texts, lang):
    header = (
        f"TRANSLATE to {lang} with these rules:\n"
        "- KEEP original numbering (e.g., 12:, 43:, ...)\n"
        "- MAINTAIN exact line count\n"
        "- PRESERVE line structure\n"
        "EXAMPLE:\n"
        "1:\nHello!\n42:\nHow are you?\n\n"
        "TRANSLATION REQUEST:\n"
    )
    numbered = "\n".join(f"{i}:\n{t}" for i, t in indexed_texts)
    return header + numbered


def extract_translations(response):
    blocks = re.split(r"(?<=\n)\d+:\n", "\n" + response.strip())
    keys = re.findall(r"(?<=\n)(\d+):\n", "\n" + response.strip())
    return {
        int(i): block.strip()
        for i, block in zip(keys, blocks[1:])  # blocks[0] is always empty
        if block.strip()
    }