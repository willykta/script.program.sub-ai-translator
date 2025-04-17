
import re

def build_prompt(texts, lang):
    header = (
        f"Tłumacz na język {lang}. Zachowaj znaczenie, styl i układ linii. "
        f"Zignoruj dźwięki w nawiasach typu [music], [laughs].\n\n"
    )
    numbered = "\n".join(f"{i + 1}:\n{t}" for i, t in enumerate(texts))
    return header + numbered

def extract_translations(raw_response):
    return [part.strip() for part in re.split(r"\n\d+:", f"\n{raw_response}")[1:]]
