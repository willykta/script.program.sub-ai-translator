
import re

def build_prompt(indexed_texts, lang):
    header = (
        f"Przetłumacz na język {lang}. Zasady:\n"
        "- Zachowaj numerację (np. 12:, 43:, ...)\n"
        "- Nie zmieniaj liczby linii\n"
        "- Zachowaj układ wierszy\n"
        "Przykład:\n"
        "1:\nHello!\n42:\nHow are you?\n\n"
        "Tekst:\n"
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