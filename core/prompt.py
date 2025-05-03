
import re

def build_prompt(texts, lang):
    header = (
        f"Przetłumacz na język {lang}. Zasady:\n"
        "- Zachowaj numerację (1:, 2:, ...)\n"
        "- Nie zmieniaj liczby linii\n"
        "- Zachowaj układ wierszy w każdej linijce\n"
        "Przykład:\n"
        "1:\nHello!\n2:\nHow are you?\n\n"
        "Tekst:\n"
    )
    numbered = "\n".join(f"{i + 1}:\n{t}" for i, t in enumerate(texts))
    return header + numbered


def extract_translations(raw_response):
    return [part.strip() for part in re.split(r"\n\d+:", f"\n{raw_response}")[1:]]
