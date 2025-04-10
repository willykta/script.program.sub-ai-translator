import re
from concurrent.futures import ThreadPoolExecutor, as_completed

SRT_REGEX = r"(\d+)\s+([\d:,]+)\s+-->\s+([\d:,]+)\s+([\s\S]+?)(?=\n\n|\Z)"

parse_srt = lambda path: [
    {
        "index": int(m[0]),
        "start": m[1].strip(),
        "end": m[2].strip(),
        "lines": m[3].strip().splitlines()
    }
    for m in re.findall(SRT_REGEX, open(path, encoding="utf-8-sig").read())
]

group = lambda n, lst: [lst[i:i+10] for i in range(0, len(lst), 10)]

build_prompt = lambda texts, lang: (
    f"Tłumacz na język {lang}. Zachowaj znaczenie, styl i układ linii. "
    f"Zignoruj dźwięki w nawiasach typu [music], [laughs].\n\n" +
    "\n".join(f"{i+1}:\n{t}" for i, t in enumerate(texts))
)

extract_translations = lambda raw: [
    part.strip() for part in re.split(r"\n\d+:", f"\n{raw}")[1:]
]

translate_batch = lambda batch, lang, model, key, call_fn: extract_translations(
    call_fn(build_prompt(["\n".join(b["lines"]) for b in batch], lang), model, key)
)

write_srt = lambda blocks, path: open(path, "w", encoding="utf-8").write(
    "\n".join(
        f"{b['index']}\n{b['start']} --> {b['end']}\n" + "\n".join(b['lines']) + "\n"
        for b in blocks
    )
)

estimate_cost = lambda path, lang: (
    lambda prompts, chars, tokens: {
        "chars": chars,
        "tokens": tokens,
        "usd": round(tokens / 1000 * 0.001, 4),  # $0.001 per 1K tokens
        "prompts": prompts
    }
)(
    prompts := [
        build_prompt(["\n".join(b["lines"]) for b in batch], lang)
        for batch in group(10, parse_srt(path))
    ],
    sum(len(p) for p in prompts),
    sum(len(p) for p in prompts) // 4
)

from itertools import chain

def translate_subtitles(path, api_key, lang, model, call_fn, report_progress=None, check_cancelled=None):
    blocks = parse_srt(path)
    batches = group(10, list(enumerate(blocks)))

    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {
            ex.submit(translate_batch, [b for _, b in batch], lang, model, api_key, call_fn): batch
            for batch in batches
        }

        total = len(futures)

        def completed():
            for idx, future in enumerate(as_completed(futures), 1):
                if check_cancelled and check_cancelled():
                    raise Exception("Tłumaczenie anulowane przez użytkownika.")
                if report_progress:
                    report_progress(idx, total)
                yield zip((i for i, _ in futures[future]), future.result())

        translated = sorted(
            chain.from_iterable(completed()),
            key=lambda x: x[0]
        )

    merged = [
        {**blocks[i], "lines": text.split("\n")}
        for i, text in translated
    ]

    new_path = path.replace(".srt", f".{lang.lower()}.translated.srt")
    write_srt(merged, new_path)
    return new_path



