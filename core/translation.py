from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from itertools import chain

from .srt import parse_srt, group_blocks, write_srt
from .prompt import build_prompt, extract_translations

def translate_batch(batch, lang, model, api_key, call_fn):
    texts = ["\n".join(b["lines"]) for b in batch]
    prompt = build_prompt(texts, lang)
    response = call_fn(prompt, model, api_key)
    translations = extract_translations(response)
    if len(translations) != len(batch):
        log_translation_mismatch(batch, translations, prompt, response)
    return translations


def log_translation_mismatch(batch, translations, prompt, response):
    print(f"[WARN] Mismatch: sent {len(batch)}, got {len(translations)}")
    print("=== Prompt ===")
    print(prompt)
    print("=== Response ===")
    print(response)
    print("=== Batch index map ===")
    for idx, block in enumerate(batch):
        print(f"[{idx}] {block['lines']}")


def execute_batch_group(group, lang, model, api_key, call_fn):
    with ThreadPoolExecutor(max_workers=len(group)) as executor:
        futures = [
            executor.submit(
                translate_batch,
                [b for _, b in batch],
                lang, model, api_key, call_fn
            )
            for batch in group
        ]
        return [
            (i, text)
            for batch, future in zip(group, futures)
            for (i, _), text in zip(batch, future.result())
        ]


def translate_in_batches(batches, lang, model, api_key, call_fn, parallel, report_progress=None, check_cancelled=None):
    results = []
    batch_iter = iter(batches)
    total = len(batches)
    done = 0

    def next_group():
        return list(islice(batch_iter, parallel))

    group = next_group()
    while group:
        if check_cancelled and check_cancelled():
            raise Exception("Translation interrupted by client")

        group_results = execute_batch_group(group, lang, model, api_key, call_fn)
        results.extend(group_results)

        done += len(group)
        if report_progress:
            report_progress(done, total)

        group = next_group()

    return results

def merge_translations(blocks, translated_pairs):
    return [
        {**blocks[i], "lines": text.split("\n")}
        for i, text in sorted(translated_pairs, key=lambda x: x[0])
    ]

def translate_subtitles(
    path,
    api_key,
    lang,
    model,
    call_fn,
    report_progress=None,
    check_cancelled=None,
    parallel=3
):
    blocks = parse_srt(path)
    batches = group_blocks(list(enumerate(blocks)), 15)
    translated_pairs = translate_in_batches(
        batches, lang, model, api_key, call_fn, parallel,
        report_progress, check_cancelled
    )
    merged = merge_translations(blocks, translated_pairs)
    new_path = path.replace(".srt", f".{lang.lower()}.translated.srt")
    return write_srt(merged, new_path) or new_path
