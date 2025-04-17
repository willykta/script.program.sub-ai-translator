
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import chain

from .srt import parse_srt, group_blocks, write_srt
from .prompt import build_prompt, extract_translations

def translate_batch(batch, lang, model, key, call_fn):
    texts = ["\n".join(b["lines"]) for b in batch]
    prompt = build_prompt(texts, lang)
    response = call_fn(prompt, model, key)
    return extract_translations(response)

def create_futures(blocks, lang, model, key, call_fn, executor):
    batches = group_blocks(list(enumerate(blocks)), 10)
    return {
        executor.submit(
            translate_batch,
            [b for _, b in batch],
            lang, model, key, call_fn
        ): batch
        for batch in batches
    }

def completed_results(futures, total, report_progress=None, check_cancelled=None):
    for idx, future in enumerate(as_completed(futures), 1):
        if check_cancelled and check_cancelled():
            raise Exception("Tłumaczenie anulowane przez użytkownika.")
        if report_progress:
            report_progress(idx, total)
        for (i, _), text in zip(futures[future], future.result()):
            yield i, text

def merge_translations(blocks, translated_pairs):
    return [
        {**blocks[i], "lines": text.split("\n")}
        for i, text in sorted(translated_pairs, key=lambda x: x[0])
    ]

def translate_subtitles(path, api_key, lang, model, call_fn, report_progress=None, check_cancelled=None):
    blocks = parse_srt(path)
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = create_futures(blocks, lang, model, api_key, call_fn, ex)
        results = completed_results(futures, len(futures), report_progress, check_cancelled)
    merged = merge_translations(blocks, results)
    new_path = path.replace(".srt", f".{lang.lower()}.translated.srt")
    return write_srt(merged, new_path) or new_path
