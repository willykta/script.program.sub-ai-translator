from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from itertools import chain

from .srt import parse_srt, group_blocks, write_srt
from .prompt import build_prompt, extract_translations

def translate_batch(batch, lang, model, api_key, call_fn):
    indexed_texts = [(i, "\n".join(b["lines"])) for i, b in batch]
    prompt = build_prompt(indexed_texts, lang)
    response = call_fn(prompt, model, api_key)
    translations = extract_translations(response)  # musi zwracać dict: i -> text

    missing = [i for i, _ in batch if i not in translations]
    if missing:
        print(f"[WARN] Missing translations for indices: {missing}")
        print("=== Prompt ===\n" + prompt)
        print("=== Response ===\n" + response)

    return [(i, translations[i]) for i, _ in batch if i in translations]


from itertools import chain

def execute_batch_group(group, lang, model, api_key, call_fn):
    with ThreadPoolExecutor(max_workers=len(group)) as executor:
        futures = [
            executor.submit(
                translate_batch,
                batch,  # już zawiera (i, block)
                lang, model, api_key, call_fn
            )
            for batch in group
        ]
        return list(chain.from_iterable(f.result() for f in futures))


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
    translated_map = dict(translated_pairs)
    return [
        {**block, "lines": translated_map[i].split("\n")}
        for i, block in enumerate(blocks)
        if i in translated_map
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
