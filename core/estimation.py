from .prompt import build_prompt
from .srt import parse_srt, group_blocks

def estimate_cost(path, lang):
    blocks = parse_srt(path)
    batches = group_blocks(blocks, 10)

    prompts = [
        build_prompt(["\n".join(b["lines"]) for b in batch], lang)
        for batch in batches
    ]
    chars = sum(len(p) for p in prompts)
    tokens = chars // 4
    usd = round(tokens / 1000 * 0.001, 4)

    return {
        "chars": chars,
        "tokens": tokens,
        "usd": usd,
        "prompts": prompts
    }
