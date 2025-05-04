from .prompt import build_prompt
from .srt import parse_srt, group_blocks

from .prompt import build_prompt
from .srt import parse_srt, group_blocks

def estimate_cost(path, lang, price_per_1000=0):
    blocks = parse_srt(path)
    indexed = list(enumerate(blocks))
    batches = group_blocks(indexed, 15)

    prompts = [
        build_prompt([(i, "\n".join(b["lines"])) for i, b in batch], lang)
        for batch in batches
    ]
    chars = sum(len(p) for p in prompts)
    tokens = chars // 4
    usd = round(2 * tokens / 1000 * price_per_1000, 4)

    return {
        "chars": chars,
        "tokens": tokens,
        "usd": usd,
        "prompts": prompts
    }

