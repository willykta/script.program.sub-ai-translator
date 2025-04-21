from .config import MODELS, LANGUAGES, DEFAULT_PARALLEL_REQUESTS, DEFAULT_PRICE_PER_1000_TOKENS
import xbmcaddon
from xbmcaddon import Addon

addon = Addon("script.program.sub-ai-translator")

def get_enum(setting_id, options):
    try:
        idx = int(addon.getSetting(setting_id))
        return options[idx] if 0 <= idx < len(options) else ""
    except Exception:
        return ""

def get_effective_lang():
    lang = get_enum("target_lang", LANGUAGES)
    return addon.getSetting("custom_lang") if lang == "Other" else lang


def get():
    return {
        "lang": get_effective_lang(),
        "api_key": addon.getSetting("api_key"),
        "model": get_enum("model", MODELS),
        "use_mock": addon.getSettingBool("use_mock"),
        "parallel": max(1, int(addon.getSetting("parallel") or DEFAULT_PARALLEL_REQUESTS)),
        "price_per_1000_tokens": float(addon.getSetting("price_per_1000_tokens") or DEFAULT_PRICE_PER_1000_TOKENS)
    }
