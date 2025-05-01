from .config import MODELS, LANGUAGES, DEFAULT_PARALLEL_REQUESTS, DEFAULT_PRICE_PER_1000_TOKENS
import xbmcaddon
from xbmcaddon import Addon
from api import mock, openai, gemini
from backoff import rate_limited_backoff_on_429

addon = Addon("script.program.sub-ai-translator")

PROVIDERS = {
    "OpenAI": {
        "get_config": lambda: {
            "provider": "OpenAI",
            "lang": get_effective_lang(),
            "api_key": addon.getSetting("api_key"),
            "model": get_enum("model", MODELS),
            "price_per_1000_tokens": float(addon.getSetting("price_per_1000_tokens") or DEFAULT_PRICE_PER_1000_TOKENS),
            "use_mock": addon.getSettingBool("use_mock"),
            "parallel": max(1, int(addon.getSetting("parallel_requests") or DEFAULT_PARALLEL_REQUESTS))
        },
        "call_fn": openai
    },
    "Gemini": {
        "get_config": lambda: {
            "provider": "Gemini",
            "lang": get_effective_lang(),
            "api_key": addon.getSetting("gemini_api_key"),
            "model": get_enum("gemini_model", ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest"]),
            "price_per_1000_tokens": 0.0,
            "use_mock": addon.getSettingBool("use_mock"),
            "parallel": 1
        },
        "call_fn": gemini
    },
    "Mock (Test)": {
        "get_config": lambda: {
            "provider": "Mock (Test)",
            "lang": get_effective_lang(),
            "api_key": "",
            "model": "mock-model",
            "price_per_1000_tokens": 0.0,
            "use_mock": True,
            "parallel": max(1, int(addon.getSetting("parallel_requests") or DEFAULT_PARALLEL_REQUESTS))
        },
        "call_fn": mock
    }
}

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
    provider_options = list(PROVIDERS.keys())
    provider = get_enum("provider", provider_options)
    return PROVIDERS.get(provider, PROVIDERS["Mock (Test)"])["get_config"]()

def get_call_fn():
    provider_options = list(PROVIDERS.keys())
    provider = get_enum("provider", provider_options)
    return PROVIDERS.get(provider, PROVIDERS["Mock (Test)"])["call_fn"]
