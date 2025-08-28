from .config import MODELS, LANGUAGES, DEFAULT_PARALLEL_REQUESTS, DEFAULT_PRICE_PER_1000_TOKENS
import xbmcaddon
from xbmcaddon import Addon
import xbmc
from api import mock, openai, gemini, openrouter
from .backoff import rate_limited_backoff_on_429

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
            "parallel": max(1, min(20, int(addon.getSetting("parallel_requests") or DEFAULT_PARALLEL_REQUESTS)))  # Increased max to 20
        },
        "call_fn": rate_limited_backoff_on_429(provider="OpenAI")(lambda prompt, model, api_key: openai(prompt, model, api_key))
    },
    "Gemini": {
        "get_config": lambda: {
            "provider": "Gemini",
            "lang": get_effective_lang(),
            "api_key": addon.getSetting("gemini_api_key"),
            "model": get_enum("gemini_model", ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest", "gemini-2.0-flash"]),
            "price_per_1000_tokens": 0.0,
            "use_mock": addon.getSettingBool("use_mock"),
            "parallel": max(1, min(5, int(addon.getSetting("parallel_requests") or DEFAULT_PARALLEL_REQUESTS)))  # Increased max to 5
        },
        "call_fn": rate_limited_backoff_on_429(provider="Gemini")(lambda prompt, model, api_key: gemini(prompt, model, api_key, logger=lambda msg: xbmc.log(msg, xbmc.LOGDEBUG)))
    },
    "OpenRouter": {
        "get_config": lambda: {
            "provider": "OpenRouter",
            "lang": get_effective_lang(),
            "api_key": addon.getSetting("openrouter_api_key"),
            "model": get_enum("openrouter_model", [
                "openai/gpt-4o-mini",
                "openai/gpt-4o",
                "openai/gpt-4-turbo",
                "anthropic/claude-3.5-sonnet",
                "google/gemini-pro",
                "meta-llama/llama-3.1-70b-instruct",
                "mistralai/mistral-large"
            ]),
            "price_per_1000_tokens": 0.0,
            "use_mock": addon.getSettingBool("use_mock"),
            "parallel": max(1, min(8, int(addon.getSetting("parallel_requests") or DEFAULT_PARALLEL_REQUESTS)))  # Increased max to 8
        },
        "call_fn": rate_limited_backoff_on_429(provider="OpenRouter")(lambda prompt, model, api_key: openrouter(prompt, model, api_key, logger=lambda msg: xbmc.log(msg, xbmc.LOGDEBUG)))
    },
    "Mock (Test)": {
        "get_config": lambda: {
            "provider": "Mock (Test)",
            "lang": get_effective_lang(),
            "api_key": "",
            "model": "mock-model",
            "price_per_1000_tokens": 0.0,
            "use_mock": True,
            "parallel": max(1, min(10, int(addon.getSetting("parallel_requests") or DEFAULT_PARALLEL_REQUESTS)))  # Increased max to 10
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
