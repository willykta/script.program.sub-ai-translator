from api import mock, openai, gemini, openrouter

PROVIDERS = {
    "Mock": mock,
    "OpenAI": openai,
    "Gemini": gemini,
    "OpenRouter": openrouter
}

def get_provider(name: str):
    return PROVIDERS.get(name, mock)
