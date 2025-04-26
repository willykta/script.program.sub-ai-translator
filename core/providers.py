from api import mock, openai, gemini

PROVIDERS = {
    "Mock": mock,
    "OpenAI": openai,
    "Gemini": gemini
}

def get_provider(name: str):
    return PROVIDERS.get(name, mock)
