from langchain_google_genai import ChatGoogleGenerativeAI

from resume_agent.config import get_settings


def get_llm(temperature: float | None = None) -> ChatGoogleGenerativeAI:
    """Return a Gemini chat model (default: gemini-3.5-flash-lite)."""
    settings = get_settings()
    if not settings.api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY (or GEMINI_API_KEY) is not set. "
            "Copy .env.example to .env and add your Gemini API key from "
            "https://aistudio.google.com/apikey"
        )

    kwargs: dict = {
        "model": settings.gemini_model,
        "google_api_key": settings.api_key,
    }

    # Gemini 3.x may reject sampling knobs; only pass when explicitly requested.
    if temperature is not None:
        kwargs["temperature"] = temperature

    thinking = settings.gemini_thinking_level.strip().lower()
    if thinking:
        # Alias for reasoning_effort on Gemini 3+
        kwargs["thinking_level"] = thinking

    return ChatGoogleGenerativeAI(**kwargs)
