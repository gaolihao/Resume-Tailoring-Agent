from langchain_openai import ChatOpenAI

from resume_agent.config import get_settings


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    kwargs: dict = {
        "model": settings.openai_model,
        "api_key": settings.openai_api_key,
        "temperature": temperature,
    }
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return ChatOpenAI(**kwargs)