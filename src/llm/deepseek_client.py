from langchain_openai import ChatOpenAI
from src.config import get_settings


def build_llm(temperature: float = 0.3) -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        model=s.deepseek_model,
        api_key=s.deepseek_api_key,
        base_url=s.deepseek_base_url,
        temperature=temperature,
        max_tokens=2048,
    )
