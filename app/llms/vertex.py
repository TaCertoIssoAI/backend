"""factory for ChatGoogleGenerativeAI instances configured for Vertex AI.

centralizes project/location/vertexai config so call sites stay compact.
auth resolves automatically via GOOGLE_APPLICATION_CREDENTIALS (the SA JSON path).
"""

from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import get_settings


def make_vertex_chat(model: str, **kwargs: Any) -> ChatGoogleGenerativeAI:
    """build a ChatGoogleGenerativeAI in Vertex mode with project/location from settings.

    args:
        model: gemini model id, e.g. 'gemini-2.5-flash-lite'
        **kwargs: forwarded to ChatGoogleGenerativeAI (temperature, thinking_budget, etc.)
    """
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=model,
        vertexai=True,
        project=settings.VERTEX_PROJECT_ID,
        location=settings.VERTEX_LOCATION,
        **kwargs,
    )
