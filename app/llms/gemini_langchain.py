"""
langchain wrapper for google gemini using official langchain-google-genai package.

provides integration with google search grounding tools and structured output.
"""

import os
from typing import Optional, Type

from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import Runnable
from google.genai import types


class ChatGoogleGenerativeAIWithTools(ChatGoogleGenerativeAI):
    """
    subclass that disables structured output when tools are present.

    gemini doesn't support both tools and structured output simultaneously.
    """

    def __init__(self, **kwargs):
        """store tools reference before parent init."""
        # capture tools before parent processes them
        tools_value = kwargs.get('tools') or kwargs.get('model_kwargs', {}).get('tools')

        print(f"[ChatGoogleGenerativeAIWithTools] Initializing")
        print(f"[ChatGoogleGenerativeAIWithTools]   Captured tools: {tools_value}")

        # call parent init
        super().__init__(**kwargs)

        # set tools after parent init (using object.__setattr__ to bypass Pydantic)
        object.__setattr__(self, '_tools_list', tools_value)

        print(f"[ChatGoogleGenerativeAIWithTools]   After init, _tools_list: {self._tools_list}")

    def with_structured_output(self, schema: Type[BaseModel], **kwargs) -> Runnable:
        """disable structured output when tools are present."""
        print(f"[ChatGoogleGenerativeAIWithTools] with_structured_output called")
        print(f"[ChatGoogleGenerativeAIWithTools] _tools_list present: {self._tools_list is not None}")
        if self._tools_list:
            print(f"[ChatGoogleGenerativeAIWithTools] Tools detected - returning self (no structured output)")
            return self
        print(f"[ChatGoogleGenerativeAIWithTools] No tools - using parent's structured output")
        return super().with_structured_output(schema, **kwargs)


def create_google_search_tool() -> types.Tool:
    """
    create google search grounding tool for gemini models.

    returns:
        types.Tool configured with google_search
    """
    return types.Tool(google_search=types.GoogleSearch())


def extract_grounding_metadata(response):
    """
    extract google search grounding metadata from gemini response.

    args:
        response: langchain AIMessage response from gemini

    returns:
        dict with grounding metadata or None if not found

    example:
        >>> response = model.invoke("what's the weather today?")
        >>> metadata = extract_grounding_metadata(response)
        >>> if metadata:
        ...     print(f"Search queries: {metadata.get('search_queries', [])}")
        ...     print(f"Grounding chunks: {len(metadata.get('grounding_chunks', []))}")
    """
    # check response_metadata first
    if hasattr(response, 'response_metadata'):
        metadata = response.response_metadata
        if metadata and 'grounding_metadata' in metadata:
            return metadata['grounding_metadata']

    # check additional_kwargs
    if hasattr(response, 'additional_kwargs'):
        kwargs = response.additional_kwargs
        if kwargs and 'grounding_metadata' in kwargs:
            return kwargs['grounding_metadata']

    return None


def has_grounding(response) -> bool:
    """
    check if response used google search grounding.

    args:
        response: langchain AIMessage response from gemini

    returns:
        True if grounding metadata is present
    """
    return extract_grounding_metadata(response) is not None


def create_gemini_model(
    model: str = "gemini-2.5-flash",
    enable_search: bool = False,
    temperature: float = 0.0,
    google_api_key: Optional[str] = None,
    **kwargs
) -> ChatGoogleGenerativeAI:
    """
    factory function to create a gemini model with optional google search.

    args:
        model: gemini model name
        enable_search: whether to enable google search grounding
        temperature: sampling temperature
        google_api_key: google api key (optional, uses env var if not provided)
        **kwargs: additional arguments passed to ChatGoogleGenerativeAI

    returns:
        ChatGoogleGenerativeAI instance

    example:
        >>> # basic model without search
        >>> model = create_gemini_model(model="gemini-2.5-flash")
        >>>
        >>> # model with google search grounding
        >>> search_model = create_gemini_model(
        ...     model="gemini-2.5-flash",
        ...     enable_search=True
        ... )
        >>>
        >>> # with structured output (no search)
        >>> from pydantic import BaseModel
        >>> class Answer(BaseModel):
        ...     verdict: str
        ...     confidence: float
        >>>
        >>> structured_model = create_gemini_model(model="gemini-2.5-flash")
        >>> chain = structured_model.with_structured_output(Answer)
    """
    print(f"[create_gemini_model] Called with enable_search={enable_search}")

    api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("google_api_key must be provided or GOOGLE_API_KEY env var must be set")

    tools = None
    if enable_search:
        tools = [create_google_search_tool()]
        print(f"[create_gemini_model] Created tools: {tools}")
    else:
        print(f"[create_gemini_model] No tools - enable_search=False")

    print(f"[create_gemini_model] About to create instance with tools={tools is not None}")

    instance = ChatGoogleGenerativeAIWithTools(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
        tools=tools,
        **kwargs
    )

    print(f"[create_gemini_model] Instance created, _tools_list={instance._tools_list is not None}")

    return instance
