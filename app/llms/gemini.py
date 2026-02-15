"""
langchain wrapper for google gemini using native api.

provides a BaseChatModel implementation that supports gemini-specific features
like thinking_config that may not be available in langchain-google-genai.
"""

from typing import Any, List, Optional, Dict, Type, Union
from pydantic import BaseModel as PydanticBaseModel, Field as PydanticField
import json
import time
import httpx
import ssl

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from google import genai
from google.genai import types
from google.genai.errors import ServerError

from app.observability.logger.logger import get_logger

# create module logger
logger = get_logger(__name__)


class GeminiChatModel(BaseChatModel):
    """
    custom langchain wrapper for google gemini with native api support.

    supports gemini-specific features like thinking_config that may not be
    available in the official langchain-google-genai package.

    example:
        >>> from app.llms.gemini import GeminiChatModel
        >>> llm = GeminiChatModel(
        ...     model="gemini-3-pro-preview",
        ...     thinking_level="low",
        ...     temperature=0.0
        ... )
        >>> response = llm.invoke([HumanMessage(content="Hello")])
    """

    model: str = PydanticField(
        default="gemini-3-pro-preview",
        description="gemini model name"
    )
    google_api_key: Optional[str] = PydanticField(
        default=None,
        description="google api key (uses GOOGLE_API_KEY env var if not provided)"
    )
    temperature: Optional[float] = PydanticField(
        default=None,
        description="model temperature (0.0-2.0)"
    )
    max_output_tokens: Optional[int] = PydanticField(
        default=None,
        description="maximum number of tokens to generate"
    )
    thinking_level: Optional[str] = PydanticField(
        default=None,
        description="thinking level for extended reasoning: 'low', 'medium', or 'high'"
    )
    top_p: Optional[float] = PydanticField(
        default=None,
        description="nucleus sampling parameter"
    )
    top_k: Optional[int] = PydanticField(
        default=None,
        description="top-k sampling parameter"
    )

    class Config:
        """pydantic config"""
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        """return identifier for this llm type."""
        return "gemini-native"

    def _convert_messages_to_gemini_format(
        self,
        messages: List[BaseMessage]
    ) -> tuple[Optional[str], str]:
        """
        convert langchain messages to gemini format.

        args:
            messages: list of langchain messages

        returns:
            tuple of (system_instruction, user_content)
        """
        # separate system messages from other messages
        system_messages = [m for m in messages if isinstance(m, SystemMessage)]
        other_messages = [m for m in messages if not isinstance(m, SystemMessage)]

        # combine system messages
        system_instruction = None
        if system_messages:
            system_instruction = "\n\n".join(m.content for m in system_messages)

        # for now, combine all non-system messages into a single user message
        # TODO: support multi-turn conversations properly
        user_content = "\n\n".join(
            m.content for m in other_messages
            if isinstance(m, (HumanMessage, AIMessage))
        )

        return system_instruction, user_content

    def _create_http_client(self) -> httpx.Client:
        """
        create a robust HTTP client with better SSL and connection handling.

        returns:
            configured httpx.Client
        """
        # create SSL context with more lenient settings
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = True
        ssl_context.verify_mode = ssl.CERT_REQUIRED

        # configure connection limits and timeouts
        limits = httpx.Limits(
            max_keepalive_connections=5,
            max_connections=10,
            keepalive_expiry=30.0
        )

        timeout = httpx.Timeout(
            timeout=60.0,
            connect=10.0,
            read=30.0,
            write=10.0,
            pool=5.0
        )

        # create client with robust settings
        return httpx.Client(
            verify=ssl_context,
            limits=limits,
            timeout=timeout,
            http2=True,  # enable HTTP/2
            follow_redirects=True
        )

    def _build_generation_config(self) -> types.GenerateContentConfig:
        """
        build gemini generation config from model parameters.

        returns:
            configured GenerateContentConfig
        """
        config_dict: Dict[str, Any] = {}

        if self.temperature is not None:
            config_dict["temperature"] = self.temperature
        if self.max_output_tokens is not None:
            config_dict["max_output_tokens"] = self.max_output_tokens
        if self.top_p is not None:
            config_dict["top_p"] = self.top_p
        if self.top_k is not None:
            config_dict["top_k"] = self.top_k
        if self.thinking_level is not None:
            config_dict["thinking_config"] = types.ThinkingConfig(
                thinking_level=self.thinking_level
            )

        return types.GenerateContentConfig(**config_dict)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        generate response from gemini with retry logic for connection errors.

        args:
            messages: list of langchain messages
            stop: optional stop sequences
            run_manager: callback manager
            **kwargs: additional arguments

        returns:
            ChatResult with generated response
        """
        # retry configuration
        max_retries = 3
        base_delay = 1.0
        last_exception = None

        for attempt in range(max_retries):
            try:
                # initialize gemini client
                client_kwargs = {}
                if self.google_api_key:
                    client_kwargs["api_key"] = self.google_api_key
                client = genai.Client(**client_kwargs)

                # convert messages
                system_instruction, user_content = self._convert_messages_to_gemini_format(messages)

                # build config
                gen_config = self._build_generation_config()

                # add system instruction if present
                if system_instruction:
                    # create new config with system instruction
                    config_dict = gen_config.to_dict() if hasattr(gen_config, 'to_dict') else {}
                    config_dict["system_instruction"] = system_instruction
                    gen_config = types.GenerateContentConfig(**config_dict)

                # call gemini api
                response = client.models.generate_content(
                    model=self.model,
                    contents=user_content,
                    config=gen_config
                )

                # convert response to langchain format
                message = AIMessage(content=response.text)
                generation = ChatGeneration(message=message)

                return ChatResult(generations=[generation])

            except (httpx.ConnectError, httpx.RemoteProtocolError, ssl.SSLError, ServerError) as e:
                last_exception = e

                if attempt < max_retries - 1:
                    # exponential backoff
                    delay = base_delay * (2 ** attempt)

                    # check if it's a 503 rate limit error
                    error_type = "rate limit/overload" if isinstance(e, ServerError) else "connection"
                    logger.warning(
                        f"gemini API {error_type} error (attempt {attempt + 1}/{max_retries}): {str(e)}. "
                        f"retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"gemini API failed after {max_retries} attempts: {str(e)}"
                    )

            except Exception as e:
                # for non-connection errors, fail immediately
                logger.error(f"gemini API error: {str(e)}")
                raise

        # if we get here, all retries failed
        raise last_exception

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        async version of _generate with retry logic.

        args:
            messages: list of langchain messages
            stop: optional stop sequences
            run_manager: callback manager
            **kwargs: additional arguments

        returns:
            ChatResult with generated response
        """
        import asyncio

        # retry configuration
        max_retries = 3
        base_delay = 1.0
        last_exception = None

        for attempt in range(max_retries):
            try:
                # initialize async client with api key if provided
                client_kwargs = {}
                if self.google_api_key:
                    client_kwargs["api_key"] = self.google_api_key
                client = genai.Client(**client_kwargs)

                # convert messages
                system_instruction, user_content = self._convert_messages_to_gemini_format(messages)

                # build config
                gen_config = self._build_generation_config()

                # add system instruction if present
                if system_instruction:
                    config_dict = gen_config.to_dict() if hasattr(gen_config, 'to_dict') else {}
                    config_dict["system_instruction"] = system_instruction
                    gen_config = types.GenerateContentConfig(**config_dict)

                # call gemini api asynchronously
                # note: using aio client for true async support
                response = await client.aio.models.generate_content(
                    model=self.model,
                    contents=user_content,
                    config=gen_config
                )

                # convert response to langchain format
                message = AIMessage(content=response.text)
                generation = ChatGeneration(message=message)

                return ChatResult(generations=[generation])

            except (ServerError, httpx.ConnectError, httpx.RemoteProtocolError) as e:
                last_exception = e

                if attempt < max_retries - 1:
                    # exponential backoff
                    delay = base_delay * (2 ** attempt)

                    # check if it's a 503 rate limit error
                    error_type = "rate limit/overload" if isinstance(e, ServerError) else "connection"
                    logger.warning(
                        f"gemini API {error_type} error (async, attempt {attempt + 1}/{max_retries}): {str(e)}. "
                        f"retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"gemini API failed after {max_retries} async attempts: {str(e)}"
                    )

            except Exception as e:
                # for non-retryable errors, fail immediately
                logger.error(f"gemini API non-retryable error: {str(e)}")
                raise

        # if we get here, all retries failed
        raise last_exception

    def with_structured_output(
        self,
        schema: Union[Dict, Type[PydanticBaseModel]],
        **kwargs: Any
    ) -> Runnable:
        """
        bind a structured output schema to the model using gemini's native support.

        uses gemini's response_schema parameter to enforce structured JSON output
        and parses it into the pydantic model.

        args:
            schema: pydantic model class or dict schema for output structure
            **kwargs: additional arguments (method, include_raw, etc.)

        returns:
            runnable that returns structured output
        """
        # convert pydantic model to json schema if needed
        if isinstance(schema, type) and issubclass(schema, PydanticBaseModel):
            json_schema = schema.model_json_schema()
        else:
            json_schema = schema

        # create a custom model with response schema
        class StructuredGeminiModel(GeminiChatModel):
            """gemini model configured for structured output."""

            response_schema: Dict = json_schema
            output_parser: JsonOutputParser = JsonOutputParser(pydantic_object=schema if isinstance(schema, type) else None)

            def _generate(
                self,
                messages: List[BaseMessage],
                stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **gen_kwargs: Any,
            ) -> ChatResult:
                """generate with response schema."""
                # initialize client with api key if provided
                client_kwargs = {}
                if self.google_api_key:
                    client_kwargs["api_key"] = self.google_api_key
                client = genai.Client(**client_kwargs)

                # convert messages
                system_instruction, user_content = self._convert_messages_to_gemini_format(messages)

                # build config with response schema
                gen_config = self._build_generation_config()
                config_dict = {}

                if system_instruction:
                    config_dict["system_instruction"] = system_instruction
                if self.temperature is not None:
                    config_dict["temperature"] = self.temperature
                if self.max_output_tokens is not None:
                    config_dict["max_output_tokens"] = self.max_output_tokens
                if self.top_p is not None:
                    config_dict["top_p"] = self.top_p
                if self.top_k is not None:
                    config_dict["top_k"] = self.top_k
                if self.thinking_level is not None:
                    config_dict["thinking_config"] = types.ThinkingConfig(
                        thinking_level=self.thinking_level
                    )

                # add response schema for structured output
                config_dict["response_mime_type"] = "application/json"
                config_dict["response_schema"] = self.response_schema

                gen_config = types.GenerateContentConfig(**config_dict)

                # call gemini api
                response = client.models.generate_content(
                    model=self.model,
                    contents=user_content,
                    config=gen_config
                )

                # parse the JSON response into pydantic model
                try:
                    if isinstance(schema, type) and issubclass(schema, PydanticBaseModel):
                        parsed_output = schema.model_validate_json(response.text)
                        message = AIMessage(content=response.text, additional_kwargs={"parsed": parsed_output})
                    else:
                        parsed_output = json.loads(response.text)
                        message = AIMessage(content=response.text, additional_kwargs={"parsed": parsed_output})
                except Exception as e:
                    # if parsing fails, return raw text
                    logger.warning(f"failed to parse structured output: {str(e)}, returning raw text")
                    message = AIMessage(content=response.text)

                generation = ChatGeneration(message=message)
                return ChatResult(generations=[generation])

        # create instance with same parameters
        structured_model = StructuredGeminiModel(
            model=self.model,
            google_api_key=self.google_api_key,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            thinking_level=self.thinking_level,
            top_p=self.top_p,
            top_k=self.top_k
        )

        # create a chain that extracts the parsed output
        from langchain_core.runnables import RunnableLambda

        def extract_parsed(output):
            """extract parsed output from message."""
            if hasattr(output, 'additional_kwargs') and 'parsed' in output.additional_kwargs:
                return output.additional_kwargs['parsed']
            # fallback: try to parse as JSON
            if isinstance(schema, type) and issubclass(schema, PydanticBaseModel):
                return schema.model_validate_json(output.content)
            return json.loads(output.content)

        return structured_model | RunnableLambda(extract_parsed)
