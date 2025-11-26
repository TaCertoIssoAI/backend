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
    tools: Optional[List[Any]] = PydanticField(
        default=None,
        description="list of tools to enable (e.g., [types.Tool(google_search=types.GoogleSearch())])"
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
        if self.tools is not None:
            config_dict["tools"] = self.tools

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
                # create HTTP client with robust settings
                http_client = self._create_http_client()

                # initialize client with api key and custom http client
                client_kwargs = {"http_options": {"client": http_client}}
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
                # extract grounding metadata if available
                additional_kwargs = {}
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                        additional_kwargs["grounding_metadata"] = {
                            "web_search_queries": [
                                str(q) for q in getattr(candidate.grounding_metadata, 'web_search_queries', [])
                            ],
                            "grounding_chunks": [
                                {
                                    "uri": chunk.web.uri,
                                    "title": chunk.web.title
                                }
                                for chunk in getattr(candidate.grounding_metadata, 'grounding_chunks', [])
                                if hasattr(chunk, 'web')
                            ],
                            "grounding_supports": [
                                {
                                    "segment_text": support.segment.text,
                                    "chunk_indices": list(support.grounding_chunk_indices)
                                }
                                for support in getattr(candidate.grounding_metadata, 'grounding_supports', [])
                            ]
                        }

                message = AIMessage(content=response.text, additional_kwargs=additional_kwargs)
                generation = ChatGeneration(message=message)

                # close http client
                http_client.close()

                return ChatResult(generations=[generation])

            except (httpx.ConnectError, httpx.RemoteProtocolError, ssl.SSLError) as e:
                last_exception = e
                http_client.close()

                if attempt < max_retries - 1:
                    # exponential backoff
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"gemini API connection error (attempt {attempt + 1}/{max_retries}): {str(e)}. "
                        f"retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"gemini API connection failed after {max_retries} attempts: {str(e)}"
                    )

            except Exception as e:
                # for non-connection errors, fail immediately
                logger.error(f"gemini API error: {str(e)}")
                if 'http_client' in locals():
                    http_client.close()
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
        async version of _generate.

        args:
            messages: list of langchain messages
            stop: optional stop sequences
            run_manager: callback manager
            **kwargs: additional arguments

        returns:
            ChatResult with generated response
        """
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
        # extract grounding metadata if available
        additional_kwargs = {}
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                additional_kwargs["grounding_metadata"] = {
                    "web_search_queries": [
                        str(q) for q in getattr(candidate.grounding_metadata, 'web_search_queries', [])
                    ],
                    "grounding_chunks": [
                        {
                            "uri": chunk.web.uri,
                            "title": chunk.web.title
                        }
                        for chunk in getattr(candidate.grounding_metadata, 'grounding_chunks', [])
                        if hasattr(chunk, 'web')
                    ],
                    "grounding_supports": [
                        {
                            "segment_text": support.segment.text,
                            "chunk_indices": list(support.grounding_chunk_indices)
                        }
                        for support in getattr(candidate.grounding_metadata, 'grounding_supports', [])
                    ]
                }

        message = AIMessage(content=response.text, additional_kwargs=additional_kwargs)
        generation = ChatGeneration(message=message)

        return ChatResult(generations=[generation])

    def with_structured_output(
        self,
        schema: Union[Dict, Type[PydanticBaseModel]],
        **kwargs: Any
    ) -> Runnable:
        """
        bind a structured output schema to the model using gemini's native support.

        supports two methods:
        - json_mode (default): uses response_schema parameter for JSON output
          (incompatible with tools)
        - function_calling: defines schema as a function that LLM must call
          (compatible with tools like Google Search)

        args:
            schema: pydantic model class or dict schema for output structure
            **kwargs: additional arguments:
                - method: "json_mode" (default) or "function_calling"
                - include_raw: whether to include raw response
                - tool_choice: for function_calling, "any" forces the function call

        returns:
            runnable that returns structured output

        example with tools:
            >>> llm = GeminiChatModel(
            ...     model="gemini-2.5-flash",
            ...     tools=[google_search_tool]
            ... )
            >>> structured_llm = llm.with_structured_output(
            ...     schema=VerdictSchema,
            ...     method="function_calling"
            ... )
        """
        # get method from kwargs
        method = kwargs.get("method", "json_mode")

        # convert pydantic model to json schema if needed
        if isinstance(schema, type) and issubclass(schema, PydanticBaseModel):
            json_schema = schema.model_json_schema()
            schema_name = schema.__name__
        else:
            json_schema = schema
            schema_name = json_schema.get("title", "output_schema")

        if method == "function_calling":
            # use function calling for structured output (compatible with tools)
            return self._with_structured_output_function_calling(schema, json_schema, schema_name, **kwargs)
        else:
            # use json mode for structured output (incompatible with tools)
            return self._with_structured_output_json_mode(schema, json_schema, **kwargs)

    def _with_structured_output_json_mode(
        self,
        schema: Union[Dict, Type[PydanticBaseModel]],
        json_schema: Dict,
        **kwargs: Any
    ) -> Runnable:
        """implement structured output using JSON mode (incompatible with tools)."""
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

                # gemini doesn't support both tools and response_mime_type at the same time
                # only add response schema if tools are not being used
                if self.tools is not None:
                    config_dict["tools"] = self.tools
                    logger.warning(
                        "tools are set - skipping response_mime_type and response_schema "
                        "due to gemini API limitation"
                    )
                else:
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
            top_k=self.top_k,
            tools=self.tools
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

    def _with_structured_output_function_calling(
        self,
        schema: Union[Dict, Type[PydanticBaseModel]],
        json_schema: Dict,
        schema_name: str,
        **kwargs: Any
    ) -> Runnable:
        """
        implement structured output using function calling (compatible with tools).

        converts the schema to a function declaration that the LLM must call,
        allowing it to work alongside other tools like Google Search.
        """
        from langchain_core.runnables import RunnableLambda

        # convert JSON schema to Gemini function declaration
        function_declaration = self._json_schema_to_function(json_schema, schema_name)

        # create combined tools list with the output function
        output_function_tool = types.Tool(function_declarations=[function_declaration])
        combined_tools = [output_function_tool]
        if self.tools:
            combined_tools.extend(self.tools if isinstance(self.tools, list) else [self.tools])

        # create model with function calling
        class FunctionCallingGeminiModel(GeminiChatModel):
            """gemini model with function calling for structured output."""

            output_schema = schema
            output_function_name: str = schema_name

            def _generate(
                self,
                messages: List[BaseMessage],
                stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **gen_kwargs: Any,
            ) -> ChatResult:
                """generate with function calling."""
                # create HTTP client
                http_client = self._create_http_client()

                # initialize client
                client_kwargs = {"http_options": {"client": http_client}}
                if self.google_api_key:
                    client_kwargs["api_key"] = self.google_api_key
                client = genai.Client(**client_kwargs)

                try:
                    # convert messages
                    system_instruction, user_content = self._convert_messages_to_gemini_format(messages)

                    # build config
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

                    # add tools (including output function)
                    config_dict["tools"] = combined_tools

                    # force calling the output function last
                    # note: gemini doesn't have explicit tool_choice, but we can use response mime type
                    # to encourage function calling

                    gen_config = types.GenerateContentConfig(**config_dict)

                    # call gemini api
                    response = client.models.generate_content(
                        model=self.model,
                        contents=user_content,
                        config=gen_config
                    )

                    # extract function call from response
                    parsed_output = self._extract_function_call(response, self.output_function_name)

                    # parse into pydantic model
                    if isinstance(self.output_schema, type) and issubclass(self.output_schema, PydanticBaseModel):
                        parsed_output = self.output_schema.model_validate(parsed_output)

                    message = AIMessage(
                        content=response.text if hasattr(response, 'text') else str(parsed_output),
                        additional_kwargs={"parsed": parsed_output}
                    )

                    return ChatResult(generations=[ChatGeneration(message=message)])

                finally:
                    http_client.close()

            def _extract_function_call(self, response, function_name: str) -> Dict:
                """extract function call arguments from gemini response."""
                if not hasattr(response, 'candidates') or not response.candidates:
                    raise ValueError("no candidates in response")

                candidate = response.candidates[0]
                if not hasattr(candidate, 'content') or not hasattr(candidate.content, 'parts'):
                    raise ValueError("no content parts in response")

                # look for function call in parts
                for part in candidate.content.parts:
                    if hasattr(part, 'function_call') and part.function_call.name == function_name:
                        # convert function call args to dict
                        return dict(part.function_call.args)

                # if no function call found, try to parse text as JSON
                if hasattr(response, 'text'):
                    logger.warning(f"no function call found for {function_name}, attempting JSON parse")
                    return json.loads(response.text)

                raise ValueError(f"no function call found for {function_name}")

        # create instance
        function_model = FunctionCallingGeminiModel(
            model=self.model,
            google_api_key=self.google_api_key,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            thinking_level=self.thinking_level,
            top_p=self.top_p,
            top_k=self.top_k,
            tools=self.tools
        )

        # create extraction chain
        def extract_parsed(output):
            """extract parsed output from message."""
            if hasattr(output, 'additional_kwargs') and 'parsed' in output.additional_kwargs:
                return output.additional_kwargs['parsed']
            raise ValueError("no parsed output in response")

        return function_model | RunnableLambda(extract_parsed)

    def _resolve_schema_refs(self, schema: Any, defs: Dict) -> Any:
        """
        recursively resolve $ref references in a JSON schema.

        args:
            schema: schema dict/list/primitive that may contain $ref
            defs: definitions dict from the root schema

        returns:
            schema with all $ref resolved and inlined
        """
        if isinstance(schema, dict):
            # if this is a $ref, resolve it
            if "$ref" in schema:
                ref_path = schema["$ref"]
                # extract the definition name from "#/$defs/DefinitionName"
                if ref_path.startswith("#/$defs/"):
                    def_name = ref_path.replace("#/$defs/", "")
                    if def_name in defs:
                        # recursively resolve the referenced schema
                        return self._resolve_schema_refs(defs[def_name], defs)
                return schema

            # recursively process all values in the dict
            resolved = {}
            for key, value in schema.items():
                if key == "$defs":
                    # skip $defs - we're inlining everything
                    continue
                resolved[key] = self._resolve_schema_refs(value, defs)
            return resolved
        elif isinstance(schema, list):
            # recursively process list items
            return [self._resolve_schema_refs(item, defs) for item in schema]
        else:
            # primitive value, return as-is
            return schema

    def _json_schema_to_function(self, json_schema: Dict, function_name: str) -> types.FunctionDeclaration:
        """
        convert JSON schema to Gemini function declaration.

        args:
            json_schema: JSON schema dict
            function_name: name for the function

        returns:
            Gemini FunctionDeclaration
        """
        # extract description
        description = json_schema.get("description", f"Return structured output as {function_name}")

        # get definitions for resolving $ref
        defs = json_schema.get("$defs", {})

        # resolve all $ref references in properties
        properties = json_schema.get("properties", {})
        resolved_properties = self._resolve_schema_refs(properties, defs)

        required = json_schema.get("required", [])

        # build parameter schema without $defs
        parameters = {
            "type": "object",
            "properties": resolved_properties,
            "required": required
        }

        return types.FunctionDeclaration(
            name=function_name,
            description=description,
            parameters=parameters
        )
