from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class LLMConfig(BaseModel):
    """Configuration for LLM model calls"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "model_name": "gpt-4o-mini",
            "temperature": 0.0,
            "timeout": 30.0
        }
    })

    model_name: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model name to use"
    )
    temperature: float = Field(
        default=0.0,
        description="Model temperature (0.0 for deterministic, higher for more creative)",
        ge=0.0,
        le=2.0
    )
    timeout: Optional[float] = Field(
        default=30.0,
        description="Timeout in seconds for the model call",
        gt=0
    )
