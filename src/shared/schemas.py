"""Pydantic schema utilities for structured LLM outputs.

## RAG Theory: Structured Outputs

LLMs can produce unreliable JSON without schema enforcement. Pydantic models:
1. Define expected response structure with type hints
2. Validate LLM outputs automatically
3. Provide IDE autocompletion and type safety
4. Generate JSON Schema for OpenRouter's strict mode

## Library Usage

Pydantic BaseModel generates JSON Schema via model_json_schema().
OpenRouter's response_format accepts this schema for guaranteed compliance.
With strict mode, the LLM is constrained to produce only valid JSON.

## Data Flow

1. Define Pydantic model with expected fields
2. Call get_openrouter_schema() to convert to OpenRouter format
3. Pass schema in API request's response_format parameter
4. LLM produces schema-compliant JSON
5. Validate with model.model_validate_json()
"""

from typing import TypeVar, Type

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def get_openrouter_schema(model: Type[T]) -> dict:
    """Convert Pydantic model to OpenRouter response_format schema.

    Creates the response_format dictionary that tells OpenRouter to
    enforce JSON Schema compliance. With strict=True, the LLM is
    constrained to produce only valid JSON matching the schema.

    Args:
        model: A Pydantic BaseModel class defining the expected response.

    Returns:
        Dict suitable for OpenRouter's response_format parameter.

    Example:
        >>> from pydantic import BaseModel
        >>> class Result(BaseModel):
        ...     answer: str
        >>> schema = get_openrouter_schema(Result)
        >>> schema["type"]
        'json_schema'
        >>> schema["json_schema"]["strict"]
        True
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": model.__name__,
            "strict": True,
            "schema": model.model_json_schema(),
        },
    }
