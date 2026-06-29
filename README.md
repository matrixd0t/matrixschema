# matrixschema

Convert Python function signatures (type annotations + docstrings) into OpenAI-compatible JSON Schema (tool call format).

```python
from src.matrixschema import build_json_schema


def get_weather(city: str, units: Literal["metric", "imperial"] = "metric") -> str:
    """Get the current weather for a city."""
    ...


schema = build_json_schema(get_weather)
# {
#   "type": "function",
#   "name": "get_weather",
#   "description": "Get the current weather for a city.",
#   "strict": True,
#   "parameters": {
#     "type": "object",
#     "properties": {
#       "city": {"type": "string"},
#       "units": {"anyOf": [{"enum": ["metric", "imperial"]}, {"type": "null"}]}
#     },
#     "additionalProperties": False,
#     "required": ["city", "units"]
#   }
# }
```

Supports: `int`, `float`, `str`, `bool`, `None`, `Optional[X]`, `Union[...]`, `Literal[...]`, `list[X]`, `tuple[X, ...]`, `dict[K, V]`, nested `pydantic.BaseModel`.

Requires Python 3.10+.

Written with love by dotmatrix.