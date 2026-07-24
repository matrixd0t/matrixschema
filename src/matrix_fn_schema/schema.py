import dataclasses
import datetime
import enum
import inspect
import sys
import uuid
from typing import Any, Awaitable, Callable, Union

from docstring_parser import parse as _parse_docstring
from typing import Annotated, Final, Literal, get_args, get_origin, get_type_hints

AsyncOrSyncFunction = Union[Callable[..., object], Callable[..., Awaitable[object]]]

_PRIMITIVES: dict[type, str] = {
    int: 'integer',
    float: 'number',
    str: 'string',
    bool: 'boolean',
    type(None): 'null',
}


def _get_union_args(annotation: Any) -> tuple[Any, ...] | None:
    """
    Returns Union/Optional/X|Y arguments, or None if not a Union.
    Handles typing.Union and types.UnionType (Python 3.10+ pipe syntax).
    """
    if get_origin(annotation) is Union:
        return get_args(annotation)
    if sys.version_info >= (3, 10):
        import types
        if type(annotation) is types.UnionType:
            return get_args(annotation)
    return None


def _literal_json_type(value: Any) -> str | None:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    return None


def _annotation_to_schema(annotation: Any, _seen: set | None = None) -> dict[str, Any]:
    if _seen is None:
        _seen = set()

    # Annotated[T, ...] - unwrap to the base type
    origin = get_origin(annotation)
    if origin is Annotated:
        return _annotation_to_schema(get_args(annotation)[0], _seen)

    # Final[T] - unwrap to T
    if hasattr(annotation, '__origin__') and annotation.__origin__ is Final:
        return _annotation_to_schema(get_args(annotation)[0], _seen)

    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        members = {e.name: e.value for e in annotation}
        values = list(members.values())
        first_val = values[0] if values else None
        if isinstance(first_val, int):
            return {'type': 'integer', 'enum': values}
        return {'type': 'string', 'enum': values}

    if annotation is uuid.UUID:
        return {'type': 'string', 'format': 'uuid'}

    if annotation is datetime.datetime:
        return {'type': 'string', 'format': 'date-time'}

    if annotation is datetime.date:
        return {'type': 'string', 'format': 'date'}

    if annotation is datetime.time:
        return {'type': 'string', 'format': 'time'}

    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}

    if annotation is type(None):
        return {'type': 'null'}

    if annotation in _PRIMITIVES:
        return {'type': _PRIMITIVES[annotation]}

    # Union / Optional / X | Y
    union_args = _get_union_args(annotation)
    if union_args is not None:
        null_schema = {'type': 'null'}
        arg_schemas = [_annotation_to_schema(a, _seen) for a in union_args]
        non_null = [s for s in arg_schemas if s != null_schema]

        if len(arg_schemas) == 2 and null_schema in arg_schemas:
            # Optional[X] / X | None - keep flat structure
            return {'anyOf': [non_null[0], null_schema]}
        return {'anyOf': arg_schemas}

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Literal["a", "b"]
    if origin is Literal:
        values = list(args)
        types = {
            json_type
            for value in values
            if (json_type := _literal_json_type(value)) is not None
        }
        if types <= {"integer", "number"} and types:
            return {
                "type": "number" if "number" in types else "integer",
                "enum": values,
            }
        if len(types) == 1:
            return {
                "type": next(iter(types)),
                "enum": values,
            }
        if types:
            return {
                "anyOf": [{"type": json_type} for json_type in sorted(types)],
                "enum": values,
            }

    # list[X]
    if origin is list:
        schema: dict[str, Any] = {'type': 'array'}
        if args:
            schema['items'] = _annotation_to_schema(args[0], _seen)
        return schema

    # tuple[X, Y] / tuple[X, ...]
    if origin is tuple:
        if not args:
            return {'type': 'array'}
        if len(args) == 2 and args[1] is Ellipsis:
            # tuple[int, ...] - variable length
            return {'type': 'array', 'items': _annotation_to_schema(args[0], _seen)}
        # tuple[int, str, float] - fixed structure
        return {
            'type': 'array',
            'prefixItems': [_annotation_to_schema(a, _seen) for a in args],
            'minItems': len(args),
            'maxItems': len(args),
        }

    # dict[K, V]
    if origin is dict:
        schema = {'type': 'object'}
        if len(args) == 2:
            val_schema = _annotation_to_schema(args[1], _seen)
            if val_schema:
                schema['additionalProperties'] = val_schema
        return schema

    # Dataclass - recursive field resolution
    if dataclasses.is_dataclass(annotation) and isinstance(annotation, type):
        if id(annotation) in _seen:
            return {}
        _seen.add(id(annotation))
        dc_hints = get_type_hints(annotation)
        dc_fields = {f.name: f for f in dataclasses.fields(annotation)}
        properties = {}
        required = []
        for field_name, field_type in dc_hints.items():
            field_def = dc_fields.get(field_name)
            has_default = (field_def.default is not dataclasses.MISSING
                           or field_def.default_factory is not dataclasses.MISSING)
            field_schema = _annotation_to_schema(field_type, _seen)
            if has_default:
                properties[field_name] = _make_strict_schema(field_schema)
            else:
                properties[field_name] = field_schema
            required.append(field_name)
        _seen.discard(id(annotation))
        return {
            'type': 'object',
            'properties': properties,
            'required': required,
            'additionalProperties': False,
        }

    # TypedDict - explicit property schema
    if isinstance(annotation, type) and hasattr(annotation, '__required_keys__'):
        if id(annotation) in _seen:
            return {}
        _seen.add(id(annotation))
        td_hints = get_type_hints(annotation)
        required_keys = annotation.__required_keys__
        properties = {}
        required = []
        for field_name, field_type in td_hints.items():
            field_schema = _annotation_to_schema(field_type, _seen)
            properties[field_name] = field_schema
            if field_name in required_keys:
                required.append(field_name)
        _seen.discard(id(annotation))
        return {
            'type': 'object',
            'properties': properties,
            'required': required,
            'additionalProperties': False,
        }

    # Pydantic BaseModel - nested schema
    try:
        from pydantic import BaseModel
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return annotation.model_json_schema()
    except ImportError:
        pass

    return {}


def _is_optional_param(annotation: Any, default: Any) -> bool:
    if default is not inspect.Parameter.empty:
        return True
    origin = get_origin(annotation)
    return get_origin(origin) is Union and type(None) in get_args(annotation)


def _make_strict_schema(base: dict[str, Any]) -> dict[str, Any]:
    """
    In strict mode, a parameter with a default must accept null
    (the LLM will pass null instead of omitting the argument).
    If the schema already has anyOf with null, do not duplicate.
    """
    null_schema = {'type': 'null'}
    if not base:
        return null_schema
    # already nullable
    if 'anyOf' in base and null_schema in base['anyOf']:
        return base
    return {'anyOf': [base, null_schema]}


def build_json_schema(fn: AsyncOrSyncFunction) -> dict[str, Any]:
    sig = inspect.signature(fn)
    if sys.version_info >= (3, 11):
        hints = get_type_hints(fn, include_extras=True)
    else:
        hints = get_type_hints(fn)

    parsed = _parse_docstring(inspect.getdoc(fn) or '')
    param_docs = {p.arg_name: p.description or '' for p in parsed.params}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name in ('self', 'cls'):
            continue

        annotation = hints.get(name, inspect.Parameter.empty)
        has_default = param.default is not inspect.Parameter.empty
        base_schema = _annotation_to_schema(annotation)

        if has_default:
            prop = _make_strict_schema(base_schema)
        else:
            prop = base_schema

        description = param_docs.get(name) or ''

        if has_default:
            default_repr = repr(param.default)
            if description:
                description = f"{description} (default: {default_repr})"
            else:
                description = f"Default: {default_repr}"

        if description:
            prop['description'] = description

        properties[name] = prop
        required.append(name)

    parts = []
    if parsed.short_description:
        parts.append(parsed.short_description)
    if parsed.long_description:
        parts.append(parsed.long_description)
    description = '\n\n'.join(parts)

    return {
        'name': fn.__name__,
        'description': description,
        'strict': True,
        'parameters': {
            'type': 'object',
            'properties': properties,
            'additionalProperties': False,
            'required': required,
        },
    }
