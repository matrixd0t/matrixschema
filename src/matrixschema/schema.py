from typing import Any, Callable, Union, Awaitable

from docstring_parser import parse as _parse_docstring
from typing import Literal, get_args, get_origin, get_type_hints
import sys
import inspect

AsyncOrSyncFunction = Union[Callable[..., object], Callable[..., Awaitable[object]]]

_PRIMITIVES: dict[type, str] = {
    int: 'integer',
    float: 'number',
    str: 'string',
    bool: 'boolean',
    type(None): 'null',
}


def _parse_param_docs(fn: AsyncOrSyncFunction) -> dict[str, str]:
    return {
        p.arg_name: p.description or ''
        for p in _parse_docstring(inspect.getdoc(fn) or '').params
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


def _annotation_to_schema(annotation: Any) -> dict[str, Any]:
    # empty / Any - no constraints
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
        arg_schemas = [_annotation_to_schema(a) for a in union_args]
        non_null = [s for s in arg_schemas if s != null_schema]

        if len(arg_schemas) == 2 and null_schema in arg_schemas:
            # Optional[X] / X | None - keep flat structure
            return {'anyOf': [non_null[0], null_schema]}
        return {'anyOf': arg_schemas}

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Literal["a", "b"]
    if origin is Literal:
        return {'enum': list(args)}

    # list[X]
    if origin is list:
        schema: dict[str, Any] = {'type': 'array'}
        if args:
            schema['items'] = _annotation_to_schema(args[0])
        return schema

    # tuple[X, Y] / tuple[X, ...]
    if origin is tuple:
        if not args:
            return {'type': 'array'}
        if len(args) == 2 and args[1] is Ellipsis:
            # tuple[int, ...] - variable length
            return {'type': 'array', 'items': _annotation_to_schema(args[0])}
        # tuple[int, str, float] - fixed structure
        return {
            'type': 'array',
            'prefixItems': [_annotation_to_schema(a) for a in args],
            'minItems': len(args),
            'maxItems': len(args),
        }

    # dict[K, V]
    if origin is dict:
        schema = {'type': 'object'}
        if len(args) == 2:
            val_schema = _annotation_to_schema(args[1])
            if val_schema:
                schema['additionalProperties'] = val_schema
        return schema

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
    hints = get_type_hints(fn)
    param_docs = _parse_param_docs(fn)

    properties: dict[str, Any] = {}
    required: list[str] = []  # in strict mode — all parameters

    for name, param in sig.parameters.items():
        if name in ('self', 'cls'):
            continue

        annotation = hints.get(name, inspect.Parameter.empty)
        has_default = param.default is not inspect.Parameter.empty
        base_schema = _annotation_to_schema(annotation)

        # optional - anyOf [type, null] so the LLM can explicitly pass null
        if has_default:
            prop = _make_strict_schema(base_schema)
        else:
            prop = base_schema

        if description := param_docs.get(name):
            prop['description'] = description

        properties[name] = prop
        required.append(name)  # always

    description = (inspect.getdoc(fn) or '').replace('\n', ' ').strip()

    return {
        'type': 'function',
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
