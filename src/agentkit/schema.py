"""Automatic tool schema generation from Python function signatures."""
from __future__ import annotations

import inspect
import logging
import types as _stdlib_types
from collections.abc import Callable
from typing import Any, Union, get_args, get_origin, get_type_hints

logger = logging.getLogger(__name__)

_TYPE_MAP: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null",
}


def _annotation_to_json_type(
    annotation: Any,
    func_name: str,
    param_name: str,
) -> str:
    """Convert a Python type annotation to a JSON Schema type string."""
    if annotation is inspect.Parameter.empty:
        logger.warning(
            "function_to_tool_definition: %s.%s has no type annotation; "
            "defaulting to 'string'. Add a type hint to suppress this warning.",
            func_name,
            param_name,
        )
        return "string"

    # X | Y  (Python 3.10+ union syntax → types.UnionType)
    if isinstance(annotation, _stdlib_types.UnionType):
        non_none = [a for a in annotation.__args__ if a is not type(None)]
        if non_none:
            return _annotation_to_json_type(non_none[0], func_name, param_name)

    # typing.Union[X, Y] / Optional[X]
    if get_origin(annotation) is Union:
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if non_none:
            return _annotation_to_json_type(non_none[0], func_name, param_name)

    mapped = _TYPE_MAP.get(annotation)
    if mapped is not None:
        return mapped

    logger.warning(
        "function_to_tool_definition: %s.%s has unsupported annotation %r; "
        "defaulting to 'string'.",
        func_name,
        param_name,
        annotation,
    )
    return "string"


def function_to_input_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """Build a JSON Schema 'parameters' object from a function's signature.

    Uses type annotations for property types. Parameters without defaults
    are added to 'required'. Parameters without annotations log a warning.
    """
    sig = inspect.signature(func)
    func_name = func.__name__

    try:
        hints = get_type_hints(func)
    except Exception:  # noqa: BLE001
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        annotation = hints.get(name, param.annotation)
        json_type = _annotation_to_json_type(annotation, func_name, name)
        properties[name] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def build_tool_definition(
    name: str,
    description: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Wrap name, description, and JSON-Schema parameters into an OpenAI tool definition."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def function_to_tool_definition(func: Callable[..., Any]) -> dict[str, Any]:
    """Build an OpenAI-format tool definition from a Python function.

    The tool name is func.__name__. The description is the first paragraph
    of the docstring (everything before the first blank line), cleaned of
    leading whitespace. Write docstrings as model instructions.
    """
    description = ""
    if func.__doc__:
        description = inspect.cleandoc(func.__doc__).split("\n\n")[0].strip()

    return build_tool_definition(func.__name__, description, function_to_input_schema(func))
