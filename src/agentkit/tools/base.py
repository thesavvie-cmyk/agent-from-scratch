"""Tool abstraction layer — BaseTool, FunctionTool, @tool decorator (block 7)."""
from __future__ import annotations

import functools
import inspect
from abc import ABC, abstractmethod
from typing import Any

from agentkit.context import ExecutionContext
from agentkit.schema import build_tool_definition, function_to_tool_definition


class BaseTool(ABC):
    """Abstract base for all agent tools.

    Subclasses must implement execute(). The tool_definition property exposes
    the OpenAI-format schema; subclasses can override _generate_definition()
    or pass a pre-built dict to __init__.
    """

    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
        tool_definition: dict[str, Any] | None = None,
    ) -> None:
        self._name = name
        self._description = description
        self._tool_definition = tool_definition

    @property
    def name(self) -> str:
        return self._name or type(self).__name__.lower()

    @property
    def description(self) -> str:
        if self._description is not None:
            return self._description
        return inspect.cleandoc(type(self).__doc__ or "")

    @property
    def tool_definition(self) -> dict[str, Any]:
        if self._tool_definition is None:
            self._tool_definition = self._generate_definition()
        return self._tool_definition

    def _generate_definition(self) -> dict[str, Any]:
        """Build an OpenAI tool definition. Override in subclasses."""
        return build_tool_definition(
            self.name, self.description, {"type": "object", "properties": {}}
        )

    @abstractmethod
    async def execute(self, context: ExecutionContext, **kwargs: Any) -> Any:
        """Run the tool and return its result."""

    async def __call__(self, context: ExecutionContext, **kwargs: Any) -> Any:
        return await self.execute(context, **kwargs)


class FunctionTool(BaseTool):
    """Wraps a plain Python function (sync or async) as a BaseTool.

    The function may declare an optional first parameter named ``context``
    (type ExecutionContext). If present, the live context is injected at call
    time; the parameter is excluded from the tool schema sent to the model.
    """

    def __init__(
        self,
        func: Any,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        self._func = func
        self._is_async = inspect.iscoroutinefunction(func)
        sig = inspect.signature(func)
        self._needs_context = "context" in sig.parameters

        _name = name or func.__name__
        # Extract description from docstring if not explicitly given
        if description is not None:
            _description = description
        else:
            raw = function_to_tool_definition(func)
            _description = raw["function"]["description"]

        super().__init__(name=_name, description=_description)

    def _generate_definition(self) -> dict[str, Any]:
        defn = function_to_tool_definition(self._func)
        # Override with potentially customised name / description
        defn["function"]["name"] = self.name
        defn["function"]["description"] = self.description
        # Remove 'context' from schema — it is injected internally
        if self._needs_context:
            props = {
                k: v
                for k, v in defn["function"]["parameters"].get("properties", {}).items()
                if k != "context"
            }
            required = [
                r
                for r in defn["function"]["parameters"].get("required", [])
                if r != "context"
            ]
            defn["function"]["parameters"] = {
                "type": "object",
                "properties": props,
                "required": required,
            }
        return defn

    async def execute(self, context: ExecutionContext, **kwargs: Any) -> Any:
        if self._needs_context:
            result = self._func(context=context, **kwargs)
        else:
            result = self._func(**kwargs)
        if inspect.isawaitable(result):
            return await result
        return result


def tool(
    _func: Any = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Any:
    """Decorator that wraps a function as a FunctionTool.

    Works in both forms::

        @tool
        def my_tool(x: str) -> str: ...

        @tool(name="custom", description="...")
        def my_tool(x: str) -> str: ...
    """

    def _wrap(f: Any) -> FunctionTool:
        ft = FunctionTool(f, name=name, description=description)
        functools.update_wrapper(ft, f)
        return ft

    if _func is not None:
        # Called as @tool (no parentheses)
        return _wrap(_func)
    # Called as @tool(...) — return the inner decorator
    return _wrap
