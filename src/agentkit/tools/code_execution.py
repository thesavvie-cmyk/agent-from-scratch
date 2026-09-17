"""Python code execution tool via e2b sandbox (block 17).

The tool returns a JSON object so the model can distinguish between stdout,
a return value, and a traceback — each carrying different semantics.
A configuration error (no sandbox) is raised as an exception so it surfaces
immediately to the developer; an execution error (bad code) is returned in
the JSON so the model can see the traceback and fix the code.
"""
from __future__ import annotations

import json

from agentkit.context import ExecutionContext
from agentkit.tools.base import tool


@tool
async def execute_python(context: ExecutionContext, code: str) -> str:
    """Execute Python code in an isolated sandbox and return the result.

    Use this tool for:
    - Arithmetic or mathematical computations where precision matters
      (e.g. large numbers, floating-point calculations, statistics)
    - Data processing: sorting, filtering, transforming lists or dicts
    - String parsing: dates, regex, structured text formats
    - Any multi-step computation where intermediate state must persist
      between calls in the same run (variables survive across calls)

    Do NOT use this tool for:
    - Web searches or HTTP requests (use search_web instead)
    - File I/O on the host system (files written here are sandbox-local)
    - Tasks answerable directly from knowledge without computation

    Parameters
    ----------
    code : str
        Valid Python code. The value of the last expression is captured
        as "result". Use print() for intermediate output (captured as
        "stdout"). Imports are allowed; standard library is available.

    Returns (JSON object)
    ----------------------
    {
      "stdout":  ["line1", ...],   -- output from print() calls
      "result":  "...",            -- text of the last evaluated expression
                                     (null if the last statement has no value)
      "error":   null              -- null on success, or:
                 {
                   "type":      "ExceptionType",
                   "message":   "...",
                   "traceback": "full traceback string"
                 }
    }

    On error the sandbox is NOT killed — you can fix the code and call
    execute_python again in the same run; prior variables are still in scope.
    """
    if context.code_env is None:
        raise RuntimeError(
            "execute_python: no sandbox available. "
            "Initialize Agent with code_execution='e2b'."
        )

    execution = await context.code_env.run_code(code)

    stdout_lines: list[str] = execution.logs.stdout if execution.logs else []

    main_result: str | None = None
    for r in (execution.results or []):
        if r.is_main_result and r.text is not None:
            main_result = r.text
            break

    error_payload: dict | None = None
    if execution.error:
        error_payload = {
            "type": execution.error.name,
            "message": execution.error.value,
            "traceback": execution.error.traceback,
        }

    return json.dumps(
        {"stdout": stdout_lines, "result": main_result, "error": error_payload},
        ensure_ascii=False,
    )
