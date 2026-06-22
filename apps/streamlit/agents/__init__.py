"""
Sistema multiagentes do chat — orquestração via CrewAI rodando local.

Veja ``AGENTS.md`` neste pacote para a arquitetura, agentes, tools e fluxo
de handoff. Este ``__init__`` apenas reexporta as APIs públicas usadas pelo
``app.py``.
"""

from __future__ import annotations

from .handoff import HandoffStep, HandoffTrace
from .runner import (
    CrewRunResult,
    parallel_tools_enabled,
    run_crew_chat,
    trace_handoff_enabled,
)
from .security import (
    InputGuardResult,
    OutputGuardResult,
    anonymize_messages_for_external,
    anonymize_pii,
    sanitize_model_output,
    scan_user_input,
)
from .tools import ToolResult

__all__ = [
    "CrewRunResult",
    "HandoffStep",
    "HandoffTrace",
    "InputGuardResult",
    "OutputGuardResult",
    "ToolResult",
    "anonymize_messages_for_external",
    "anonymize_pii",
    "parallel_tools_enabled",
    "run_crew_chat",
    "sanitize_model_output",
    "scan_user_input",
    "trace_handoff_enabled",
]
