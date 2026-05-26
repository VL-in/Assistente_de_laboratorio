"""
Trilha de handoff entre agentes/tools — modo aprendizado e auditoria.

Cada etapa do Crew (Greeter, Triage, Tools, Synthesizer) registra um
``HandoffStep`` com entrada/saída resumidas, tempo decorrido e metadados.
A UI mostra a trilha em um expander quando o usuário ativa o toggle.

A coleta tem custo desprezível (cópia de strings + ``time.perf_counter``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# Tamanho máximo dos resumos de input/output mostrados na UI.
# 240 chars já mostra contexto sem inflar a interface ao ler 50 evidências.
DEFAULT_SUMMARY_CHARS = 240


def _summarize(value: Any, *, max_chars: int = DEFAULT_SUMMARY_CHARS) -> str:
    """Converte qualquer valor em string curta para exibição na trilha."""
    if value is None:
        return ""
    text = str(value).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


@dataclass
class HandoffStep:
    """Uma etapa registrada na trilha do Crew."""

    name: str
    started_at: str
    elapsed_ms: float = 0.0
    input_summary: str = ""
    output_summary: str = ""
    ok: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "started_at": self.started_at,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "ok": self.ok,
            "metadata": dict(self.metadata),
            "note": self.note,
        }


@dataclass
class HandoffTrace:
    """Coleção ordenada de ``HandoffStep`` produzida em uma execução do Crew."""

    steps: list[HandoffStep] = field(default_factory=list)

    def append(self, step: HandoffStep) -> None:
        self.steps.append(step)

    def start(self, name: str, *, input_summary: str = "") -> "_StepHandle":
        """
        Abre um step e retorna um handle. Uso:

        >>> with trace.start("rag_tool", input_summary="busca documentos") as h:
        ...     ...
        ...     h.set_output("3 evidências encontradas")
        ...     h.set_metadata(top_k=6)
        """
        return _StepHandle(self, name=name, input_summary=_summarize(input_summary))

    def to_list(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self.steps]

    @property
    def total_elapsed_ms(self) -> float:
        return round(sum(s.elapsed_ms for s in self.steps), 2)

    @property
    def step_count(self) -> int:
        return len(self.steps)


class _StepHandle:
    """Context manager interno que registra ``HandoffStep`` ao sair do bloco."""

    def __init__(self, trace: HandoffTrace, *, name: str, input_summary: str) -> None:
        self._trace = trace
        self._name = name
        self._input = input_summary
        self._output = ""
        self._metadata: dict[str, Any] = {}
        self._note = ""
        self._ok = True
        self._start: float = 0.0
        self._started_iso = ""

    def __enter__(self) -> "_StepHandle":
        self._start = time.perf_counter()
        self._started_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        if exc_type is not None:
            self._ok = False
            self._note = f"{exc_type.__name__}: {exc}"
            self._output = self._output or _summarize(exc)
        self._trace.append(
            HandoffStep(
                name=self._name,
                started_at=self._started_iso,
                elapsed_ms=elapsed_ms,
                input_summary=self._input,
                output_summary=self._output,
                ok=self._ok,
                metadata=dict(self._metadata),
                note=self._note,
            )
        )
        return False

    def set_output(self, value: Any) -> None:
        self._output = _summarize(value)

    def set_metadata(self, **kwargs: Any) -> None:
        self._metadata.update(kwargs)

    def mark_failed(self, note: str) -> None:
        self._ok = False
        self._note = note
