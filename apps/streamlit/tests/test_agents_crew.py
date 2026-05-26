"""Testes do sistema multiagentes (CrewAI local) — sem dependência do CrewAI em si.

Os testes mockam o cliente OpenAI e os subsistemas (RAG/OLAP/ML) para validar:

- Greeter rule-based curto-circuita saudações.
- Triage classifica corretamente e respeita disponibilidade.
- Dispatcher executa as Tools certas em paralelo/serial.
- Synthesizer concatena contextos sem duplicar.
- Trace registra cada step com tempo.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.crew import CrewContext, dispatch_specialists, run_triage  # noqa: E402
from agents.greeter import handle_greeting, is_social_only  # noqa: E402
from agents.handoff import HandoffStep, HandoffTrace  # noqa: E402
from agents.synthesizer import build_messages  # noqa: E402
from agents.tools import ToolResult  # noqa: E402
from agents.triage import TriageDecision, classify_intent  # noqa: E402


class GreeterTests(unittest.TestCase):
    def test_recognizes_greeting(self) -> None:
        self.assertTrue(is_social_only("Olá!"))
        self.assertTrue(is_social_only("Bom dia"))
        self.assertTrue(is_social_only("obrigado"))

    def test_rejects_substantive_message(self) -> None:
        self.assertFalse(is_social_only("Qual a validade do reagente X?"))
        self.assertFalse(is_social_only("Faça uma predição"))

    def test_handle_returns_text_for_greeting(self) -> None:
        out = handle_greeting("Olá!")
        self.assertIsNotNone(out)
        assert out is not None  # type: ignore[truthy-bool]
        self.assertIn("assistente", out.lower())

    def test_handle_returns_thanks_response(self) -> None:
        out = handle_greeting("obrigado")
        self.assertIsNotNone(out)
        assert out is not None
        self.assertIn("nada", out.lower())

    def test_handle_returns_none_for_real_question(self) -> None:
        self.assertIsNone(handle_greeting("Qual o lote do anticorpo X?"))


class HandoffTraceTests(unittest.TestCase):
    def test_step_records_elapsed(self) -> None:
        trace = HandoffTrace()
        with trace.start("test", input_summary="abc") as h:
            h.set_output("done")
            h.set_metadata(rows=3)
        self.assertEqual(trace.step_count, 1)
        step = trace.steps[0]
        self.assertEqual(step.name, "test")
        self.assertEqual(step.input_summary, "abc")
        self.assertEqual(step.output_summary, "done")
        self.assertTrue(step.ok)
        self.assertEqual(step.metadata, {"rows": 3})
        self.assertGreaterEqual(step.elapsed_ms, 0.0)

    def test_step_marks_failed_on_exception(self) -> None:
        trace = HandoffTrace()
        with self.assertRaises(ValueError):
            with trace.start("test") as h:
                h.set_output("partial")
                raise ValueError("boom")
        self.assertEqual(trace.step_count, 1)
        self.assertFalse(trace.steps[0].ok)
        self.assertIn("boom", trace.steps[0].note)

    def test_summarize_truncates_long_text(self) -> None:
        long_input = "x" * 1000
        trace = HandoffTrace()
        with trace.start("test", input_summary=long_input):
            pass
        self.assertLessEqual(len(trace.steps[0].input_summary), 241)
        self.assertTrue(trace.steps[0].input_summary.endswith("…"))


class TriageTests(unittest.TestCase):
    def test_social_message_short_circuits(self) -> None:
        d = classify_intent("Olá!", documents_available=True, ml_available=True)
        self.assertEqual(d.source, "social")
        self.assertFalse(d.use_rag)
        self.assertFalse(d.use_olap)
        self.assertFalse(d.use_ml)

    @patch("agents.triage._classify_with_llm")
    def test_uses_llm_when_client_provided(self, mock_llm: MagicMock) -> None:
        mock_llm.return_value = TriageDecision(
            use_rag=True, use_olap=False, use_ml=False, source="llm", reason="test"
        )
        d = classify_intent(
            "Qual a validade do lote 3?",
            client=MagicMock(),
            model="m",
            documents_available=True,
        )
        self.assertEqual(d.source, "llm")
        self.assertTrue(d.use_rag)

    @patch("agents.triage._classify_with_llm")
    def test_ml_hint_overrides_rag(self, mock_llm: MagicMock) -> None:
        mock_llm.return_value = TriageDecision(
            use_rag=True, use_olap=False, use_ml=False, source="llm", reason=""
        )
        d = classify_intent(
            "Faça a predição de log_Aff para Agtype SARS-CoV-2",
            client=MagicMock(),
            model="m",
            documents_available=True,
            ml_available=True,
        )
        self.assertTrue(d.use_ml)
        self.assertFalse(d.use_rag)
        self.assertIn("ml_hint", d.source)

    @patch("agents.triage._classify_with_llm", return_value=None)
    def test_falls_back_to_rules(self, _llm: MagicMock) -> None:
        d = classify_intent(
            "Quantos registros por projeto na planilha?",
            client=MagicMock(),
            model="m",
            spreadsheets_available=True,
        )
        self.assertEqual(d.source, "rules_fallback")
        self.assertTrue(d.use_olap)

    @patch("agents.triage._classify_with_llm")
    def test_respects_unavailable_subsystems(self, mock_llm: MagicMock) -> None:
        mock_llm.return_value = TriageDecision(
            use_rag=True, use_olap=True, use_ml=True, source="llm", reason=""
        )
        d = classify_intent(
            "consulta",
            client=MagicMock(),
            model="m",
            documents_available=False,
            spreadsheets_available=False,
            ml_available=False,
        )
        self.assertFalse(d.use_rag)
        self.assertFalse(d.use_olap)
        self.assertFalse(d.use_ml)


class DispatchSpecialistsTests(unittest.TestCase):
    def _ctx(self, **overrides) -> CrewContext:
        defaults = dict(
            user_message="x",
            history=[],
            client=MagicMock(),
            model="m",
            rag_backend=MagicMock(),
            rag_top_k=4,
            rag_project_ids=None,
            ml_bundle=None,
            ml_model_path=None,
            documents_available=True,
            spreadsheets_available=True,
            ml_available=True,
            parallel_tools=False,
        )
        defaults.update(overrides)
        return CrewContext(**defaults)

    @patch("agents.crew.rag_search_tool")
    def test_runs_only_rag(self, mock_rag: MagicMock) -> None:
        mock_rag.return_value = ToolResult(name="rag", ok=True, summary="3 hits")
        ctx = self._ctx()
        triage = MagicMock()
        triage.decision = TriageDecision(
            use_rag=True, use_olap=False, use_ml=False, source="llm", reason=""
        )
        trace = HandoffTrace()
        out = dispatch_specialists(ctx, triage, trace=trace)
        self.assertEqual(set(out.keys()), {"rag"})
        mock_rag.assert_called_once()

    @patch("agents.crew.ml_predict_tool")
    @patch("agents.crew.duckdb_query_tool")
    @patch("agents.crew.rag_search_tool")
    def test_ml_excludes_rag_olap(
        self,
        mock_rag: MagicMock,
        mock_olap: MagicMock,
        mock_ml: MagicMock,
    ) -> None:
        mock_ml.return_value = ToolResult(name="ml", ok=True, summary="1 pred")
        ctx = self._ctx()
        triage = MagicMock()
        triage.decision = TriageDecision(
            use_rag=True, use_olap=True, use_ml=True, source="llm", reason=""
        )
        trace = HandoffTrace()
        out = dispatch_specialists(ctx, triage, trace=trace)
        self.assertEqual(set(out.keys()), {"ml"})
        mock_rag.assert_not_called()
        mock_olap.assert_not_called()
        mock_ml.assert_called_once()

    @patch("agents.crew.duckdb_query_tool")
    @patch("agents.crew.rag_search_tool")
    def test_runs_rag_and_olap_in_parallel(
        self,
        mock_rag: MagicMock,
        mock_olap: MagicMock,
    ) -> None:
        mock_rag.return_value = ToolResult(name="rag", ok=True, summary="hits")
        mock_olap.return_value = ToolResult(name="olap", ok=True, summary="rows")
        ctx = self._ctx(parallel_tools=True)
        triage = MagicMock()
        triage.decision = TriageDecision(
            use_rag=True, use_olap=True, use_ml=False, source="llm", reason=""
        )
        trace = HandoffTrace()
        out = dispatch_specialists(ctx, triage, trace=trace)
        self.assertEqual(set(out.keys()), {"rag", "olap"})

    @patch("agents.crew.rag_search_tool")
    def test_tool_exception_is_captured(self, mock_rag: MagicMock) -> None:
        mock_rag.side_effect = RuntimeError("backend offline")
        ctx = self._ctx(parallel_tools=False)
        triage = MagicMock()
        triage.decision = TriageDecision(
            use_rag=True, use_olap=False, use_ml=False, source="llm", reason=""
        )
        trace = HandoffTrace()
        out = dispatch_specialists(ctx, triage, trace=trace)
        self.assertFalse(out["rag"].ok)
        self.assertIn("RuntimeError", out["rag"].error or "")


class BuildMessagesTests(unittest.TestCase):
    def test_basic_history_passthrough(self) -> None:
        si = build_messages(
            user_message="Qual a validade?",
            history=[{"role": "user", "content": "Olá"}, {"role": "assistant", "content": "Oi"}],
            tool_results={},
            model_id="qwen3.5-9b-mtp",
        )
        self.assertEqual(si.messages[0]["role"], "system")
        self.assertEqual(len(si.messages), 3)
        self.assertFalse(si.used_ml)

    def test_ml_result_changes_system_prompt(self) -> None:
        ml_ok = ToolResult(
            name="ml",
            ok=True,
            context_for_llm="### Resultado da inferência ML\nlog_Aff=8.1",
        )
        si = build_messages(
            user_message="Predição",
            history=[],
            tool_results={"ml": ml_ok},
            model_id="qwen3.5-9b-mtp",
        )
        self.assertTrue(si.used_ml)
        self.assertIn("predição", si.system_prompt.lower())
        self.assertIn("log_Aff=8.1", si.system_prompt)

    def test_rag_and_olap_both_appended(self) -> None:
        rag_ok = ToolResult(
            name="rag",
            ok=True,
            context_for_llm="### Contexto RAG\nDoc 1",
        )
        olap_ok = ToolResult(
            name="olap",
            ok=True,
            context_for_llm="### Consulta OLAP\nRow 1",
        )
        si = build_messages(
            user_message="combinada",
            history=[],
            tool_results={"rag": rag_ok, "olap": olap_ok},
            model_id="any",
        )
        self.assertIn("Doc 1", si.system_prompt)
        self.assertIn("Row 1", si.system_prompt)
        # Garante ordem RAG → OLAP no prompt final.
        self.assertLess(
            si.system_prompt.index("Doc 1"),
            si.system_prompt.index("Row 1"),
        )

    def test_failed_rag_includes_explanation(self) -> None:
        rag_fail = ToolResult(
            name="rag",
            ok=False,
            error="Índice ainda não foi construído.",
        )
        si = build_messages(
            user_message="Q",
            history=[],
            tool_results={"rag": rag_fail},
            model_id="any",
        )
        self.assertIn("RAG (falha)", si.system_prompt)
        self.assertIn("não foi construído", si.system_prompt)


if __name__ == "__main__":
    unittest.main()
