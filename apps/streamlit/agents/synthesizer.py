"""
Synthesizer Agent — agente final do Crew, gera a resposta para o usuário.

Não é um ``crewai.Agent`` formal porque já chamamos o cliente OpenAI direto
para preservar streaming, perfis Qwen3.5 e ``iter_stream_answer_text``. Dessa
forma o Synthesizer integra-se ao ``st.write_stream`` sem fricção.

A função ``build_messages`` monta o ``messages`` do chat completions e o
``system_prompt`` final concatenando os blocos de contexto vindos das Tools.
"""

from __future__ import annotations

from dataclasses import dataclass

from agents.tools import ToolResult
from qwen35_inference import sanitize_history_message

# System prompts mantidos compatíveis com ``app.py`` para preservar tom e
# constraints já validados em campo. Quando há predição ML, o prompt do
# AssistenteML toma o lugar do principal (é mais focado).
CHAT_SYSTEM_PROMPT = (
    """<role>
Você é um assistente de laboratório experiente que atua em pesquisa e desenvolvimento de imunodiagnósticos, principalmente ELISA. Você passou muitos anos trabalhando com dados laboratoriais, planejamento de ensaios, interpretação de resultados. Já viu inúmeros erros por desatenção e sabe que documentação e rastreabilidade de informação é crucial em projetos de desenvolvimento. Você entende que existem múltiplos documentos de experimentos que representam linhas de raciocínio contínua, tratando, por tanto, os documentos não só isoladamente, mas uma sequência.
</role>

<context>
Estamos trabalhando em um laboratório de p&D que está com projeto de ELISA ativa. Os pesquisadores planejam e documentam por meio de arquivos docx dados como materiais e insumos, lotes e validades. Os documentos possuem padrões, e os insumos se apresentam na ordem de "nome"/"Fabricante ou código"/ "Lote" ou "Ativo" do equipamento/"Validade". Os pesquisadores vão vir até você para fazer perguntas sobre o que foi feito ou usado nos experimentos passados. O seu trabalho é identificar o que o usuário está buscando e, através dos resultados, apresentar as informações relevantes. Perceba que a mesma informação pode aparecer em diferentes documentos, que podem compor a resposta retornada.
</context>

<constraints>
- Nunca invente dados, busque por retrieval as respostas quando a pergunta é voltada para os ensaios.
- Analise todos os chunks e entenda que a resposta pode ser composta por dados de diferentes documentos.
- Não altere nenhum dado dos documentos.
- Retorne as informações junto ao título do documento e a sua data de planejamento.
- Ao final de cada resposta, seja cordial e pergunte se pode ajudar com mais alguma dúvida.
- Faça sempre uma pergunta de cada vez.
- Caso não tenha encontrado respostas nos documentos, expresse isso educadamente.
- Quando houver bloco de predição ML no contexto, explique o resultado com base somente nesses números; não invente predições fora do que foi calculado.
</constraints>

<goals>
- Identifique o objetivo do usuário.
- Se o usuário perguntar sobre nome de insumo, fabricante, lote ou validade, lembre que os dados estão sempre descritos nessa ordem.
- Sintetizar uma resposta objetiva que contenha o dado referente às perguntas feitas, sempre referenciando o documento.
</goals>

<invocation>
Sempre use a mesma língua do usuário. Por padrão, utilize português brasileiro. Seja cordial, profissional, objetivo e educado.
</invocation>"""
)

CHAT_ML_SYSTEM_PROMPT = (
    """<role>
Você é um assistente de laboratório que interpreta predições de um modelo de ML já treinado (afinidade Ab–Ag, benchmark AbRank).
</role>

<constraints>
- A predição numérica JÁ FOI CALCULADA localmente e está no bloco "Resultado da inferência ML" abaixo.
- Use SOMENTE os valores dessa tabela na resposta. Não diga que falta acesso a documentos ou planilhas para esta pergunta.
- Não peça ao usuário para buscar documentos quando a tabela de predição estiver presente.
- Se o bloco indicar falha ou dados insuficientes, explique o que falta (nomes exatos das colunas) sem inventar números.
- Seja objetivo: 2–3 parágrafos curtos com o valor de `predicao_log_aff` (ou `predicao`) e interpretação breve.
- Ao final, pergunte se pode ajudar com mais alguma dúvida.
</constraints>

<invocation>
Responda em português brasileiro, cordial e profissional.
</invocation>"""
)


@dataclass
class SynthesizerInput:
    """Pacote pronto para o Synthesizer enviar ao OpenRouter."""

    system_prompt: str
    messages: list[dict[str, str]]
    used_ml: bool


def build_messages(
    *,
    user_message: str,
    history: list[dict],
    tool_results: dict[str, ToolResult],
    model_id: str,
    max_history_turns: int | None = None,
    max_chars_per_message: int | None = None,
) -> SynthesizerInput:
    """
    Monta ``messages`` para ``chat.completions.create`` no Synthesizer.

    Convenções:
    - Quando ``tool_results['ml']`` está presente e ``ok``, o prompt-base passa
      a ser ``CHAT_ML_SYSTEM_PROMPT`` (foco em interpretar a predição).
    - Demais ``context_for_llm`` das tools são concatenados na ordem RAG → OLAP
      → ML (compatibilidade com o pipeline atual de ``app.py``).
    - Mensagens do histórico do assistant passam por ``sanitize_history_message``
      para remover blocos ``<think>`` antes de irem para a API.

    O ``user_message`` é gravado em ``messages`` apenas como o último turno; o
    chamador deve já ter incluído essa mensagem no ``history`` se quiser que
    apareça no rerun do Streamlit.

    Truncagem (opcional, single-source-of-truth):
    - ``max_history_turns``: limite em pares user+assistant. ``None`` = sem
      limite; ``0`` = histórico vazio (apenas o ``system_prompt``).
    - ``max_chars_per_message``: corte por mensagem (com ``…`` no fim). ``None``
      = não corta. Útil na rota ML para evitar inflar o prompt com respostas
      longas anteriores.
    """
    used_ml = "ml" in tool_results and tool_results["ml"].ok

    base_prompt = CHAT_ML_SYSTEM_PROMPT if used_ml else CHAT_SYSTEM_PROMPT
    blocks: list[str] = [base_prompt]

    rag_result = tool_results.get("rag")
    if rag_result is not None:
        if rag_result.ok and rag_result.context_for_llm:
            blocks.append(rag_result.context_for_llm)
        elif rag_result.ok and not rag_result.context_for_llm:
            blocks.append(
                "(Nenhum trecho relevante foi recuperado do índice para esta pergunta — "
                "não invente dados de ensaios.)"
            )
        elif rag_result.error:
            blocks.append(
                "### Contexto RAG (falha)\n"
                f"{rag_result.error}\nExplique a limitação ao usuário sem inventar."
            )

    olap_result = tool_results.get("olap")
    if olap_result is not None:
        if olap_result.ok and olap_result.context_for_llm:
            blocks.append(olap_result.context_for_llm)
        elif olap_result.error:
            blocks.append(
                "### Dados tabulares (falha na consulta)\n"
                f"{olap_result.error}\n"
                "Explique o problema ao usuário sem inventar números de planilhas."
            )

    ml_result = tool_results.get("ml")
    if ml_result is not None:
        if ml_result.ok and ml_result.context_for_llm:
            blocks.append(ml_result.context_for_llm)
        elif ml_result.error:
            blocks.append(
                ml_result.context_for_llm
                or f"### Predição ML (falha)\n{ml_result.error}"
            )

    system_prompt = "\n\n".join(blocks)

    trimmed_history = _trim_history(
        history,
        max_history_turns=max_history_turns,
        max_chars_per_message=max_chars_per_message,
    )

    api_messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for m in trimmed_history:
        role = m.get("role", "user")
        content = str(m.get("content") or "")
        api_messages.append(
            {
                "role": role,
                "content": sanitize_history_message(role, content, model_id=model_id),
            }
        )

    return SynthesizerInput(
        system_prompt=system_prompt,
        messages=api_messages,
        used_ml=used_ml,
    )


def _trim_history(
    messages: list[dict],
    *,
    max_history_turns: int | None,
    max_chars_per_message: int | None,
) -> list[dict]:
    """
    Aplica ``max_history_turns`` (pares user+assistant) e ``max_chars_per_message``.

    Quando ambos são ``None``, retorna a lista original sem cópia. Quando
    ``max_history_turns == 0``, retorna lista vazia (apenas system_prompt
    será enviado).
    """
    if max_history_turns is None and max_chars_per_message is None:
        return messages
    if max_history_turns is not None and max_history_turns <= 0:
        return []
    if max_history_turns is not None:
        max_msgs = max_history_turns * 2
        trimmed = messages if len(messages) <= max_msgs else messages[-max_msgs:]
    else:
        trimmed = messages
    if not max_chars_per_message or max_chars_per_message <= 0:
        return list(trimmed)
    out: list[dict] = []
    for m in trimmed:
        role = m.get("role", "user")
        content = str(m.get("content") or "")
        if len(content) > max_chars_per_message:
            content = content[:max_chars_per_message] + "…"
        out.append({"role": role, "content": content})
    return out
