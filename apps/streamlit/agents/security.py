"""
Camada de segurança do Crew — guardrails de entrada, redação de PII e
sanitização de saída.

Implementa as tratativas P1-1, P1-2, P1-3 e P1-4 do relatório de segurança
(`logs/correcoes/2026-06-12-relatorio-seguranca-llm.md`), respondendo ao
requisito do RDD (`logs/ReadmeDrivenDev.md/RDD_RAG.md`) de que **PII/NER e
segredos industriais não podem vazar**.

Três bibliotecas externas, em camadas distintas:

- **LLM Guard** (https://github.com/protectai/llm-guard) — guardrails de
  prompt-injection/jailbreak na ENTRADA e neutralização de exfiltração via
  markdown na SAÍDA.
- **Microsoft Presidio** (https://github.com/microsoft/presidio) — detecção e
  anonimização de PII de PESSOA FÍSICA (CPF, e-mail, telefone, pessoa, etc.)
  na **borda externa**: tudo que sai do perímetro (OpenRouter, Langfuse) é
  anonimizado; a resposta renderizada ao usuário autenticado permanece íntegra
  (decisão de política: redação só na borda externa).
- **detect-secrets** (https://github.com/Yelp/detect-secrets) — detecção de
  CREDENCIAIS TÉCNICAS (chaves de API AWS/GitHub/Slack/Stripe..., chaves
  privadas, JWT) em AMBAS as pontas (entrada bloqueia, saída redige). É o
  mesmo motor do scanner ``Secrets`` do LLM Guard
  (https://protectai.github.io/llm-guard/input_scanners/secrets/), usado
  diretamente (sem o pacote ``llm-guard``) com uma allowlist de plugins que
  exclui os detectores de entropia genérica — ver seção 4 / ``scan_secrets``
  para o porquê.

O scanner de segredos é complementar ao Presidio: Presidio cobre dados
pessoais; ``scan_secrets`` cobre credenciais técnicas — coladas pelo usuário
(ex.: trecho de ``.env``) ou ecoadas de documentos indexados no RAG.

Política de fronteira (decisão registrada com a usuária)
--------------------------------------------------------
```
Usuário (autenticado) ──[PII íntegra]──► vê resposta completa
OpenRouter            ◄─[PII anonimizada]── prompt + contexto
Langfuse (se ativo)   ◄─[PII anonimizada]── trace
```

Dependências obrigatórias (sem bypass silencioso), porém enxutas: modelos
spaCy *small* (Presidio) e ``detect-secrets`` puro (sem LLM Guard/
transformers/sentencepiece) — tudo baseado em regex/heurística, sem modelos
pesados.

Configuração por ambiente
-------------------------
- ``SECURITY_INPUT_GUARD_ENABLED`` (default ``1``) — guardrails de entrada.
- ``SECURITY_OUTPUT_GUARD_ENABLED`` (default ``1``) — sanitização de saída.
- ``SECURITY_PII_REDACTION_ENABLED`` (default ``1``) — anonimização na borda.
- ``SECURITY_SECRETS_GUARD_ENABLED`` (default ``1``) — detecção de segredos
  técnicos (detect-secrets) na entrada (bloqueia) e na saída (redige).
- ``SECURITY_MAX_INPUT_CHARS`` (default ``4000``) — limite da mensagem do usuário.
- ``SECURITY_PII_LANGUAGES`` (default ``pt,en``) — idiomas do Presidio.
- ``SECURITY_ALLOWED_LINK_DOMAINS`` (default vazio) — domínios permitidos em
  links/imagens markdown da saída (qualquer outro é neutralizado).
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache

logger = logging.getLogger(__name__)


# ── Configuração por ambiente ────────────────────────────────────────────────


def _env_flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def input_guard_enabled() -> bool:
    return _env_flag("SECURITY_INPUT_GUARD_ENABLED")


def output_guard_enabled() -> bool:
    return _env_flag("SECURITY_OUTPUT_GUARD_ENABLED")


def pii_redaction_enabled() -> bool:
    return _env_flag("SECURITY_PII_REDACTION_ENABLED")


def max_input_chars() -> int:
    try:
        return int(os.environ.get("SECURITY_MAX_INPUT_CHARS", "4000"))
    except (TypeError, ValueError):
        return 4000


def pii_languages() -> list[str]:
    raw = os.environ.get("SECURITY_PII_LANGUAGES", "pt,en")
    langs = [s.strip().lower() for s in raw.split(",") if s.strip()]
    return langs or ["pt", "en"]


# Entidades que SÃO de fato PII de pessoa física e devem ser anonimizadas na
# borda externa. Deliberadamente NÃO inclui DATE_TIME / ORGANIZATION / LOCATION:
# neste domínio de laboratório esses correspondem a VALIDADE / FABRICANTE / local
# — dado de negócio que o Sintetizador precisa LER e INTERPRETAR (requisito do
# RDD: "o agente sintetizador deverá ler e interpretar os chunks e elaborar uma
# resposta ou relatório usando os dados recebidos"). Redigi-los esvaziaria a
# própria resposta. Configurável via ``SECURITY_PII_ENTITIES``.
_DEFAULT_PII_ENTITIES = (
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "IP_ADDRESS",
    "BR_CPF",
    "BR_CNPJ",
)


def pii_entities() -> list[str]:
    raw = os.environ.get("SECURITY_PII_ENTITIES", "")
    custom = [s.strip().upper() for s in raw.split(",") if s.strip()]
    return custom or list(_DEFAULT_PII_ENTITIES)


def allowed_link_domains() -> set[str]:
    raw = os.environ.get("SECURITY_ALLOWED_LINK_DOMAINS", "")
    return {s.strip().lower() for s in raw.split(",") if s.strip()}


def secrets_guard_enabled() -> bool:
    return _env_flag("SECURITY_SECRETS_GUARD_ENABLED")


# ── Resultados ───────────────────────────────────────────────────────────────


@dataclass
class InputGuardResult:
    """Veredito do guardrail de entrada sobre a mensagem do usuário."""

    allowed: bool
    sanitized_text: str
    reason: str | None = None
    triggered: list[str] = field(default_factory=list)


@dataclass
class PiiResult:
    """Texto anonimizado + contagem de entidades por tipo (para auditoria)."""

    text: str
    entities: dict[str, int] = field(default_factory=dict)

    @property
    def redacted(self) -> bool:
        return bool(self.entities)


@dataclass
class SecretsResult:
    """Texto com segredos técnicos redigidos (chaves de API, tokens, etc.)."""

    text: str
    found: bool = False


# ── 1) Guardrail de entrada (LLM Guard + heurística) ─────────────────────────
#
# Mitiga relatório 1.3 (prompt injection direta), 2.6 (eco de system prompt) e
# 1.5 (mensagens gigantes / abuso de custo).

# Padrões de prompt-injection / extração de system prompt comuns (PT-BR + EN).
# Camada de defesa rápida e auditável que NÃO depende do download de modelos.
_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ignore_previous",
        re.compile(
            r"\b(ignore|desconsidere|esque[çc]a|despreze)\b.{0,40}"
            r"\b(instru[çc][õo]es|prompt|regras|anteriores|acima|system)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "reveal_prompt",
        re.compile(
            r"\b(repita|revele|mostre|imprima|qual\s+[ée]|exiba|print|show|reveal|repeat)\b"
            r".{0,40}\b(system\s*prompt|seu\s+prompt|suas\s+instru[çc][õo]es|"
            r"prompt\s+do\s+sistema|instru[çc][õo]es\s+do\s+sistema)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    # NOTA: removido o padrão "override_role" ("aja como", "assuma o papel"):
    # gerava falso positivo em perguntas compostas legítimas do laboratório
    # (RDD:19-20, ex.: "aja como revisor e compare os protocolos 252 e 253").
    # Os padrões abaixo (jailbreak/DAN/modo desenvolvedor) são sinais mais
    # fortes de ataque e continuam ativos.
    (
        "developer_mode",
        re.compile(
            r"\b(modo\s+desenvolvedor|developer\s+mode|jailbreak|DAN\b|"
            r"sem\s+restri[çc][õo]es|without\s+restrictions)\b",
            re.IGNORECASE,
        ),
    ),
)


@lru_cache(maxsize=1)
def _llm_guard_input_scanners():
    """
    Constrói os scanners de entrada do LLM Guard uma única vez.

    Usamos apenas os scanners *leves* (sem modelos transformers pesados):
    ``PromptInjection`` é opcional e só é adicionado se o modelo estiver
    disponível localmente; ``TokenLimit`` e a heurística de regex cobrem o
    essencial sem custo de download.
    """
    scanners = []
    try:
        from llm_guard.input_scanners import TokenLimit  # type: ignore

        scanners.append(TokenLimit(limit=4096))
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM Guard TokenLimit indisponível: %s", exc)
    return scanners


def scan_user_input(text: str) -> InputGuardResult:
    """
    Aplica os guardrails de entrada à mensagem do usuário ANTES do Triage.

    Combina:
    1. Limite de tamanho (``SECURITY_MAX_INPUT_CHARS``) — mitiga 1.5.
    2. Heurística regex de prompt-injection/extração de prompt — mitiga 1.3/2.6.
    3. Scanners leves do LLM Guard (``TokenLimit``) quando disponíveis.

    Retorna ``allowed=False`` quando a mensagem deve ser recusada; o chamador
    exibe ``reason`` ao usuário sem chamar o LLM (zero tokens gastos).
    """
    original = text or ""
    if not input_guard_enabled():
        return InputGuardResult(allowed=True, sanitized_text=original)

    triggered: list[str] = []
    sanitized = original.strip()

    limit = max_input_chars()
    if len(sanitized) > limit:
        triggered.append("max_length")
        return InputGuardResult(
            allowed=False,
            sanitized_text=sanitized[:limit],
            reason=(
                f"Mensagem muito longa ({len(sanitized)} caracteres). "
                f"O limite é {limit}. Resuma a pergunta ou divida em partes."
            ),
            triggered=triggered,
        )

    for name, pattern in _INJECTION_PATTERNS:
        if pattern.search(sanitized):
            triggered.append(name)

    if triggered:
        return InputGuardResult(
            allowed=False,
            sanitized_text=sanitized,
            reason=(
                "Sua mensagem foi bloqueada por conter um padrão associado a "
                "tentativa de manipulação do assistente (ex.: pedir para ignorar "
                "instruções ou revelar a configuração interna). Reformule a "
                "pergunta focando nos dados do laboratório."
            ),
            triggered=triggered,
        )

    # Scanners leves do LLM Guard como segunda camada (não bloqueiam por padrão
    # além do TokenLimit; mantêm o ponto de extensão para scanners futuros).
    for scanner in _llm_guard_input_scanners():
        try:
            sanitized, is_valid, _risk = scanner.scan(sanitized)
            if not is_valid:
                triggered.append(type(scanner).__name__)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Scanner de entrada %s falhou: %s", type(scanner).__name__, exc)

    if triggered:
        return InputGuardResult(
            allowed=False,
            sanitized_text=sanitized,
            reason=(
                "Sua mensagem excede os limites de segurança configurados. "
                "Reduza o tamanho e reformule a pergunta."
            ),
            triggered=triggered,
        )

    # Segredos técnicos (chaves de API, tokens, chaves privadas...) coladas por
    # engano: bloqueia ANTES do Triage para o segredo nunca saber ao LLM remoto.
    secrets_result = scan_secrets(sanitized)
    if secrets_result.found:
        return InputGuardResult(
            allowed=False,
            sanitized_text=secrets_result.text,
            reason=(
                "Sua mensagem foi bloqueada porque parece conter um segredo "
                "técnico (ex.: chave de API, token ou chave privada). Remova "
                "o segredo e reenvie a pergunta."
            ),
            triggered=["secrets"],
        )

    return InputGuardResult(allowed=True, sanitized_text=sanitized)


# ── 2) Redação de PII na borda externa (Presidio) ────────────────────────────
#
# Mitiga relatório 1.4 (dado confidencial → OpenRouter) e 2.4 (saída → Langfuse).

# Reconhecedores PT-BR adicionais que o Presidio não traz por padrão.
_BR_CPF = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")
_BR_CNPJ = re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b")


@lru_cache(maxsize=1)
def _presidio_engines():
    """
    Inicializa (uma vez) o ``AnalyzerEngine`` + ``AnonymizerEngine`` do Presidio
    com suporte multilíngue (PT + EN) usando modelos spaCy *small*.

    Levanta ``RuntimeError`` se o Presidio/modelos não estiverem instalados —
    dependência obrigatória, sem bypass silencioso.
    """
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, PatternRecognizer, Pattern
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine

    langs = pii_languages()
    spacy_by_lang = {
        "pt": "pt_core_news_sm",
        "en": "en_core_web_sm",
    }
    models = [
        {"lang_code": lang, "model_name": spacy_by_lang.get(lang, "en_core_web_sm")}
        for lang in langs
    ]
    provider = NlpEngineProvider(
        nlp_configuration={"nlp_engine_name": "spacy", "models": models}
    )
    nlp_engine = provider.create_engine()

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(
        nlp_engine=nlp_engine, languages=langs
    )

    # Reconhecedores brasileiros (CPF/CNPJ) para cada idioma suportado.
    for lang in langs:
        registry.add_recognizer(
            PatternRecognizer(
                supported_entity="BR_CPF",
                name="br_cpf_recognizer",
                supported_language=lang,
                patterns=[Pattern(name="cpf", regex=_BR_CPF.pattern, score=0.8)],
            )
        )
        registry.add_recognizer(
            PatternRecognizer(
                supported_entity="BR_CNPJ",
                name="br_cnpj_recognizer",
                supported_language=lang,
                patterns=[Pattern(name="cnpj", regex=_BR_CNPJ.pattern, score=0.8)],
            )
        )

    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        registry=registry,
        supported_languages=langs,
    )
    anonymizer = AnonymizerEngine()
    return analyzer, anonymizer


# Linhas que carregam a FONTE da evidência (rastreabilidade — RDD:13). Devem
# passar íntegras pela anonimização: o nome do arquivo/projeto é o que garante
# a auditabilidade e frequentemente contém datas/nomes que o Presidio
# classificaria como PII (ex.: ``protocolo_Dra_Silva_2024-03.docx``).
_SOURCE_LINE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*###\s*Evidência\s*\[\d+\]"),          # cabeçalho RAG
    re.compile(r"\[Projeto:\s*[^\]]+\]\s*\[Arquivo:"),     # prefixo embutido RAG
    re.compile(r"_project_id|_source_file|_sheet_name"),    # colunas de fonte OLAP
)


def _is_source_line(line: str) -> bool:
    return any(p.search(line) for p in _SOURCE_LINE_PATTERNS)


def anonymize_pii(
    text: str,
    *,
    language: str | None = None,
    preserve_source_lines: bool = False,
) -> PiiResult:
    """
    Anonimiza PII em ``text`` substituindo cada entidade por ``<TIPO>``.

    Apenas as entidades em ``pii_entities()`` (PII de pessoa física: CPF, CNPJ,
    e-mail, telefone, cartão, PERSON…) são redigidas. ``DATE_TIME``,
    ``ORGANIZATION`` e ``LOCATION`` são **preservados** — neste domínio eles são
    validade/fabricante/local, dado que o Sintetizador precisa ler e interpretar.

    Usado SOMENTE na borda externa (antes de enviar a OpenRouter/Langfuse).
    A resposta entregue ao usuário autenticado **não** passa por aqui.

    ``preserve_source_lines`` (decisão de projeto): quando ``True``, as linhas
    que carregam a fonte da evidência (``### Evidência [N] — Projeto: … ·
    Arquivo: …``, o prefixo ``[Projeto: …] [Arquivo: …]`` e as colunas de fonte
    do OLAP) passam **íntegras**, preservando a rastreabilidade exigida pelo RDD
    (`RDD_RAG.md:13`). O corpo do documento continua sendo anonimizado.

    Quando a redação está desligada (``SECURITY_PII_REDACTION_ENABLED=0``),
    devolve o texto original sem alteração.
    """
    raw = text or ""
    if not pii_redaction_enabled() or not raw.strip():
        return PiiResult(text=raw)

    lang = (language or pii_languages()[0]).lower()
    try:
        analyzer, anonymizer = _presidio_engines()
    except Exception as exc:  # noqa: BLE001
        # Dependência obrigatória: falha de inicialização é um erro real, não um
        # bypass. Logamos e devolvemos o texto original para não derrubar o chat,
        # mas o aviso indica que a proteção não está ativa.
        logger.error("Presidio indisponível — PII NÃO redigida: %s", exc)
        return PiiResult(text=raw)

    entities: dict[str, int] = {}
    allow = pii_entities()

    def _anon_segment(segment: str) -> str:
        if not segment.strip():
            return segment
        # Restringe aos tipos de PII de pessoa física (não redige
        # DATE_TIME/ORGANIZATION/LOCATION, que aqui são validade/fabricante).
        results = analyzer.analyze(text=segment, language=lang, entities=allow)
        if not results:
            return segment
        for r in results:
            entities[r.entity_type] = entities.get(r.entity_type, 0) + 1
        return anonymizer.anonymize(text=segment, analyzer_results=results).text

    if preserve_source_lines and "\n" in raw:
        # Anonimização linha-a-linha: preserva as linhas de fonte verbatim e
        # anonimiza apenas o corpo. Mantém os offsets do Presidio corretos
        # (cada linha é analisada isoladamente).
        out_lines = [
            line if _is_source_line(line) else _anon_segment(line)
            for line in raw.split("\n")
        ]
        return PiiResult(text="\n".join(out_lines), entities=entities)

    return PiiResult(text=_anon_segment(raw), entities=entities)


def anonymize_messages_for_external(
    messages: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """
    Anonimiza PII em todas as ``messages`` (system + histórico + atual) antes do
    envio ao provedor LLM remoto. Devolve a lista anonimizada e o agregado de
    entidades encontradas (para o trace de auditoria).
    """
    if not pii_redaction_enabled():
        return messages, {}

    out: list[dict[str, str]] = []
    totals: dict[str, int] = {}
    for m in messages:
        content = str(m.get("content") or "")
        # Preserva linhas de fonte (Projeto:/Arquivo:/colunas OLAP) para não
        # quebrar a citação/rastreabilidade do RDD; o corpo é anonimizado.
        res = anonymize_pii(content, preserve_source_lines=True)
        for k, v in res.entities.items():
            totals[k] = totals.get(k, 0) + v
        out.append({**m, "content": res.text})
    return out, totals


# ── 3) Sanitização de saída (LLM Guard + anti-exfiltração markdown) ───────────
#
# Mitiga relatório 2.3 (exfiltração/clickjacking via markdown) e reforça 1.2.

# Links/imagens markdown: ![alt](url) e [texto](url).
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\(\s*([^)\s]+)[^)]*\)")
_MD_LINK = re.compile(r"(?<!!)\[([^\]]*)\]\(\s*([^)\s]+)[^)]*\)")
# URLs cruas com esquema http(s).
_RAW_URL = re.compile(r"https?://[^\s)>\]]+", re.IGNORECASE)
_HTML_TAG = re.compile(r"<\s*/?\s*(script|img|iframe|svg|a|object|embed|link|style)\b[^>]*>", re.IGNORECASE)


def _domain_of(url: str) -> str:
    m = re.match(r"https?://([^/]+)/?", url.strip(), re.IGNORECASE)
    return (m.group(1).lower() if m else "").split(":")[0]


@dataclass
class OutputGuardResult:
    text: str
    neutralized: list[str] = field(default_factory=list)


def sanitize_model_output(text: str) -> OutputGuardResult:
    """
    Neutraliza vetores de exfiltração na resposta do LLM ANTES de ``st.markdown``.

    Defesa em profundidade contra o cenário do relatório 2.3: um documento
    malicioso induz o LLM a emitir ``![x](http://evil/leak?d=<dado>)``; quando o
    navegador renderiza a imagem, o dado vaza na query string.

    Ações:
    - Imagens markdown para domínio não permitido → viram texto inerte.
    - Links markdown para domínio não permitido → preservam o rótulo, removem o
      destino (vira ``texto (link removido por segurança)``).
    - Tags HTML ativas (script/img/iframe/svg/a/...) → escapadas.

    Domínios em ``SECURITY_ALLOWED_LINK_DOMAINS`` são preservados.
    """
    raw = text or ""
    if not output_guard_enabled() or not raw:
        return OutputGuardResult(text=raw)

    allowed = allowed_link_domains()
    neutralized: list[str] = []

    def _img_repl(match: re.Match[str]) -> str:
        alt, url = match.group(1), match.group(2)
        if _domain_of(url) in allowed:
            return match.group(0)
        neutralized.append(f"image:{_domain_of(url) or url}")
        return f"`[imagem removida por segurança: {alt or 'sem descrição'}]`"

    def _link_repl(match: re.Match[str]) -> str:
        label, url = match.group(1), match.group(2)
        if _domain_of(url) in allowed:
            return match.group(0)
        neutralized.append(f"link:{_domain_of(url) or url}")
        return f"{label} (link removido por segurança)"

    out = _MD_IMAGE.sub(_img_repl, raw)
    out = _MD_LINK.sub(_link_repl, out)

    # Tags HTML ativas: escapa o '<' para que o Streamlit não interprete.
    def _tag_repl(match: re.Match[str]) -> str:
        neutralized.append(f"html:{match.group(1).lower()}")
        return match.group(0).replace("<", "&lt;").replace(">", "&gt;")

    out = _HTML_TAG.sub(_tag_repl, out)

    # Segunda camada opcional via LLM Guard (output scanners leves), se presente.
    for scanner in _llm_guard_output_scanners():
        try:
            out, is_valid, _risk = scanner.scan(raw, out)
            if not is_valid:
                neutralized.append(type(scanner).__name__)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Scanner de saída %s falhou: %s", type(scanner).__name__, exc)

    # Segredos técnicos ecoados de documentos indexados (ex.: chave de API
    # esquecida num .docx/.xlsx): redige antes de renderizar, sem bloquear a
    # resposta inteira.
    secrets_result = scan_secrets(out)
    if secrets_result.found:
        neutralized.append("secrets")
        out = secrets_result.text

    return OutputGuardResult(text=out, neutralized=neutralized)


@lru_cache(maxsize=1)
def _llm_guard_output_scanners():
    """Scanners de saída leves do LLM Guard (sem modelos pesados)."""
    scanners = []
    try:
        from llm_guard.output_scanners import MaliciousURLs  # type: ignore

        # MaliciousURLs sem modelo remoto: depende do modelo local; se indisponível
        # cai no except e seguimos só com a heurística regex acima.
        scanners.append(MaliciousURLs())
    except Exception as exc:  # noqa: BLE001
        logger.info("LLM Guard MaliciousURLs indisponível (usando heurística regex): %s", exc)
    return scanners


# ── 4) Detecção de segredos técnicos (detect-secrets) ────────────────────────
#
# Complementa a redação de PII (Presidio, seção 2): enquanto o Presidio cobre
# dados de PESSOA FÍSICA (CPF, e-mail, telefone...), esta seção cobre
# CREDENCIAIS TÉCNICAS — chaves de API (AWS, GitHub, Slack, Stripe...), chaves
# privadas e JWT — via ``detect-secrets`` (Yelp), o mesmo motor usado pelo
# scanner ``Secrets`` do LLM Guard
# (https://protectai.github.io/llm-guard/input_scanners/secrets/). Usamos
# ``detect-secrets`` DIRETO em vez do pacote ``llm-guard``: o construtor do
# scanner ``Secrets`` do LLM Guard não permite customizar os plugins, e sua
# config padrão inclui ``Base64HighEntropyString``/``HexHighEntropyString``,
# que classificam palavras comuns em PT-BR ("ELISA", "lote", "Chikungunya")
# como "segredo de alta entropia" — bloquearia toda mensagem normal do chat.
#
# Allowlist de plugins (``_SECRETS_PLUGINS``): todos os detectores de PADRÃO
# específico do ``detect-secrets`` (prefixo/formato conhecido de credencial),
# SEM os dois detectores de entropia genérica.
#
# Aplicado em DUAS pontas:
# - ENTRADA (``scan_user_input``): se o usuário colar um segredo (ex.: por
#   engano, um trecho de ``.env``), a mensagem é bloqueada ANTES do Triage —
#   o segredo nunca chega a sair para o LLM remoto.
# - SAÍDA (``sanitize_model_output``): se um documento indexado no RAG contiver
#   um segredo esquecido e o LLM o ecoar na resposta, o trecho é redigido antes
#   de ``st.markdown``.

_SECRETS_PLUGINS: tuple[dict[str, str], ...] = (
    {"name": "ArtifactoryDetector"},
    {"name": "AWSKeyDetector"},
    {"name": "AzureStorageKeyDetector"},
    {"name": "BasicAuthDetector"},
    {"name": "CloudantDetector"},
    {"name": "DiscordBotTokenDetector"},
    {"name": "GitHubTokenDetector"},
    {"name": "IbmCloudIamDetector"},
    {"name": "IbmCosHmacDetector"},
    {"name": "JwtTokenDetector"},
    {"name": "MailchimpDetector"},
    {"name": "NpmDetector"},
    {"name": "PrivateKeyDetector"},
    {"name": "SendGridDetector"},
    {"name": "SlackDetector"},
    {"name": "SoftlayerDetector"},
    {"name": "SquareOAuthDetector"},
    {"name": "StripeDetector"},
    {"name": "TwilioKeyDetector"},
    {"name": "KeywordDetector"},
)


def _redact_partial(value: str) -> str:
    """Mostra só os 2 primeiros/últimos caracteres — mesma convenção do LLM Guard ``REDACT_PARTIAL``."""
    if len(value) <= 4:
        return "****"
    return f"{value[:2]}..{value[-2:]}"


def scan_secrets(text: str) -> SecretsResult:
    """
    Detecta e redige segredos técnicos (chaves de API, tokens, chaves
    privadas, JWT...) em ``text`` usando ``detect-secrets`` com
    ``_SECRETS_PLUGINS`` (sem detectores de entropia genérica).

    Quando ``found=True``, ``text`` já vem com os segredos parcialmente
    mascarados (2 primeiros/últimos caracteres) — o chamador decide se
    bloqueia a mensagem (entrada) ou apenas usa o texto redigido (saída).

    Se ``detect-secrets`` estiver indisponível (dependência não instalada) ou
    ``SECURITY_SECRETS_GUARD_ENABLED=0``, devolve o texto original sem
    alteração — registrando um aviso no log.
    """
    raw = text or ""
    if not secrets_guard_enabled() or not raw.strip():
        return SecretsResult(text=raw)

    try:
        from detect_secrets.core.scan import scan_line
        from detect_secrets.settings import transient_settings
    except Exception as exc:  # noqa: BLE001
        logger.error("detect-secrets indisponível — segredos NÃO verificados: %s", exc)
        return SecretsResult(text=raw)

    try:
        out_lines: list[str] = []
        found = False
        with transient_settings({"plugins_used": list(_SECRETS_PLUGINS)}):
            for line in raw.split("\n"):
                results = list(scan_line(line)) if line.strip() else []
                for result in results:
                    if not result.secret_value:
                        continue
                    found = True
                    line = line.replace(result.secret_value, _redact_partial(result.secret_value))
                out_lines.append(line)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Scanner de segredos falhou: %s", exc)
        return SecretsResult(text=raw)

    return SecretsResult(text="\n".join(out_lines), found=found)
