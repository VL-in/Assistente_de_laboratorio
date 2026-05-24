"""
Embeddings ESM-2 (8M) para colunas de sequência AbRank + compactação PCA.

Fluxo: 3 colunas de sequência → vetores 320-d cada → concat 960-d → PCA → ``seq_pca_*``.
O transformador é serializado no ``ModelBundle`` para inferência idêntica no chat e na UI.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
import warnings
from typing import Any, Callable

# (mensagem, fração 0–1 ou None)
EmbeddingProgressFn = Callable[[str, float | None], None]

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

from ml.paths import esm_cache_root

# facebook/esm2_t6_8M_UR50D — hidden_size=320 (modelo pequeno 8M parâmetros).
DEFAULT_ESM_MODEL_ID = "facebook/esm2_t6_8M_UR50D"
DEFAULT_SEQUENCE_COLUMNS = (
    "Ab_heavy_chain_seq",
    "Ab_light_chain_seq",
    "Ag_seq",
)
DEFAULT_MAX_SEQ_LENGTH = 1022
DEFAULT_PCA_VARIANCE = 0.95
DEFAULT_PCA_MAX_COMPONENTS = 64
DEFAULT_PCA_MIN_COMPONENTS = 8
DEFAULT_EMBED_BATCH_SIZE = 16

_AA_CLEAN_RE = re.compile(r"[^A-Za-z*]")


def _disable_esm_token_dropout(model: Any) -> None:
    """
    Desativa token_dropout na inferência.

    O ESM-2 foi treinado com essa regularização; em embedding para ML não é necessária.
    Em várias versões do ``transformers`` (4.x), sem ``attention_mask`` na camada de
    embeddings isso gera tensores com comprimentos diferentes (ex.: 463 vs 246).
    """
    if hasattr(model, "config") and hasattr(model.config, "token_dropout"):
        model.config.token_dropout = False
    embeddings = getattr(model, "embeddings", None)
    if embeddings is not None and hasattr(embeddings, "token_dropout"):
        embeddings.token_dropout = False
    esm = getattr(model, "esm", None)
    if esm is not None:
        inner = getattr(esm, "embeddings", None)
        if inner is not None and hasattr(inner, "token_dropout"):
            inner.token_dropout = False


def _mean_pool_sequence_embeddings(hidden: Any, attention_mask: Any) -> Any:
    """Mean pooling alinhado ao comprimento real de ``last_hidden_state``."""
    import torch

    seq_len = hidden.size(1)
    mask = attention_mask[:, :seq_len].unsqueeze(-1).to(dtype=hidden.dtype)
    summed = torch.sum(hidden * mask, dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def sequence_column_names() -> tuple[str, ...]:
    return DEFAULT_SEQUENCE_COLUMNS


def esm_available() -> tuple[bool, str]:
    try:
        import torch  # noqa: F401
        from transformers import AutoModel, AutoTokenizer  # noqa: F401

        return True, ""
    except ImportError:
        return (
            False,
            "Pacotes ausentes para ESM-2. Instale: pip install torch transformers",
        )


def clean_protein_sequence(raw: object) -> str:
    """Normaliza sequência (remove espaços, FASTA header, caracteres inválidos)."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return ""
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "null", ""}:
        return ""
    # Remove header FASTA se presente
    if text.startswith(">"):
        lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith(">")]
        text = "".join(lines)
    text = text.replace(" ", "").replace("\n", "").replace("\r", "")
    return _AA_CLEAN_RE.sub("", text).upper()


@dataclass
class SequenceEmbeddingConfig:
    model_id: str = DEFAULT_ESM_MODEL_ID
    sequence_columns: tuple[str, ...] = DEFAULT_SEQUENCE_COLUMNS
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH
    pca_variance: float = DEFAULT_PCA_VARIANCE
    pca_max_components: int = DEFAULT_PCA_MAX_COMPONENTS
    pca_min_components: int = DEFAULT_PCA_MIN_COMPONENTS
    embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE
    random_state: int = 42


class _EsmEmbedder:
    """Carrega ESM-2 uma vez (cache de processo) e gera embeddings mean-pooled."""

    _instance: "_EsmEmbedder | None" = None

    def __init__(self, model_id: str, max_length: int) -> None:
        self.model_id = model_id
        self.max_length = max_length
        self._model = None
        self._tokenizer = None
        self._device = None
        self._warned_batch_fallback = False

    @classmethod
    def shared(cls, model_id: str, max_length: int) -> "_EsmEmbedder":
        if cls._instance is None or cls._instance.model_id != model_id:
            cls._instance = cls(model_id, max_length)
        return cls._instance

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        ok, msg = esm_available()
        if not ok:
            raise RuntimeError(msg)
        import torch
        from transformers import AutoModel, AutoTokenizer

        cache = str(esm_cache_root())
        os.environ.setdefault("HF_HOME", cache)
        os.environ.setdefault("TRANSFORMERS_CACHE", cache)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir=cache)
        self._model = AutoModel.from_pretrained(self.model_id, cache_dir=cache)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.eval()
        _disable_esm_token_dropout(self._model)

    @property
    def hidden_size(self) -> int:
        self._ensure_loaded()
        return int(self._model.config.hidden_size)

    def embed_batch(self, sequences: list[str]) -> np.ndarray:
        """Retorna matriz (n, hidden_size) com mean pooling por sequência."""
        self._ensure_loaded()
        import torch

        cleaned = [clean_protein_sequence(s) for s in sequences]
        out = np.zeros((len(cleaned), self.hidden_size), dtype=np.float32)
        valid_idx: list[int] = []
        valid_seqs: list[str] = []
        for i, seq in enumerate(cleaned):
            if seq:
                valid_idx.append(i)
                valid_seqs.append(seq)

        if not valid_seqs:
            return out

        try:
            pooled = self._forward_embed_batch(valid_seqs)
        except RuntimeError as exc:
            if "size of tensor" not in str(exc).lower():
                raise
            if not self._warned_batch_fallback:
                warnings.warn(
                    "ESM-2: processando sequências uma a uma (mais lento). "
                    "Atualize transformers>=4.48 ou reconstrua a imagem Docker.",
                    stacklevel=2,
                )
                self._warned_batch_fallback = True
            chunks = [self._forward_embed_batch([seq]) for seq in valid_seqs]
            pooled = np.vstack(chunks)

        for row_i, emb_i in zip(valid_idx, pooled, strict=True):
            out[row_i] = emb_i.astype(np.float32)
        return out

    def _forward_embed_batch(self, sequences: list[str]) -> np.ndarray:
        import torch

        encoded = self._tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {k: v.to(self._device) for k, v in encoded.items()}
        with torch.no_grad():
            hidden = self._model(**encoded).last_hidden_state
            pooled = _mean_pool_sequence_embeddings(hidden, encoded["attention_mask"])
        return pooled.cpu().numpy()


class SequenceEmbeddingTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible: ``fit`` ajusta PCA; ``transform`` adiciona colunas ``seq_pca_*``.

    Entrada: DataFrame com colunas de sequência (+ opcionalmente outras colunas ignoradas).
    """

    def __init__(self, config: SequenceEmbeddingConfig | None = None) -> None:
        self.config = config or SequenceEmbeddingConfig()
        self.pca_: PCA | None = None
        self.pca_columns_: list[str] = []
        self.mean_embeddings_: dict[str, np.ndarray] = {}
        self.n_concat_dims_: int = 0
        self.fitted_: bool = False
        self._train_pca_matrix_: np.ndarray | None = None

    def _active_sequence_columns(self, df: pd.DataFrame) -> list[str]:
        return [c for c in self.config.sequence_columns if c in df.columns]

    def _concat_embeddings(
        self,
        df: pd.DataFrame,
        *,
        progress: EmbeddingProgressFn | None = None,
        progress_base: float = 0.0,
        progress_span: float = 1.0,
        phase_label: str = "ESM-2",
    ) -> np.ndarray:
        cols = self._active_sequence_columns(df)
        if not cols:
            raise ValueError("Nenhuma coluna de sequência encontrada no DataFrame.")
        embedder = _EsmEmbedder.shared(self.config.model_id, self.config.max_seq_length)
        hidden = embedder.hidden_size
        n = len(df)
        batch_size = max(1, self.config.embed_batch_size)
        total_batches = sum((n + batch_size - 1) // batch_size for _ in cols)
        parts: list[np.ndarray] = []
        done_batches = 0
        for col in cols:
            col_emb = np.zeros((n, hidden), dtype=np.float32)
            values = df[col].tolist()
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                col_emb[start:end] = embedder.embed_batch(values[start:end])
                done_batches += 1
                if progress is not None and total_batches > 0:
                    frac = progress_base + progress_span * (done_batches / total_batches)
                    progress(
                        f"{phase_label}: «{col}» — lote {done_batches}/{total_batches} "
                        f"({end}/{n} linhas)",
                        frac,
                    )
            parts.append(col_emb)
        return np.hstack(parts)

    def fit(
        self,
        X: pd.DataFrame,
        y: Any = None,
        *,
        progress: EmbeddingProgressFn | None = None,
    ) -> "SequenceEmbeddingTransformer":
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        cols = self._active_sequence_columns(df)
        if not cols:
            raise ValueError(
                f"Dataset sem colunas de sequência esperadas: {list(self.config.sequence_columns)}"
            )
        embedder = _EsmEmbedder.shared(self.config.model_id, self.config.max_seq_length)
        hidden = embedder.hidden_size
        self.n_concat_dims_ = hidden * len(cols)

        if progress is not None:
            progress(f"ESM-2: gerando embeddings de treino ({len(df)} linhas)…", 0.05)
        matrix = self._concat_embeddings(
            df,
            progress=progress,
            progress_base=0.05,
            progress_span=0.45,
            phase_label="ESM-2 (treino)",
        )
        for i, col in enumerate(cols):
            slot = matrix[:, i * hidden : (i + 1) * hidden]
            nonempty = np.any(slot != 0, axis=1)
            if nonempty.any():
                self.mean_embeddings_[col] = slot[nonempty].mean(axis=0).astype(np.float32)
            else:
                self.mean_embeddings_[col] = np.zeros(hidden, dtype=np.float32)

        n_samples, n_features = matrix.shape
        if n_samples < 2:
            self.pca_ = PCA(n_components=1, random_state=self.config.random_state)
            self.pca_.fit(matrix)
            self.pca_columns_ = [f"seq_pca_{i}" for i in range(self.pca_.n_components_)]
            self.fitted_ = True
            return self

        max_comp = min(
            self.config.pca_max_components,
            n_samples - 1 if n_samples > 1 else 1,
            n_features,
        )
        min_comp = min(self.config.pca_min_components, max_comp)
        self.pca_ = PCA(
            n_components=self.config.pca_variance,
            random_state=self.config.random_state,
        )
        self.pca_.fit(matrix)
        n_comp = self.pca_.n_components_
        if n_comp < min_comp:
            self.pca_ = PCA(n_components=min_comp, random_state=self.config.random_state)
            self.pca_.fit(matrix)
        elif n_comp > max_comp:
            self.pca_ = PCA(n_components=max_comp, random_state=self.config.random_state)
            self.pca_.fit(matrix)

        self.pca_columns_ = [f"seq_pca_{i}" for i in range(self.pca_.n_components_)]
        self._train_pca_matrix_ = self.pca_.transform(matrix).astype(np.float32)
        self.fitted_ = True
        if progress is not None:
            progress(
                f"ESM-2: PCA concluída — {len(self.pca_columns_)} componentes (`seq_pca_*`).",
                0.52,
            )
        return self

    def training_pca_frame(self) -> pd.DataFrame:
        """PCA do conjunto usado em ``fit`` (evita re-embedar as mesmas linhas de treino)."""
        if not self.fitted_ or self._train_pca_matrix_ is None:
            raise RuntimeError(
                "PCA de treino indisponível: chame fit() antes ou o cache já foi liberado."
            )
        return pd.DataFrame(self._train_pca_matrix_, columns=self.pca_columns_)

    def clear_training_cache(self) -> None:
        """Remove matriz PCA de treino da memória (não serializar no .pkl)."""
        self._train_pca_matrix_ = None

    def _impute_missing_slots(self, df: pd.DataFrame, matrix: np.ndarray) -> np.ndarray:
        embedder = _EsmEmbedder.shared(self.config.model_id, self.config.max_seq_length)
        hidden = embedder.hidden_size
        cols = self._active_sequence_columns(df)
        out = matrix.copy()
        for i, col in enumerate(cols):
            slot = out[:, i * hidden : (i + 1) * hidden]
            empty_rows = ~np.any(slot != 0, axis=1)
            if empty_rows.any() and col in self.mean_embeddings_:
                slot[empty_rows] = self.mean_embeddings_[col]
                out[:, i * hidden : (i + 1) * hidden] = slot
        return out

    def transform(
        self,
        X: pd.DataFrame,
        *,
        progress: EmbeddingProgressFn | None = None,
    ) -> pd.DataFrame:
        if not self.fitted_ or self.pca_ is None:
            raise RuntimeError("SequenceEmbeddingTransformer não foi ajustado (fit).")
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if progress is not None:
            progress(f"ESM-2: gerando embeddings ({len(df)} linhas)…", 0.55)
        matrix = self._concat_embeddings(
            df,
            progress=progress,
            progress_base=0.55,
            progress_span=0.2,
            phase_label="ESM-2",
        )
        matrix = self._impute_missing_slots(df, matrix)
        reduced = self.pca_.transform(matrix)
        pca_df = pd.DataFrame(reduced, columns=self.pca_columns_, index=df.index)
        return pd.concat([df, pca_df], axis=1)

    def transform_pca_only(
        self,
        X: pd.DataFrame,
        *,
        progress: EmbeddingProgressFn | None = None,
    ) -> pd.DataFrame:
        """Retorna só as colunas PCA (útil para montar matriz de features)."""
        full = self.transform(X, progress=progress)
        return full[self.pca_columns_]

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config)


def prefetch_esm_model(model_id: str = DEFAULT_ESM_MODEL_ID) -> str:
    """Baixa pesos do ESM-2 para o cache persistente (Docker: ``/data/ml/huggingface``)."""
    embedder = _EsmEmbedder.shared(model_id, DEFAULT_MAX_SEQ_LENGTH)
    embedder._ensure_loaded()
    return model_id


def apply_sequence_embeddings(
    df: pd.DataFrame,
    transformer: SequenceEmbeddingTransformer,
) -> pd.DataFrame:
    """Atalho: transforma DataFrame adicionando colunas ``seq_pca_*``."""
    return transformer.transform(df)


def extract_sequences_from_text(message: str) -> dict[str, str]:
    """
    Extrai sequências da mensagem do chat (atribuição ``col = SEQ`` ou bloco FASTA).
    """
    found: dict[str, str] = {}
    col_map = {c.lower(): c for c in DEFAULT_SEQUENCE_COLUMNS}

    for col_key, canonical in col_map.items():
        escaped = re.escape(canonical)
        patterns = [
            rf"{escaped}\s*[:=]\s*([A-Za-z*\n\r>]+?)(?=\s*(?:{escaped}|Ab_|Ag_|$|\n\n))",
            rf"{escaped}\s*[:=]\s*([A-Za-z*]{{20,}})",
        ]
        for pat in patterns:
            m = re.search(pat, message, re.IGNORECASE | re.DOTALL)
            if m:
                seq = clean_protein_sequence(m.group(1))
                if len(seq) >= 10:
                    found[canonical] = seq
                    break

    fasta_blocks = re.findall(
        r">?\s*(Ab_heavy|Ab_light|Ag|heavy|light|antigen)[^\n]*\n([A-Za-z*\n\r]+)",
        message,
        re.IGNORECASE,
    )
    alias = {
        "ab_heavy": "Ab_heavy_chain_seq",
        "heavy": "Ab_heavy_chain_seq",
        "ab_light": "Ab_light_chain_seq",
        "light": "Ab_light_chain_seq",
        "ag": "Ag_seq",
        "antigen": "Ag_seq",
    }
    for label, body in fasta_blocks:
        key = alias.get(label.lower().replace(" ", "_"))
        if key and key not in found:
            seq = clean_protein_sequence(body)
            if len(seq) >= 10:
                found[key] = seq
    return found
