"""
Download e carga de datasets Kaggle via ``kagglehub``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

KAGGLE_ABRANK_HANDLE = "aurlienplissier/abrank"
DEFAULT_ABRANK_SPLIT = "AbRank_dataset.csv"
ABRANK_BENCHMARK_REGRESSION_SPLIT = "Benchmarks/train_regression.csv"


def kaggle_cache_root() -> Path:
    """Cache do kagglehub (``KAGGLEHUB_CACHE`` ou padrão do usuário)."""
    raw = os.environ.get("KAGGLEHUB_CACHE", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / ".cache" / "kagglehub").resolve()


def download_kaggle_dataset(
    handle: str,
    *,
    force_download: bool = False,
    output_dir: str | Path | None = None,
) -> Path:
    """
    Baixa (ou reutiliza cache) um dataset Kaggle.

    Requer credenciais Kaggle fora do ambiente Kaggle (``KAGGLE_API_TOKEN``,
    ``~/.kaggle/kaggle.json`` ou ``kagglehub.login()``).
    """
    import kagglehub

    kwargs: dict = {"force_download": force_download}
    if output_dir is not None:
        kwargs["output_dir"] = str(output_dir)
    path = kagglehub.dataset_download(handle, **kwargs)
    return Path(path).resolve()


def load_kaggle_csv(
    handle: str,
    relative_file: str,
    *,
    separator: str = ",",
    max_rows: int | None = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Carrega um arquivo tabular do dataset via cache local.

    Usa ``kagglehub.dataset_load`` (adapter Pandas) quando disponível; caso
    contrário lê do path retornado por ``dataset_download``.
    """
    pandas_kwargs: dict = {"sep": separator, "low_memory": False}
    if max_rows is not None and max_rows > 0:
        pandas_kwargs["nrows"] = max_rows

    adapter_error: Exception | None = None
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter

        return kagglehub.dataset_load(
            KaggleDatasetAdapter.PANDAS,
            handle,
            relative_file,
            pandas_kwargs=pandas_kwargs,
        )
    except ImportError:
        pass
    except Exception as exc:
        adapter_error = exc

    root = download_kaggle_dataset(handle, force_download=force_download)
    file_path = root / relative_file
    if not file_path.is_file():
        if adapter_error is not None:
            hint = (
                " Verifique KAGGLE_API_TOKEN ou ~/.kaggle/kaggle.json."
                if "401" in str(adapter_error).lower()
                or "credential" in str(adapter_error).lower()
                else ""
            )
            raise RuntimeError(
                f"Falha ao carregar `{relative_file}` do Kaggle: {adapter_error}.{hint}"
            ) from adapter_error
        raise FileNotFoundError(
            f"Arquivo `{relative_file}` não encontrado em `{root}`. "
            "Verifique o handle Kaggle e o caminho no catálogo YAML."
        )
    return pd.read_csv(file_path, **pandas_kwargs)


def load_abrank_split(
    split_file: str = DEFAULT_ABRANK_SPLIT,
    *,
    max_rows: int | None = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """Carrega um split tabular do AbRank (TSV por padrão)."""
    return load_kaggle_csv(
        KAGGLE_ABRANK_HANDLE,
        split_file,
        separator="\t",
        max_rows=max_rows,
        force_download=force_download,
    )
