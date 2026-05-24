"""
Aba Streamlit — ML tradicional (FLAML + exportação .pkl + predição).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import PurePath

import pandas as pd
import streamlit as st

from ml.datasets import (
    default_feature_columns,
    load_dataset_from_catalog,
    prepare_feature_matrix,
)
from ml.dictionary import DatasetCatalog, load_dataset_catalog
from ml.kaggle_sources import KAGGLE_ABRANK_HANDLE, kaggle_cache_root
from ml.paths import ensure_ml_models_root
from ml.predict import (
    DEFAULT_MERGE_KEY,
    merge_with_amostras_if_needed,
    predict_from_bundle,
    read_prediction_table,
    validate_prediction_columns,
)
from ml.training import (
    EstimatorSummary,
    FlamlTrainConfig,
    MINIMAL_ESTIMATOR_LIST,
    TrainReport,
    bundle_metadata_json,
    estimator_display_label,
    flaml_available,
    load_model_bundle,
    save_model_bundle,
    train_flaml_model,
)

SESSION_DF = "ml_df"
SESSION_CATALOG = "ml_catalog"
SESSION_BUNDLE = "ml_model_bundle"
SESSION_REPORT = "ml_train_report"
SESSION_CATALOG_ID = "ml_catalog_id"
ABRANK_CATALOG_ID = "abrank_kaggle"


def _resolve_catalog(catalog_id: str = ABRANK_CATALOG_ID) -> DatasetCatalog:
    """Catálogo alinhado aos dados carregados ou ao YAML do picker."""
    loaded = st.session_state.get(SESSION_CATALOG)
    active_id = st.session_state.get(SESSION_CATALOG_ID)
    if loaded and active_id == catalog_id and getattr(loaded, "dataset_id", None) == catalog_id:
        return loaded
    return load_dataset_catalog(catalog_id)


def _load_catalog_dataset(
    catalog_id: str,
    *,
    max_rows: int | None,
    force_download: bool,
) -> None:
    catalog = load_dataset_catalog(catalog_id)
    with st.spinner("Baixando/carregando dataset…"):
        df, catalog = load_dataset_from_catalog(
            catalog,
            max_rows=max_rows if max_rows and max_rows > 0 else None,
            force_download=force_download,
        )
    st.session_state[SESSION_DF] = df
    st.session_state[SESSION_CATALOG] = catalog
    st.session_state[SESSION_CATALOG_ID] = catalog_id
    if catalog.is_kaggle:
        st.session_state["ml_kaggle_handle"] = catalog.kaggle_handle


def _section_dados() -> None:
    st.subheader("1. Dataset e dicionário de colunas")
    catalog_id = ABRANK_CATALOG_ID
    catalog = load_dataset_catalog(catalog_id)

    st.caption(
        f"Fonte Kaggle: `{catalog.kaggle_handle}` · arquivo `{catalog.kaggle_split_file}`. "
        f"Cache: `{kaggle_cache_root()}`."
    )
    with st.expander("Autenticação Kaggle (primeira vez)"):
        st.markdown(
            "Fora do Kaggle Notebooks, defina **`KAGGLE_API_TOKEN`** no `.env` "
            "(token em [kaggle.com/settings/api](https://www.kaggle.com/settings/api)) "
            "ou use `~/.kaggle/kaggle.json`. No Docker, monte o token ou passe a variável no Compose."
        )

    c1, c2, c3 = st.columns([1, 1, 1])
    max_rows = c1.number_input(
        "Máx. linhas (0 = todas)",
        min_value=0,
        value=15000,
        step=1000,
        key="ml_max_rows",
        help="AbRank tem centenas de milhares de linhas; use amostra para treinar mais rápido.",
    )
    force_dl = c2.checkbox("Forçar novo download", value=False, key="ml_force_kaggle_dl")
    with c3:
        label = f"Carregar {catalog.display_name}"
        if st.button(label, type="primary", key="ml_load_catalog"):
            try:
                _load_catalog_dataset(
                    catalog_id,
                    max_rows=int(max_rows),
                    force_download=force_dl,
                )
                st.success(f"{len(st.session_state[SESSION_DF])} linha(s) carregada(s).")
            except Exception as exc:
                st.error(str(exc))

    uploaded = st.file_uploader(
        "Ou envie CSV/Excel (opcional)",
        type=["csv", "xlsx", "xlsm"],
        key="ml_upload_dataset",
    )
    if uploaded is not None:
        try:
            df = read_prediction_table(uploaded)
            st.session_state[SESSION_DF] = df
            st.session_state[SESSION_CATALOG] = catalog
            st.session_state[SESSION_CATALOG_ID] = catalog_id
            st.info(f"Arquivo `{uploaded.name}` — {len(df)} linha(s).")
        except ValueError as exc:
            st.error(str(exc))

    df: pd.DataFrame | None = st.session_state.get(SESSION_DF)
    if df is None:
        st.info("Selecione o dataset e clique em **Carregar** para começar.")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Linhas", len(df))
    m2.metric("Colunas", len(df.columns))
    m3.metric("Tarefa sugerida", catalog.ml_task)
    target = catalog.default_target
    if target in df.columns and catalog.ml_task == "classification":
        m4.metric("Classes (alvo)", df[target].nunique(dropna=True))
    elif target in df.columns:
        m4.metric(f"Média {target}", f"{pd.to_numeric(df[target], errors='coerce').mean():.2f}")

    with st.expander("Prévia dos dados", expanded=False):
        st.dataframe(df.head(20), use_container_width=True, hide_index=True)

    with st.expander("Dicionário de colunas", expanded=True):
        dict_df = pd.DataFrame(catalog.column_dict_rows())
        dict_df["presente_no_dataset"] = dict_df["coluna"].isin(df.columns)
        st.dataframe(dict_df, use_container_width=True, hide_index=True)

    suggested = [c for c in catalog.suggested_drop if c in df.columns]
    st.markdown("**Colunas sugeridas para remover:** " + (", ".join(suggested) or "—"))

    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0].sort_values(ascending=False)
    if not nulls.empty:
        with st.expander("Valores ausentes por coluna"):
            st.dataframe(nulls.rename("n_ausentes"), use_container_width=True)


def _metric_higher_is_better(metric: str) -> bool:
    return metric.lower() in {"r2", "accuracy", "roc_auc", "f1", "micro_f1", "macro_f1", "ap"}


def _format_cv_score(metric: str, value: float | None) -> str:
    if value is None:
        return "—"
    if metric.lower() in {"r2", "f1", "accuracy", "roc_auc"}:
        return f"{value:.4f}"
    return f"{value:.4f}"


def _estimator_summaries_from_report(report: TrainReport) -> list[EstimatorSummary]:
    rows = getattr(report, "estimator_summaries", None) or []
    if rows and isinstance(rows[0], dict):
        return [EstimatorSummary(**row) for row in rows]
    return list(rows)


def _render_train_results(report: TrainReport) -> None:
    """Tabela e gráfico com dados, tentativas e métricas por estimador FLAML."""
    st.markdown("**Comparativo de estimadores (validação FLAML)**")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Linhas (após limpeza)", report.n_total)
    d2.metric("Treino", report.n_train)
    d3.metric("Teste (holdout)", report.n_test)
    d4.metric("Tempo de busca (s)", f"{report.train_seconds:.1f}")

    summaries = _estimator_summaries_from_report(report)
    if not summaries:
        st.caption("Resumo por estimador indisponível para este treino.")
        return

    metric_label = report.metric.upper()
    higher_better = _metric_higher_is_better(report.metric)
    rows = []
    for row in summaries:
        rows.append(
            {
                "Estimador": row.label,
                "Tentativas": row.n_trials,
                "Linhas no treino": row.n_samples,
                f"{metric_label} (CV)": _format_cv_score(report.metric, row.cv_score),
                "Melhor": "✓" if row.is_best else "",
            }
        )
    table = pd.DataFrame(rows)
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Melhor": st.column_config.TextColumn("Melhor", help="Estimador escolhido pelo FLAML."),
        },
    )

    best = next((r for r in summaries if r.is_best), summaries[0])
    best_score_txt = _format_cv_score(report.metric, best.cv_score)
    st.success(
        f"**Melhor modelo:** {estimator_display_label(report.best_estimator)} · "
        f"{metric_label} (CV) = {best_score_txt} · "
        f"{best.n_trials} tentativa(s) · {best.n_samples} linha(s) de treino"
    )

    chart_rows = [
        {"estimador": r.label, report.metric: r.cv_score}
        for r in summaries
        if r.cv_score is not None
    ]
    if chart_rows:
        chart_df = pd.DataFrame(chart_rows).set_index("estimador")
        st.caption(
            f"Gráfico: {metric_label} na validação cruzada "
            f"({'maior' if higher_better else 'menor'} é melhor)."
        )
        st.bar_chart(chart_df, height=260)

    st.markdown("**Métricas no conjunto de teste (holdout)**")
    if report.task == "regression":
        t1, t2, t3 = st.columns(3)
        t1.metric("R²", f"{report.r2:.3f}" if report.r2 is not None else "—")
        t2.metric("RMSE", f"{report.rmse:.3f}" if report.rmse is not None else "—")
        t3.metric(
            f"{metric_label} (CV) — vencedor",
            _format_cv_score(report.metric, report.best_cv_score),
        )
    else:
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("F1", f"{report.f1:.3f}" if report.f1 is not None else "—")
        t2.metric("Acurácia", f"{report.accuracy:.3f}" if report.accuracy is not None else "—")
        t3.metric(
            f"{metric_label} (CV) — vencedor",
            _format_cv_score(report.metric, report.best_cv_score),
        )
        if report.roc_auc is not None:
            t4.metric("ROC-AUC", f"{report.roc_auc:.3f}")


def _section_treino() -> None:
    st.subheader("2. Treino AutoML (FLAML)")
    ok, flaml_msg = flaml_available()
    if not ok:
        st.warning(flaml_msg)
        return

    df: pd.DataFrame | None = st.session_state.get(SESSION_DF)
    if df is None:
        st.info("Carregue o dataset na seção 1 antes de treinar.")
        return

    catalog = _resolve_catalog()
    task = st.radio(
        "Tipo de problema",
        options=["regression", "classification"],
        index=0 if catalog.ml_task == "regression" else 1,
        horizontal=True,
        key="ml_task",
    )
    if task != catalog.ml_task:
        st.caption(
            f"O catálogo sugere **{catalog.ml_task}**; você selecionou **{task}**. "
            "Confirme se o alvo faz sentido para o problema."
        )

    target_options = [catalog.default_target] if catalog.default_target in df.columns else list(df.columns)
    target_column = st.selectbox(
        "Coluna-alvo",
        options=target_options if target_options else list(df.columns),
        index=0,
        key="ml_target_col",
        help=catalog.description_for(catalog.default_target),
    )

    default_features = default_feature_columns(df, catalog)
    feature_columns = st.multiselect(
        "Features",
        options=sorted(df.columns),
        default=default_features,
        key="ml_feature_cols",
    )

    default_metric = catalog.default_metric or ("r2" if task == "regression" else "f1")
    metric_options = (
        ["r2", "rmse", "mse", "mae"] if task == "regression" else ["f1", "accuracy", "roc_auc", "log_loss"]
    )
    metric_idx = metric_options.index(default_metric) if default_metric in metric_options else 0

    with st.expander("Hiperparâmetros FLAML", expanded=True):
        c1, c2, c3 = st.columns(3)
        time_budget = c1.number_input("Tempo máximo (s)", 10, 3600, 120, 10, key="ml_time_budget")
        metric = c2.selectbox("Métrica", metric_options, index=metric_idx, key="ml_metric")
        test_size = c3.slider("Proporção teste", 0.1, 0.4, 0.2, 0.05, key="ml_test_size")
        c4, c5, c6 = st.columns(3)
        n_splits = c4.number_input("Folds (CV)", 2, 10, 5, key="ml_n_splits")
        seed = c5.number_input("Seed", 0, value=42, key="ml_seed")
        eval_method = c6.selectbox("Avaliação", ["cv", "holdout"], key="ml_eval_method")
        estimator_list = st.multiselect(
            "Estimadores",
            list(MINIMAL_ESTIMATOR_LIST),
            default=list(MINIMAL_ESTIMATOR_LIST),
            key="ml_estimators",
        )

    if st.button("Treinar modelo", type="primary", key="ml_train_btn"):
        if not feature_columns:
            st.error("Selecione ao menos uma feature.")
            return
        if target_column in feature_columns:
            st.error("Remova a coluna-alvo da lista de features.")
            return
        try:
            prepare_feature_matrix(
                df,
                feature_columns=feature_columns,
                target_column=target_column,
                regression_target=(task == "regression"),
            )
        except ValueError as exc:
            st.error(str(exc))
            return

        config = FlamlTrainConfig(
            task=task,
            time_budget=int(time_budget),
            metric=metric,
            n_splits=int(n_splits),
            estimator_list=tuple(estimator_list) or MINIMAL_ESTIMATOR_LIST,
            eval_method=eval_method,
            seed=int(seed),
            test_size=float(test_size),
        )
        with st.spinner("FLAML buscando o melhor modelo…"):
            try:
                bundle, report = train_flaml_model(
                    df,
                    feature_columns=feature_columns,
                    target_column=target_column,
                    config=config,
                    dataset_id=catalog.dataset_id,
                    catalog_id=st.session_state.get(SESSION_CATALOG_ID, catalog.dataset_id),
                )
            except Exception as exc:
                st.error(f"Falha no treino: {exc}")
                return

        st.session_state[SESSION_BUNDLE] = bundle
        st.session_state[SESSION_REPORT] = report
        if task == "regression":
            st.success(
                f"**{report.best_estimator}** · R²={report.r2:.3f} · RMSE={report.rmse:.3f}"
            )
        else:
            st.success(
                f"**{report.best_estimator}** · F1={report.f1:.3f} · acurácia={report.accuracy:.3f}"
            )

    report = st.session_state.get(SESSION_REPORT)
    if report is not None:
        if not getattr(report, "n_total", None):
            report.n_total = report.n_train + report.n_test
        st.markdown("**Último treino**")
        _render_train_results(report)
        if report.task != "regression":
            if report.classification_report:
                with st.expander("Relatório de classificação (teste)"):
                    st.code(report.classification_report, language="text")
            if report.confusion:
                with st.expander("Matriz de confusão (teste)"):
                    cm = pd.DataFrame(
                        report.confusion,
                        index=[f"real_{l}" for l in report.labels],
                        columns=[f"pred_{l}" for l in report.labels],
                    )
                    st.dataframe(cm, use_container_width=True)
        with st.expander("Configuração FLAML do vencedor"):
            st.json(report.best_config)


def _section_modelo() -> None:
    st.subheader("3. Salvar / carregar modelo (.pkl)")
    models_dir = ensure_ml_models_root()
    st.caption(f"Diretório: `{models_dir}`")

    bundle = st.session_state.get(SESSION_BUNDLE)
    catalog = st.session_state.get(SESSION_CATALOG)
    prefix = (catalog.dataset_id if catalog else "modelo") + "_"
    default_name = f"{prefix}{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"

    c1, c2 = st.columns(2)
    with c1:
        filename = st.text_input("Nome do arquivo", value=default_name, key="ml_save_name")
        if st.button("Salvar modelo", key="ml_save_btn", disabled=bundle is None):
            safe_name = PurePath(filename).name
            if not safe_name.endswith(".pkl"):
                safe_name = f"{safe_name}.pkl"
            path = save_model_bundle(bundle, models_dir / safe_name)
            st.success(f"Salvo em `{path}`")
    with c2:
        existing = sorted(models_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
        pick = st.selectbox("Modelos salvos", [""] + [p.name for p in existing], key="ml_pick_saved")
        if pick and st.button("Carregar", key="ml_load_saved"):
            st.session_state[SESSION_BUNDLE] = load_model_bundle(models_dir / pick)
            st.success(f"`{pick}` carregado.")

    bundle = st.session_state.get(SESSION_BUNDLE)
    if bundle is not None:
        with st.expander("Metadados do modelo"):
            st.code(bundle_metadata_json(bundle), language="json")


def _section_predicao() -> None:
    st.subheader("4. Predizer em dados novos")
    bundle = st.session_state.get(SESSION_BUNDLE)
    if bundle is None:
        st.info("Treine ou carregue um modelo .pkl antes de predizer.")
        return

    st.caption(f"Features esperadas: {', '.join(bundle.feature_columns)}")
    upload = st.file_uploader("CSV ou Excel", type=["csv", "xlsx", "xlsm"], key="ml_predict_upload")
    enrich = st.checkbox(
        "Mesclar metadados do dataset carregado (pela chave do catálogo)",
        value=False,
        key="ml_predict_enrich",
    )

    if upload is None:
        return

    try:
        new_df = read_prediction_table(upload)
    except ValueError as exc:
        st.error(str(exc))
        return

    if enrich:
        base_df = st.session_state.get(SESSION_DF)
        if (
            isinstance(base_df, pd.DataFrame)
            and DEFAULT_MERGE_KEY in new_df.columns
            and DEFAULT_MERGE_KEY in base_df.columns
        ):
            new_df = merge_with_amostras_if_needed(
                new_df, base_df, merge_key=DEFAULT_MERGE_KEY
            )

    missing = validate_prediction_columns(bundle, new_df)
    if missing:
        st.error(f"Colunas ausentes: {', '.join(missing)}")
        return

    if st.button("Executar predição", type="primary", key="ml_predict_btn"):
        result = predict_from_bundle(bundle, new_df)
        st.session_state["ml_last_predictions"] = result
        st.success(f"{len(result)} linha(s) predita(s).")

    result = st.session_state.get("ml_last_predictions")
    if isinstance(result, pd.DataFrame):
        st.dataframe(result, use_container_width=True, hide_index=True)
        st.download_button(
            "Baixar CSV",
            data=result.to_csv(index=False).encode("utf-8"),
            file_name="predicoes_ml.csv",
            mime="text/csv",
            key="ml_download_preds",
        )


def render_ml_tab() -> None:
    st.header("ML tradicional")
    st.caption(
        "Dataset padrão: **AbRank** (Kaggle) — regressão de `log_Aff` para pares anticorpo–antígeno. "
        f"Handle: `{KAGGLE_ABRANK_HANDLE}`."
    )

    tab_dados, tab_treino, tab_modelo, tab_pred = st.tabs(
        ["Dados", "Treino", "Modelo .pkl", "Predição"]
    )
    with tab_dados:
        _section_dados()
    with tab_treino:
        _section_treino()
    with tab_modelo:
        _section_modelo()
    with tab_pred:
        _section_predicao()
