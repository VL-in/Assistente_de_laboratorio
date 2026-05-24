"""
Aba Streamlit — ML tradicional (FLAML + exportação .pkl + predição).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from ml.datasets import (
    default_feature_columns,
    load_dengue_elisa_253,
    prepare_feature_matrix,
)
from ml.dictionary import load_dataset_catalog
from ml.paths import ensure_ml_models_root, ml_models_root, resolve_dengue_results_dir
from ml.predict import (
    merge_with_amostras_if_needed,
    predict_from_bundle,
    read_prediction_table,
    validate_prediction_columns,
)
from ml.training import (
    FlamlTrainConfig,
    MINIMAL_ESTIMATOR_LIST,
    bundle_metadata_json,
    flaml_available,
    load_model_bundle,
    save_model_bundle,
    train_flaml_classifier,
)

SESSION_DF = "ml_df"
SESSION_CATALOG = "ml_catalog"
SESSION_BUNDLE = "ml_model_bundle"
SESSION_REPORT = "ml_train_report"


def _load_default_dataset() -> None:
    catalog = load_dataset_catalog()
    results_dir = resolve_dengue_results_dir()
    df, _ = load_dengue_elisa_253(results_dir, catalog=catalog)
    st.session_state[SESSION_DF] = df
    st.session_state[SESSION_CATALOG] = catalog
    st.session_state["ml_results_dir"] = str(results_dir)


def _section_dados() -> None:
    st.subheader("1. Dataset e dicionário de colunas")
    results_dir = resolve_dengue_results_dir()
    st.caption(
        f"Pasta de dados padrão: `{results_dir}`. "
        "No Docker, use a pasta Projetos montada ou defina `ASSISTENTE_ML_DENGUE_RESULTS`."
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("Carregar projeto 253 (Dengue)", type="primary", key="ml_load_default"):
            try:
                _load_default_dataset()
                st.success("Dataset carregado.")
            except (FileNotFoundError, ValueError) as exc:
                st.error(str(exc))
    with c2:
        uploaded = st.file_uploader(
            "Ou envie CSV/Excel merged (opcional)",
            type=["csv", "xlsx", "xlsm"],
            key="ml_upload_dataset",
        )
        if uploaded is not None:
            try:
                df = read_prediction_table(uploaded)
                st.session_state[SESSION_DF] = df
                st.session_state[SESSION_CATALOG] = load_dataset_catalog()
                st.info(f"Arquivo `{uploaded.name}` carregado — {len(df)} linha(s).")
            except ValueError as exc:
                st.error(str(exc))

    df: pd.DataFrame | None = st.session_state.get(SESSION_DF)
    if df is None:
        st.info("Carregue o dataset do projeto 253 ou envie um arquivo para começar.")
        return

    catalog = st.session_state.get(SESSION_CATALOG) or load_dataset_catalog()
    m1, m2, m3 = st.columns(3)
    m1.metric("Linhas", len(df))
    m2.metric("Colunas", len(df.columns))
    target = catalog.default_target
    if target in df.columns:
        vc = df[target].value_counts(dropna=False)
        m3.metric("Classes (alvo)", len(vc))

    with st.expander("Prévia dos dados", expanded=False):
        st.dataframe(df.head(20), use_container_width=True, hide_index=True)

    with st.expander("Dicionário de colunas (o que cada campo significa)", expanded=True):
        dict_df = pd.DataFrame(catalog.column_dict_rows())
        present = set(df.columns)
        dict_df["presente_no_dataset"] = dict_df["coluna"].isin(present)
        st.dataframe(dict_df, use_container_width=True, hide_index=True)

    st.markdown("**Colunas sugeridas para remover** (PII, identificadores, datas brutas):")
    suggested = [c for c in catalog.suggested_drop if c in df.columns]
    st.write(", ".join(suggested) if suggested else "— nenhuma encontrada com esse nome exato —")

    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0].sort_values(ascending=False)
    if not nulls.empty:
        with st.expander("Valores ausentes por coluna"):
            st.dataframe(nulls.rename("n_ausentes"), use_container_width=True)


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

    catalog = st.session_state.get(SESSION_CATALOG) or load_dataset_catalog()
    target_options = [c for c in df.columns if c == catalog.default_target] or list(df.columns)
    target_column = st.selectbox(
        "Coluna-alvo (variável a prever)",
        options=target_options,
        index=0,
        key="ml_target_col",
        help=catalog.description_for(catalog.default_target),
    )

    default_features = default_feature_columns(df, catalog)
    feature_columns = st.multiselect(
        "Features (variáveis explicativas)",
        options=sorted(df.columns),
        default=default_features,
        key="ml_feature_cols",
        help="Sugestão inicial exclui PII e identificadores do catálogo.",
    )

    with st.expander("Hiperparâmetros FLAML", expanded=True):
        c1, c2, c3 = st.columns(3)
        time_budget = c1.number_input(
            "Tempo máximo (s)",
            min_value=10,
            max_value=3600,
            value=120,
            step=10,
            key="ml_time_budget",
        )
        metric = c2.selectbox(
            "Métrica",
            options=["f1", "accuracy", "roc_auc", "log_loss"],
            index=0,
            key="ml_metric",
        )
        test_size = c3.slider(
            "Proporção teste",
            min_value=0.1,
            max_value=0.4,
            value=0.2,
            step=0.05,
            key="ml_test_size",
        )
        c4, c5, c6 = st.columns(3)
        n_splits = c4.number_input("Folds (CV)", min_value=2, max_value=10, value=5, key="ml_n_splits")
        seed = c5.number_input("Seed", min_value=0, value=42, key="ml_seed")
        eval_method = c6.selectbox("Avaliação", ["cv", "holdout"], index=0, key="ml_eval_method")
        estimator_list = st.multiselect(
            "Estimadores (instalação mínima = sklearn)",
            options=list(MINIMAL_ESTIMATOR_LIST),
            default=list(MINIMAL_ESTIMATOR_LIST),
            key="ml_estimators",
        )
        st.caption(
            "Para LightGBM/XGBoost instale extras depois (`pip install lightgbm`). "
            "No MVP usamos só estimadores que vêm com FLAML + scikit-learn."
        )

    if st.button("Treinar modelo", type="primary", key="ml_train_btn"):
        if not feature_columns:
            st.error("Selecione ao menos uma feature.")
            return
        if target_column in feature_columns:
            st.error("Remova a coluna-alvo da lista de features.")
            return
        try:
            prepare_feature_matrix(df, feature_columns=feature_columns, target_column=target_column)
        except ValueError as exc:
            st.error(str(exc))
            return

        config = FlamlTrainConfig(
            time_budget=int(time_budget),
            metric=metric,
            n_splits=int(n_splits),
            estimator_list=tuple(estimator_list) or MINIMAL_ESTIMATOR_LIST,
            eval_method=eval_method,
            seed=int(seed),
            test_size=float(test_size),
        )
        with st.spinner("FLAML está buscando o melhor modelo (pode levar alguns minutos)…"):
            try:
                bundle, report = train_flaml_classifier(
                    df,
                    feature_columns=feature_columns,
                    target_column=target_column,
                    config=config,
                    dataset_id=catalog.dataset_id,
                )
            except Exception as exc:
                st.error(f"Falha no treino: {exc}")
                return

        st.session_state[SESSION_BUNDLE] = bundle
        st.session_state[SESSION_REPORT] = report
        st.success(
            f"Melhor estimador: **{report.best_estimator}** · "
            f"F1={report.f1:.3f} · acurácia={report.accuracy:.3f}"
        )

    report = st.session_state.get(SESSION_REPORT)
    if report is not None:
        st.markdown("**Resultados do último treino**")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("F1", f"{report.f1:.3f}")
        r2.metric("Acurácia", f"{report.accuracy:.3f}")
        r3.metric("Treino / teste", f"{report.n_train} / {report.n_test}")
        if report.roc_auc is not None:
            r4.metric("ROC-AUC", f"{report.roc_auc:.3f}")

        st.text("Relatório por classe:")
        st.code(report.classification_report, language="text")
        cm = pd.DataFrame(
            report.confusion,
            index=[f"real_{l}" for l in report.labels],
            columns=[f"pred_{l}" for l in report.labels],
        )
        st.dataframe(cm, use_container_width=True)

        with st.expander("Melhor configuração FLAML"):
            st.json(report.best_config)


def _section_modelo() -> None:
    st.subheader("3. Salvar / carregar modelo (.pkl)")
    models_dir = ensure_ml_models_root()
    st.caption(f"Diretório de modelos: `{models_dir}`")

    bundle = st.session_state.get(SESSION_BUNDLE)
    default_name = f"dengue_elisa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"

    c1, c2 = st.columns(2)
    with c1:
        filename = st.text_input("Nome do arquivo", value=default_name, key="ml_save_name")
        if st.button("Salvar modelo treinado", key="ml_save_btn", disabled=bundle is None):
            if bundle is None:
                st.warning("Treine um modelo antes de salvar.")
            else:
                path = save_model_bundle(bundle, models_dir / filename)
                st.success(f"Modelo salvo em `{path}`")

    with c2:
        existing = sorted(models_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
        pick = st.selectbox(
            "Modelos salvos",
            options=[""] + [p.name for p in existing],
            key="ml_pick_saved",
        )
        if pick and st.button("Carregar selecionado", key="ml_load_saved"):
            loaded = load_model_bundle(models_dir / pick)
            st.session_state[SESSION_BUNDLE] = loaded
            st.success(f"Modelo `{pick}` carregado.")

    bundle = st.session_state.get(SESSION_BUNDLE)
    if bundle is not None:
        with st.expander("Metadados do modelo ativo"):
            st.code(bundle_metadata_json(bundle), language="json")


def _section_predicao() -> None:
    st.subheader("4. Predizer em dados novos")
    bundle = st.session_state.get(SESSION_BUNDLE)
    if bundle is None:
        st.info("Treine ou carregue um modelo .pkl antes de predizer.")
        return

    st.caption(
        f"O modelo espera estas features: {', '.join(bundle.feature_columns)}"
    )
    upload = st.file_uploader(
        "Planilha ou CSV com novas amostras",
        type=["csv", "xlsx", "xlsm"],
        key="ml_predict_upload",
    )
    enrich = st.checkbox(
        "Enriquecer por ID amostra com metadados do dataset carregado (se disponível)",
        value=True,
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
        if isinstance(base_df, pd.DataFrame):
            new_df = merge_with_amostras_if_needed(new_df, base_df)

    missing = validate_prediction_columns(bundle, new_df)
    if missing:
        st.error(f"Colunas obrigatórias ausentes: {', '.join(missing)}")
        return

    if st.button("Executar predição", type="primary", key="ml_predict_btn"):
        try:
            result = predict_from_bundle(bundle, new_df)
        except (ValueError, AttributeError) as exc:
            st.error(str(exc))
            return
        st.session_state["ml_last_predictions"] = result
        st.success(f"Predição concluída — {len(result)} linha(s).")

    result = st.session_state.get("ml_last_predictions")
    if isinstance(result, pd.DataFrame):
        st.dataframe(result, use_container_width=True, hide_index=True)
        csv_bytes = result.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Baixar resultados (CSV)",
            data=csv_bytes,
            file_name="predicoes_ml.csv",
            mime="text/csv",
            key="ml_download_preds",
        )


def render_ml_tab() -> None:
    """Renderiza a aba principal ML tradicional."""
    st.header("ML tradicional")
    st.caption(
        "Fluxo preparado para cientista de dados: dicionário de colunas, seleção de features, "
        "AutoML com FLAML (instalação mínima), exportação .pkl e predição em lote."
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
