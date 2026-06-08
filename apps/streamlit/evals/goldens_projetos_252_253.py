"""
Golden dataset DeepEval — projetos 252 (Chikungunya) e 253 (Dengue ELISA).

Gerado a partir dos documentos em:
- ``D:\\Vanessa\\AI_project\\Projetos\\252 - Teste rápido Chikungunya``
- ``D:\\Vanessa\\AI_project\\Projetos\\253 - ELISA indireto Dengue``

Distribuição: 5 RAG · 5 OLAP · 3 ML (pares de literatura Ab–Ag) · 5 RAG+OLAP (18 casos).
"""

from __future__ import annotations

from golden_dataset_template import (
    ChatGolden,
    EvalCategory,
    ExpectedRoutes,
)

PROJ_CHIK = "252 - Teste rápido Chikungunya"
PROJ_DENGUE = "253 - ELISA indireto Dengue"

# Planilhas OLAP do projeto Dengue (caminho relativo ao project_id).
XLSX_AMOSTRAS = "results/amostras_dengue.xlsx"
XLSX_COMPILADO = "results/Compilado_resultado_otimizacao.xlsx"

# Pares Ab–Ag de literatura para evals ML (AbRank / log_Aff), um score por golden.
_LITERATURE_ML_PAIRS: tuple[dict[str, str], ...] = (
    {
        "pair_id": "cr3022-sars2-rbd",
        "reference": "CR3022 × SARS-CoV-2 Spike S1/RBD",
        "heavy": (
            "QMQLVQSGTEVKKPGESLKISCKGSGYGFITYWIGWVRQMPGKGLEWMGIIYPGDSETRYSPSFQGQVTISADKSINTAYLQWSSLKASDTAIYYCAGGSGISTPMDVWGQGTTVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC"
        ),
        "light": (
            "DIQLTQSPDSLAVSLGERATINCKSSQSVLYSSINKNYLAWYQQKPGQPPKLLIYWASTRESGVPDRFSGSGSGTDFTLTISSLQAEDVAVYYCQQYYSTPYTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGECS"
        ),
        "ag": (
            "RVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFSGHHHHHH"
        ),
    },
    {
        "pair_id": "1f1-influenza-ha2",
        "reference": "Anticorpo 1F1 × hemaglutinina HA2 (PDB 4GXU)",
        "heavy": (
            "QVQLVQSGGGVVQPRRSLRLSCAASGFTFSSYAMHWVRQAPGKGLEWVAVISYDGRNKYYADSVKGRFTVSRDNSKNTLYLQMNSLRAEDTSVYYCARELLMDYYDHIGYSPGPTWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKRVEPKSCDK"
        ),
        "light": (
            "QPVLTQPPSASGSPGQRVTISCSGSSSNIGSYTVNWYQQLPGTAPKLLIYSLNQRPSGVPDRFSGSKSGTSASLAISGLQSEDEAVYYCAAWDDSLSAHVVFGGGTKLTVLGQPKAAPSVTLFPPSSEELQANKATLVCLISDFYPGAVTVAWKADSSPVKAGVETTTPSKQSNNKYAASSYLSLTPEQWKSHRSYSCQVTHEGSTVEKTVAPTECS"
        ),
        "ag": (
            "GLFGAIAGFIEGGWTGMIDGWYGYHHQNEQGSGYAADQKSTQNAIDGITNKVNSVIEKMNTQFTAVGKEFNNLERRIENLNKKVDDGFLDIWTYNAELLVLLENERTLDFHDSNVRNLYEKVKSQLKNNAKEIGNGCFEFYHKCDDACMESVRNGTYDYPKYSEESKLNREEIDGV"
        ),
    },
    {
        "pair_id": "1f1-influenza-ha1",
        "reference": "Anticorpo 1F1 × hemaglutinina HA1 (PDB 4GXU)",
        "heavy": (
            "QVQLVQSGGGVVQPRRSLRLSCAASGFTFSSYAMHWVRQAPGKGLEWVAVISYDGRNKYYADSVKGRFTVSRDNSKNTLYLQMNSLRAEDTSVYYCARELLMDYYDHIGYSPGPTWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKRVEPKSCDK"
        ),
        "light": (
            "QPVLTQPPSASGSPGQRVTISCSGSSSNIGSYTVNWYQQLPGTAPKLLIYSLNQRPSGVPDRFSGSKSGTSASLAISGLQSEDEAVYYCAAWDDSLSAHVVFGGGTKLTVLGQPKAAPSVTLFPPSSEELQANKATLVCLISDFYPGAVTVAWKADSSPVKAGVETTTPSKQSNNKYAASSYLSLTPEQWKSHRSYSCQVTHEGSTVEKTVAPTECS"
        ),
        "ag": (
            "ADPGDTICIGYHANNSTDTVDTVLEKNVTVTHSVNLLEDSHNGKLCKLKGIAPLQLGKCNIAGWLLGNPECDLLLTASSWSYIVETSNSENGTCYPGDFIDYEELREQLSSVSSFEKFEIFPKTSSWPNHETTKGVTAACSYAGASSFYRNLLWLTKKGSSYPKLSKSYVNNKGKEVLVLWGVHHPPTGTDQQSLYQNADAYVSVGSSKYNRRFTPEIAARPKVRDQAGRMNYYWTLLEPGDTITFEATGNLIAPWYAFALNRGSGSGIITSDAPVHDCNTKCQTPHGAINSSLPFQNIHPVTIGECPKYVRSTKLRMATGLRNIPSIQSR"
        ),
    },
    {
        "pair_id": "hyhel10-lysozyme",
        "reference": "HyHEL-10 × lisozima de clara de ovo (PDB 3HFM)",
        "heavy": (
            "DVQLQESGPSLVKPSQTLSLTCSVTGDSITSDYWSWIRKFPGNRLEYMGYVSYSGSTYYNPSLKSRISITRDTSKNQYYLDLNSVTTEDTATYYCANWDGDYWGQGTLVTVSAAKTTPPSVYPLAPGSAAQTNSMVTLGCLVKGYFPEPVTVTWNSGSLSSGVHTFPAVLQSDLYTLSSSVTVPSSPRPSETVTCNVAHPASSTKVDKKIVPRDC"
        ),
        "light": (
            "DIVLTQSPATLSVTPGNSVSLSCRASQSIGNNLHWYQQKSHESPRLLIKYASQSISGIPSRFSGSGSGTDFTLSINSVETEDFGMYFCQQSNSWPYTFGGGTKLEIKRADAAPTVSIFPPSSEQLTSGGASVVCFLNNFYPKDINVKWKIDGSERQNGVLNSWTDQDSKDSTYSMSSTLTLTKDEYERHNSYTCEATHKTSTSPIVKSFNRNEC"
        ),
        "ag": (
            "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"
        ),
    },
)

_ML_LITERATURE_PROMPTS: tuple[tuple[int, str], ...] = (
    (0, "Prediga log_Aff para o par anticorpo–antígeno de literatura abaixo (AbRank)."),
    (1, "Qual a afinidade predita log_Aff do anticorpo 1F1 contra a cadeia HA2 da hemaglutinina?"),
    (2, "Estime log_Aff do anticorpo 1F1 contra a cadeia HA1 da hemaglutinina influenza A."),
)


def _rag_goldens() -> list[ChatGolden]:
    return [
        ChatGolden(
            golden_id="rag-chik-lt260325-validade",
            input="Qual a validade do antígeno recombinante Chikungunya E1 lote LT260325 no projeto de validação de lote piloto?",
            expected_output=(
                "O antígeno recombinante Chikungunya E1 [2,2 mg/ml], lote LT260325 "
                "tem validade 26/03/2027, conforme planejamento 1a (07/04/2026)."
            ),
            context=[
                "Item 1 — Nome/Concentração: Antígeno recombinante Chikungunya E1 [2,2 mg/ml] "
                "— Fabricante/Código: LifeScience/CKV220223 — Lote/ativo: LT260325 — Validade: 26/03/2027"
            ],
            category=EvalCategory.RAG,
            expected_routes=ExpectedRoutes(documents=True),
            project_ids=[PROJ_CHIK],
            tags=["rag", "chikungunya", "validade"],
            comments="Fonte: planning/1a.docx",
        ),
        ChatGolden(
            golden_id="rag-chik-lt310625-objetivo",
            input="Qual o objetivo do planejamento 1b do teste rápido Chikungunya executado em 10/04/2026?",
            expected_output=(
                "Validar o lote de antígeno recombinante comercial LT310625 como controle de reação, "
                "com execução em 10/04/2026 (planejamento de 09/04/2026)."
            ),
            context=[
                "Objetivo: Validar lote de antígeno recombinante comercial LT310625 como controle de reação.",
                "Planejamento: 09/04/2026 — Execução: 10/04/2026",
            ],
            category=EvalCategory.RAG,
            expected_routes=ExpectedRoutes(documents=True),
            project_ids=[PROJ_CHIK],
            tags=["rag", "chikungunya", "objetivo"],
            comments="Fonte: planning/Planejamento_1b.docx",
        ),
        ChatGolden(
            golden_id="rag-chik-diluicao-v1",
            input="No ensaio de validação do lote Chikungunya, qual volume de antígeno diluir para obter 0,2 mg/ml?",
            expected_output=(
                "V1 = 0,045 ml (45 µl) de antígeno para obter 0,2 mg/ml."
            ),
            context=[
                "Diluição do antígeno: C1 x V1 = C2 x V2 — 2,2 x V1 = 0,2 x 0,5 — V1 = 0,045 ml ou 45 ul.",
            ],
            category=EvalCategory.RAG,
            expected_routes=ExpectedRoutes(documents=True),
            project_ids=[PROJ_CHIK],
            tags=["rag", "chikungunya", "calculo"],
            comments="Fonte: planning/1a.docx",
        ),
        ChatGolden(
            golden_id="rag-chik-amostras-lacen",
            input="Quais IDs de amostra de origem LACEN-PE foram usados na validação Chikungunya (planejamento 1a)?",
            expected_output=(
                "Amostras LACEN-PE: 265, 266, 267, 302, 315 e 360."
            ),
            context=[
                "Item 1 — ID amostra: 265 — Origem: LACEN-PE",
                "Item 2 — ID amostra: 266 — Origem: LACEN-PE",
                "Item 6 — ID amostra: 360 — Origem: LACEN-PE",
            ],
            category=EvalCategory.RAG,
            expected_routes=ExpectedRoutes(documents=True),
            project_ids=[PROJ_CHIK],
            tags=["rag", "chikungunya", "amostras"],
            comments="Fonte: planning/1a.docx — tabela Seleção de amostras",
        ),
        ChatGolden(
            golden_id="rag-chik-metodologia-leitura",
            input="Em quanto tempo deve ser feita a leitura do teste rápido Chikungunya no protocolo 1a?",
            expected_output="Leitura de resultado entre 15 e 20 minutos.",
            context=[
                "Metodologia: Aplicar 20 ul de amostra no poço de amostra; aplicar 3 gotas de tampão de corrida "
                "(ou 90 ul com micropipeta) no poço de tampão. Leitura de resultado entre 15 e 20 minutos.",
            ],
            category=EvalCategory.RAG,
            expected_routes=ExpectedRoutes(documents=True),
            project_ids=[PROJ_CHIK],
            tags=["rag", "chikungunya", "metodologia"],
            comments="Fonte: planning/1a.docx",
        ),
    ]


def _olap_goldens() -> list[ChatGolden]:
    return [
        ChatGolden(
            golden_id="olap-dengue-total-amostras",
            input="Quantas amostras constam na planilha amostras_dengue.xlsx do projeto ELISA indireto Dengue?",
            expected_output=(
                f"Segundo {XLSX_AMOSTRAS} (aba Amostras): 320 amostras registradas."
            ),
            category=EvalCategory.OLAP,
            expected_routes=ExpectedRoutes(spreadsheets=True),
            project_ids=[PROJ_DENGUE],
            requires_index=False,
            requires_olap=True,
            tags=["olap", "dengue", "contagem"],
            comments="Contagem de linhas da planilha Amostras (320 registros).",
        ),
        ChatGolden(
            golden_id="olap-dengue-media-idade",
            input="Qual a idade média das amostras na planilha amostras_dengue?",
            expected_output=(
                f"Segundo {XLSX_AMOSTRAS} (coluna Idade): idade média ≈ 53,4 anos (53,41 arredondado)."
            ),
            category=EvalCategory.OLAP,
            expected_routes=ExpectedRoutes(spreadsheets=True),
            project_ids=[PROJ_DENGUE],
            requires_index=False,
            requires_olap=True,
            tags=["olap", "dengue", "media"],
            comments="Média aritmética da coluna Idade.",
        ),
        ChatGolden(
            golden_id="olap-dengue-idade-maior-80",
            input="Quantos pacientes têm idade superior a 80 anos em amostras_dengue?",
            expected_output=(
                f"Segundo {XLSX_AMOSTRAS} (coluna Idade): 38 pacientes com idade > 80 anos."
            ),
            category=EvalCategory.OLAP,
            expected_routes=ExpectedRoutes(spreadsheets=True),
            project_ids=[PROJ_DENGUE],
            requires_index=False,
            requires_olap=True,
            tags=["olap", "dengue", "filtro"],
            comments="Filtro Idade > 80.",
        ),
        ChatGolden(
            golden_id="olap-dengue-dias-sintoma-amostra-133",
            input="Quantos dias de sintoma tem a amostra ID 133 na planilha amostras_dengue?",
            expected_output=(
                f"Segundo {XLSX_AMOSTRAS}: amostra 133 (Yasmin Ribeiro) — 1 dia de sintoma "
                "(coluna Dias de sintoma)."
            ),
            category=EvalCategory.OLAP,
            expected_routes=ExpectedRoutes(spreadsheets=True),
            project_ids=[PROJ_DENGUE],
            requires_index=False,
            requires_olap=True,
            tags=["olap", "dengue", "lookup"],
            comments="Lookup por ID amostra = 133.",
        ),
        ChatGolden(
            golden_id="olap-dengue-compilado-positivos",
            input="Quantas amostras positivas há no Compilado_resultado_otimizacao.xlsx?",
            expected_output=(
                f"Segundo {XLSX_COMPILADO} (coluna Resultado caracterização ELISA comercial): "
                "89 amostras Positivo."
            ),
            category=EvalCategory.OLAP,
            expected_routes=ExpectedRoutes(spreadsheets=True),
            project_ids=[PROJ_DENGUE],
            requires_index=False,
            requires_olap=True,
            tags=["olap", "dengue", "contagem"],
            comments="Compilado_resultado_otimizacao.xlsx — coluna Resultado caracterização.",
        ),
    ]


def _ml_goldens() -> list[ChatGolden]:
    """Predição AbRank — pares Ab–Ag de literatura (um score log_Aff por golden)."""
    goldens: list[ChatGolden] = []
    for i, (pair_idx, prompt) in enumerate(_ML_LITERATURE_PROMPTS, start=1):
        pair = _LITERATURE_ML_PAIRS[pair_idx]
        pair_id = pair["pair_id"]
        reference = pair["reference"]
        goldens.append(
            ChatGolden(
                golden_id=f"ml-literature-{pair_id}-{i:02d}",
                input=(
                    f"{prompt}\n"
                    f"Ab_heavy_chain_seq: {pair['heavy']}\n"
                    f"Ab_light_chain_seq: {pair['light']}\n"
                    f"Ag_seq: {pair['ag']}"
                ),
                expected_output=(
                    f"Retornar o score predito de log_Aff para o par de literatura "
                    f'"{reference}", via modelo .pkl AbRank. '
                    "Cada par deve ser avaliado separadamente — congele o valor numérico "
                    "após a primeira predição manual com o .pkl ativo."
                ),
                category=EvalCategory.ML,
                expected_routes=ExpectedRoutes(ml_prediction=True),
                requires_index=False,
                requires_olap=False,
                requires_ml_model=True,
                tags=["ml", "abrank", "literature", pair_id],
                comments=(
                    f"Par de literatura: {reference}. "
                    "Sequências completas H/L/Ag conforme PDB/literatura; não consulta projetos 252/253."
                ),
            )
        )
    return goldens


def _combined_goldens() -> list[ChatGolden]:
    return [
        ChatGolden(
            golden_id="comb-dengue-tmb-lote-abs12",
            input=(
                "No ensaio de otimização de tempo Dengue, qual lote do TMB/substrato foi usado "
                "e qual a absorbância 16 h da amostra 12 no compilado?"
            ),
            expected_output=(
                "TMB/Substrato LifeScience/965014, lote LT099185 (protocolo). "
                f"Segundo {XLSX_COMPILADO}: amostra 12 (Positivo) — ABS Tempo sensibilização 16h = 2,344."
            ),
            context=[
                "Item 5 — TMB/Substrato — LifeScience/965014 — Lote: LT099185 — Validade: 30/09/2027",
            ],
            category=EvalCategory.COMBINED,
            expected_routes=ExpectedRoutes(documents=True, spreadsheets=True),
            project_ids=[PROJ_DENGUE],
            requires_olap=True,
            tags=["combined", "dengue", "tmb", "abs"],
            comments="RAG: 20260209ensaiootimizacaotempo.docx + OLAP: Compilado ID 12.",
        ),
        ChatGolden(
            golden_id="comb-dengue-placa-lote-pos-count",
            input=(
                "Qual lote de placas 96 poços foi usado na sensibilização Dengue e quantas amostras "
                "positivas há no compilado de otimização?"
            ),
            expected_output=(
                "Placas 96 poços Scientifics/9668620, lote PLRT6501M (protocolo). "
                f"Segundo {XLSX_COMPILADO}: 89 amostras Positivo."
            ),
            context=[
                "Item 3 — Placas 96 poços — Scientifics/9668620 — Lote: PLRT6501M — Validade: 02/2029",
            ],
            category=EvalCategory.COMBINED,
            expected_routes=ExpectedRoutes(documents=True, spreadsheets=True),
            project_ids=[PROJ_DENGUE],
            requires_olap=True,
            tags=["combined", "dengue", "placa"],
            comments="Sensibilização placa + contagem Positivo no compilado.",
        ),
        ChatGolden(
            golden_id="comb-dengue-amostra36-doc-abs",
            input=(
                "A amostra 36 foi classificada como positiva no protocolo de otimização de tempo? "
                "Qual sua absorbância com sensibilização 4 h no compilado?"
            ),
            expected_output=(
                "Sim — amostra 36 é Positivo no ELISA comercial (protocolo). "
                f"Segundo {XLSX_COMPILADO}: ABS Tempo sensibilização 4h = 2,631."
            ),
            context=["Item 3 — ID amostra: 36 — Caracterização ELISA comercial: Positivo"],
            category=EvalCategory.COMBINED,
            expected_routes=ExpectedRoutes(documents=True, spreadsheets=True),
            project_ids=[PROJ_DENGUE],
            requires_olap=True,
            tags=["combined", "dengue", "amostra-36"],
        ),
        ChatGolden(
            golden_id="comb-dengue-temperatura-doc-abs41",
            input=(
                "Quais temperaturas de incubação foram testadas no ensaio de 10/02/2026 e qual a ABS "
                "da amostra 41 a 37°C no compilado?"
            ),
            expected_output=(
                "Protocolo avalia temperatura ambiente e 37 °C. "
                f"Segundo {XLSX_COMPILADO}: amostra 41 (Positivo) — ABS Temperatura amostra 37°C = 3,214."
            ),
            context=["Observação: serão avaliadas temperatura ambiente e 37°C."],
            category=EvalCategory.COMBINED,
            expected_routes=ExpectedRoutes(documents=True, spreadsheets=True),
            project_ids=[PROJ_DENGUE],
            requires_olap=True,
            tags=["combined", "dengue", "temperatura"],
            comments="20260210ensaiootimizacaotemperatura.docx + compilado ID 41.",
        ),
        ChatGolden(
            golden_id="comb-dengue-diluicao-antigeno-volume",
            input=(
                "Quantos ml de antígeno concentrado são necessários para 40 ml a 100 ng/ml na sensibilização "
                "e qual a idade da amostra ID 1 na planilha amostras_dengue?"
            ),
            expected_output=(
                "Protocolo: V1 = 4,08 ml de antígeno + 35,92 ml de tampão bicarbonato (980 ng/ml → 100 ng/ml, 40 ml). "
                f"Segundo {XLSX_AMOSTRAS}: amostra 1 (Yasmin Ribeiro), idade 56 anos."
            ),
            context=[
                "980 x V1 = 100 x 40 — V1 = 4,08 ml e 35,92 ml de tampão.",
            ],
            category=EvalCategory.COMBINED,
            expected_routes=ExpectedRoutes(documents=True, spreadsheets=True),
            project_ids=[PROJ_DENGUE],
            requires_olap=True,
            tags=["combined", "dengue", "calculo", "idade"],
        ),
    ]


def build_projetos_goldens() -> list[ChatGolden]:
    """Retorna os 18 goldens curados dos projetos 252 e 253."""
    items = _rag_goldens() + _olap_goldens() + _ml_goldens() + _combined_goldens()
    assert len(items) == 18, f"Esperado 18 goldens, obtido {len(items)}"
    return items
