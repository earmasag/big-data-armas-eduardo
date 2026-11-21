"""Dashboard Web Interactivo en Streamlit — Proyecto Final de Big Data (Los Libertadores).

Diseño limpio y visual centrado en gráficos interactivos (Altair + paleta Okabe-Ito),
consumiendo exclusivamente datos ya procesados (Parquet, MongoDB), el artefacto entrenado
en models/modelo_orinoco.joblib y el reporte PDF en reports/reporte_bigdata_orinoco.pdf.

Secciones obligatorias:
1. Resumen: KPIs ejecutivos y dos gráficos principales (serie del río y comparativa MAE).
2. Datos y Cubos OLAP: visualizaciones OLAP (Heatmap + Barras ENSO) y tablas compactas.
3. Visualizaciones: explorador interactivo multi-variable y selector limpio de las 6 figuras.
4. Modelo ML: evaluación visual (Real vs. Predicho, MAE por estación, importancia) y simulador simplificado.
5. Insights y Reporte: tarjetas colapsables desde MongoDB, diccionario hidrometeorológico y descarga PDF.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.etl_pipeline import PROCESSED_DIR, STATION_LABELS
from src.generate_pdf import PDF_REPORT_PATH
from src.ml_pipeline import MODEL_ARTIFACT_PATH
from src.mongodb_store import load_analysis_from_mongodb, store_and_query_mongodb

st.set_page_config(
    page_title="Big Data Orinoco — CRISP-DM + ML + MongoDB",
    page_icon="🌊",
    layout="wide",
)

st.markdown(
    """
    <style>
    [data-testid="stTable"] td, [data-testid="stTable"] th {
        white-space: normal !important;
        word-break: break-word;
        vertical-align: top;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Paleta accesible Okabe-Ito (consistente con las figuras del estudio)
STATION_COLOR_DOMAIN = [
    "Puerto Ayacucho",
    "Caicara del Orinoco",
    "Ciudad Bolívar",
    "Palúa (San Félix)",
]
STATION_COLOR_RANGE = ["#0072B2", "#009E73", "#D55E00", "#CC79A7"]

ENSO_COLOR_DOMAIN = ["La Niña (Frío)", "Neutro", "El Niño (Cálido)"]
ENSO_COLOR_RANGE = ["#0072B2", "#7F7F7F", "#D55E00"]

MODEL_COLOR_DOMAIN = [
    "Línea Base (Naive)",
    "Ridge Regression",
    "Random Forest",
    "XGBoost Regressor",
]
MODEL_COLOR_RANGE = ["#B0B6C2", "#56B4E9", "#009E73", "#FF7B29"]

EPOCA_OPTIONS = [
    "1. Estiaje (Ene-Mar)",
    "2. Ascenso (Abr-May)",
    "3. Aguas Altas (Jun-Sep)",
    "4. Descenso (Oct-Dic)",
]
ENSO_OPTIONS = ["La Niña (Frío)", "Neutro", "El Niño (Cálido)"]

VARIABLE_META = {
    "y": "Nivel Hidrométrico (m s.n.m.)",
    "nivel_minmax_pct": "Capacidad Relativa (%)",
    "delta_y_3d": "Velocidad de Cambio 3 días (m)",
    "lluvia_local_14d": "Lluvia Local Acumulada 14 días (mm)",
    "lluvia_cuenca_14d": "Lluvia Media Cuenca 14 días (mm)",
    "oni": "Índice Climático NOAA ONI (°C)",
}


@st.cache_data(show_spinner=False)
def load_parquet_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga los Parquet procesados sin reejecutar el ETL."""
    main_path = PROCESSED_DIR / "orinoco_limpio.parquet"
    if not main_path.exists():
        from src.run_all import execute_full_pipeline
        execute_full_pipeline()

    df = pd.read_parquet(main_path)
    olap1 = pd.read_parquet(PROCESSED_DIR / "olap_anio_estacion_epoca.parquet")
    olap2 = pd.read_parquet(PROCESSED_DIR / "olap_enso_estacion_trimestre.parquet")
    preds_test = pd.read_parquet(PROCESSED_DIR / "predicciones_test_2023_2024.parquet")
    return df, olap1, olap2, preds_test


@st.cache_resource(show_spinner=False)
def load_trained_artifact() -> dict[str, Any]:
    """Carga el modelo entrenado desde models/modelo_orinoco.joblib sin reentrenar."""
    if not MODEL_ARTIFACT_PATH.exists():
        from src.run_all import execute_full_pipeline
        execute_full_pipeline()
    return joblib.load(MODEL_ARTIFACT_PATH)


def _get_scenario_values(
    df: pd.DataFrame,
    defaults_by_station: dict[str, dict[str, Any]],
    station_id: str,
    scenario: str,
) -> tuple[dict[str, Any], str]:
    """Retorna valores reales del dataset para el escenario seleccionado en el simulador."""
    if scenario == "Último registro":
        base = dict(defaults_by_station[station_id])
        base.setdefault("doy", 352)
        return base, f"Último registro disponible ({base.get('ultima_fecha', '2024-12-17')})"

    required = [
        "y", "y_lag_7", "y_lag_14", "delta_y_1d", "delta_y_3d",
        "ayacucho_lag_7", "ayacucho_lag_14", "caicara_lag_7",
        "ciudad_bolivar_lag_1", "lluvia_local_14d", "lluvia_cuenca_14d", "oni",
    ]
    sub = df[df["unique_id"] == station_id].dropna(subset=required)

    if scenario == "🌊 Crecida (Aguas Altas / La Niña)":
        pool = sub[
            (sub["epoca_hidrologica"] == "3. Aguas Altas (Jun-Sep)")
            & (sub["fase_enso"] == "La Niña (Frío)")
        ]
        if pool.empty:
            pool = sub[sub["epoca_hidrologica"] == "3. Aguas Altas (Jun-Sep)"]
        row = pool.loc[pool["y"].idxmax()]
    else:
        pool = sub[
            (sub["epoca_hidrologica"] == "1. Estiaje (Ene-Mar)")
            & (sub["fase_enso"] == "El Niño (Cálido)")
        ]
        if pool.empty:
            pool = sub[sub["epoca_hidrologica"] == "1. Estiaje (Ene-Mar)"]
        row = pool.loc[pool["y"].idxmin()]

    values = {col: float(row[col]) for col in required}
    values["epoca_hidrologica"] = str(row["epoca_hidrologica"])
    values["fase_enso"] = str(row["fase_enso"])
    values["doy"] = int(row["doy"])
    fecha_str = str(pd.Timestamp(row["ds"]).date())
    return values, f"Escenario real histórico del {fecha_str}"


def _build_glossary_dataframe() -> pd.DataFrame:
    """Devuelve el diccionario hidrometeorológico de las 18 variables + target."""
    rows = [
        ("y", "Hidrometría local", "Nivel diario del río en la estación (m s.n.m.).", "Excel INAMEH/UCAB", "Base física del pronóstico; referencia del modelo Naive."),
        ("y_lag_7 / y_lag_14", "Hidrometría local", "Nivel de la estación 7 y 14 días atrás (m).", "Rezago de y", "Captura la inercia semanal y quincenal del hidrograma."),
        ("delta_y_1d / 3d / 7d", "Hidrometría local", "Cambio de nivel en las últimas 24 h, 3 días y 7 días (m).", "Derivada de y", "Mide la velocidad y aceleración de la onda de crecida o bajante."),
        ("unique_id", "Hidrometría local", "Estación limnimétrica (Ayacucho, Caicara, Ciudad Bolívar, Palúa).", "Catálogo", "Ajusta amplitud y comportamiento propio de cada tramo fluvial."),
        ("ayacucho_lag_7 / 14", "Tránsito aguas arriba", "Nivel en Puerto Ayacucho hace 7 y 14 días (m).", "Serie Ayacucho", "Alerta temprana del Alto Orinoco hacia estaciones aguas abajo."),
        ("caicara_lag_7", "Tránsito aguas arriba", "Nivel en Caicara del Orinoco hace 7 días (m).", "Serie Caicara", "Integra aportes del Meta y Apure antes de llegar a Ciudad Bolívar."),
        ("ciudad_bolivar_lag_1", "Tránsito aguas arriba", "Nivel en Ciudad Bolívar hace 1 día (m).", "Serie C. Bolívar", "Referencia inmediata del Bajo Orinoco hacia Palúa."),
        ("lluvia_local_14d", "Meteorología satelital", "Precipitación acumulada en 14 días sobre la estación (mm).", "NASA POWER", "Mide aporte pluvial directo y saturación local del suelo."),
        ("lluvia_cuenca_14d", "Meteorología satelital", "Precipitación media acumulada 14d en 9 puntos de la cuenca (mm).", "NASA POWER", "Principal forzante externa de escorrentía en la cuenca alta."),
        ("oni", "Macroclima ENSO", "Anomalía térmica Niño 3.4 con lag de publicación de 3 meses (°C).", "NOAA CPC ONI", "Contextualiza ciclos interanuales secos (El Niño) o húmedos (La Niña)."),
        ("fase_enso", "Macroclima ENSO", "Régimen climático (La Niña <= -0.5, Neutro, El Niño >= +0.5 °C).", "Derivada de ONI", "Segmenta comportamiento hidrológico por fase macroclimática."),
        ("sin_doy / cos_doy", "Estacionalidad", "Armónicos de Fourier del día del año (periodo 365.25 días).", "Fecha ds", "Representa el ciclo monomodal anual de lluvias de forma continua."),
        ("epoca_hidrologica", "Estacionalidad", "Estiaje (Ene-Mar), Ascenso (Abr-May), Aguas Altas (Jun-Sep), Descenso (Oct-Dic).", "Mes de ds", "Diferencia dinámicas opuestas para un mismo nivel (subiendo vs. bajando)."),
        ("target_y_t14", "Variable Objetivo", "Nivel real observado 14 días en el futuro y(t+14) (m).", "Desplazamiento -14d", "Objetivo del pronóstico; el modelo predice Δy y suma y(t)."),
    ]
    return pd.DataFrame(
        rows,
        columns=["Variable", "Categoría", "Significado hidrometeorológico", "Origen", "Aporte al análisis / modelo"],
    )


def _impact_badge(impacto: str) -> str:
    return {
        "Crítico": "🔴 Crítico",
        "Alto": "🟠 Alto",
        "Medio": "🟡 Medio",
    }.get(impacto, f"🔵 {impacto}")


def main() -> None:
    df, olap1, olap2, preds_test = load_parquet_data()
    artifact = load_trained_artifact()
    mongo_doc, mongo_backend = load_analysis_from_mongodb()

    meta = mongo_doc["metadata"]
    ds_meta = meta["dataset"]
    bench = meta["benchmark_arquitectura"]
    cal = mongo_doc["calidad_datos"]
    mod = mongo_doc["modelo"]

    st.title("🌊 Pronóstico Hidrológico del Río Orinoco (+14 días)")
    st.caption(
        f"**Big Data (Los Libertadores)** · ETL: `{bench['herramienta_elegida']}` ({bench['tiempo_etl_completo_seg']} s) · "
        f"NoSQL: `{mongo_backend}` · Modelo Ganador: `{mod['algoritmo']}` (MAE `{mod['metricas']['mae_m']:.4f} m`)"
    )

    tabs = st.tabs([
        "1. 📊 Resumen",
        "2. 🗂️ Datos y OLAP",
        "3. 📈 Visualizaciones",
        "4. 🤖 Modelo y Simulador",
        "5. 💡 Insights y PDF",
    ])

    # =========================================================================
    # PESTAÑA 1: RESUMEN EJECUTIVO (KPIs + 2 GRÁFICOS PRINCIPALES)
    # =========================================================================
    with tabs[0]:
        k1, k2, k3, k4, k5 = st.columns([0.95, 0.95, 1.05, 0.95, 1.1])
        k1.metric(
            "Registros Crudos",
            f"{ds_meta['registros_crudos_integrados']:,}",
            "Orinoco + NASA + NOAA",
        )
        k2.metric(
            "Dataset Limpio",
            f"{ds_meta['registros']:,}",
            f"2009–2024 · {cal['nulos_despues']} nulos",
        )
        k3.metric(
            "Eficiencia ETL",
            f"{bench['tiempo_etl_completo_seg']} s",
            f"-{bench['reduccion_ram_pct']}% RAM ({bench['peso_parquet_procesado_mb']} MB)",
            delta_color="inverse",
        )
        k4.metric(
            "Modelo Ganador",
            "XGBoost" if "XGBoost" in artifact["summary"]["modelo_ganador"] else artifact["summary"]["modelo_ganador"],
            f"R² = {mod['metricas']['r2']:.4f}",
        )
        k5.metric(
            "MAE (+14 días)",
            f"{mod['metricas']['mae_m']:.4f} m",
            f"-{mod['metricas']['mejora_vs_baseline_pct']:.2f}% vs. Naive ({mod['linea_base']['mae_m']:.4f} m)",
            delta_color="inverse",
        )

        st.divider()

        ctrl1, ctrl2, _ = st.columns([1.15, 0.9, 1.4])
        with ctrl1:
            res_metric = st.radio(
                "Escala del río",
                options=["nivel_minmax_pct", "y"],
                format_func=lambda m: "Capacidad Relativa (%)" if m == "nivel_minmax_pct" else "Nivel Absoluto (m s.n.m.)",
                horizontal=True,
                key="res_metric",
            )
        with ctrl2:
            res_window = st.radio(
                "Ventana",
                options=["2020–2024", "2009–2024"],
                horizontal=True,
                key="res_window",
            )

        g_left, g_right = st.columns([1.45, 1.0])
        with g_left:
            min_year_res = 2020 if res_window == "2020–2024" else 2009
            df_res = df[df["anio"] >= min_year_res][["ds", "estacion_nombre", res_metric]].copy()
            # Remuestreo semanal para renderizado fluido y trazos limpios
            df_res = (
                df_res.set_index("ds")
                .groupby("estacion_nombre", observed=True)[res_metric]
                .resample("W")
                .mean()
                .reset_index()
            )
            y_title = "Capacidad Relativa (%)" if res_metric == "nivel_minmax_pct" else "Nivel (m s.n.m.)"
            chart_river = (
                alt.Chart(df_res)
                .mark_line(strokeWidth=2)
                .encode(
                    x=alt.X(
                        "ds:T",
                        title="Fecha (Año-Mes)",
                        axis=alt.Axis(format="%Y-%m", labelAngle=0, tickCount=8),
                    ),
                    y=alt.Y(f"{res_metric}:Q", title=y_title),
                    color=alt.Color(
                        "estacion_nombre:N",
                        title="Estación",
                        legend=alt.Legend(orient="bottom"),
                        scale=alt.Scale(domain=STATION_COLOR_DOMAIN, range=STATION_COLOR_RANGE),
                    ),
                    tooltip=[
                        alt.Tooltip("ds:T", title="Fecha", format="%Y-%m-%d"),
                        alt.Tooltip("estacion_nombre:N", title="Estación"),
                        alt.Tooltip(f"{res_metric}:Q", title=y_title, format=".2f"),
                    ],
                )
                .properties(height=360, title=f"Dinámica en Cascada del Río Orinoco — {y_title}")
                .interactive()
            )
            st.altair_chart(chart_river, width="stretch")

        with g_right:
            comp_df = pd.DataFrame(mod["comparacion_modelos"])
            comp_df["etiqueta"] = comp_df.apply(
                lambda r: f"{r['mae_m']:.4f} m (-{r['mejora_vs_baseline_pct']:.1f}%)"
                if r["mejora_vs_baseline_pct"] > 0
                else f"{r['mae_m']:.4f} m (Base)",
                axis=1,
            )
            max_mae = float(comp_df["mae_m"].max()) * 1.55
            bars_mae = (
                alt.Chart(comp_df)
                .mark_bar(cornerRadiusEnd=6)
                .encode(
                    x=alt.X(
                        "mae_m:Q",
                        title="Error Medio Absoluto MAE (m) — Menor es mejor",
                        scale=alt.Scale(domain=[0, max_mae]),
                    ),
                    y=alt.Y("modelo:N", sort="x", title="", axis=alt.Axis(labelLimit=160)),
                    color=alt.Color(
                        "modelo:N",
                        legend=None,
                        scale=alt.Scale(domain=MODEL_COLOR_DOMAIN, range=MODEL_COLOR_RANGE),
                    ),
                    tooltip=[
                        alt.Tooltip("modelo:N", title="Modelo"),
                        alt.Tooltip("mae_m:Q", title="MAE (m)", format=".4f"),
                        alt.Tooltip("rmse_m:Q", title="RMSE (m)", format=".4f"),
                        alt.Tooltip("r2:Q", title="R²", format=".4f"),
                        alt.Tooltip("mejora_vs_baseline_pct:Q", title="Mejora vs Naive (%)", format=".2f"),
                    ],
                )
            )
            text_mae = (
                alt.Chart(comp_df)
                .mark_text(align="left", baseline="middle", dx=6, fontWeight="bold")
                .encode(
                    x="mae_m:Q",
                    y=alt.Y("modelo:N", sort="x"),
                    text="etiqueta:N",
                    color=alt.Color(
                        "modelo:N",
                        legend=None,
                        scale=alt.Scale(domain=MODEL_COLOR_DOMAIN, range=MODEL_COLOR_RANGE),
                    ),
                )
            )
            chart_mae = (
                (bars_mae + text_mae)
                .properties(height=360, title="Error MAE a +14 Días (m) — Test 2023–2024")
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(chart_mae, width="stretch")

        with st.expander("ℹ️ Ver síntesis metodológica CRISP-DM y auditoría de calidad ETL (Antes vs. Después)", expanded=False):
            st.markdown(
                "- **Objetivo:** Pronosticar el nivel del río Orinoco a **+14 días** integrando hidrometría local, propagación aguas arriba, precipitación satelital NASA POWER (15 puntos) e índice climático NOAA ONI.\n"
                f"- **Arquitectura (`pandas` vs. `Spark`):** El ETL completo corre en **{bench['tiempo_etl_completo_seg']} s** usando **{bench['ram_dataset_enriquecido_despues_opt_mb']} MB** de RAM (`-60%`), evitando el arranque JVM de Spark.\n"
                f"- **Sin fuga de datos:** Partición temporal estricta (`Train: 2009–2022` [{artifact['summary']['registros_train']:,}], embargo de 14 días, `Test: 2023–2024` [{artifact['summary']['registros_test']:,}]) y lag de 3 meses en ONI."
            )
            prob_df = pd.DataFrame(cal["problemas_detectados"])[
                ["id", "problema", "tecnica_etl", "antes", "despues"]
            ].rename(columns={
                "id": "#",
                "problema": "Problema Detectado",
                "tecnica_etl": "Técnica ETL",
                "antes": "Antes",
                "despues": "Después",
            })
            st.table(prob_df, border="horizontal", width="stretch", hide_index=True)

    # =========================================================================
    # PESTAÑA 2: DATOS Y CUBOS OLAP (GRÁFICOS OLAP ARRIBA + TABLAS COMPACTAS)
    # =========================================================================
    with tabs[1]:
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            sel_stations = st.multiselect(
                "Estaciones",
                options=STATION_COLOR_DOMAIN,
                default=STATION_COLOR_DOMAIN,
            )
        with f2:
            min_yr, max_yr = int(df["anio"].min()), int(df["anio"].max())
            sel_years = st.slider("Años", min_yr, max_yr, (2015, 2024))
        with f3:
            sel_epocas = st.multiselect(
                "Época Hidrológica",
                options=EPOCA_OPTIONS,
                default=EPOCA_OPTIONS,
            )
        with f4:
            sel_enso = st.multiselect(
                "Fase ENSO",
                options=ENSO_OPTIONS,
                default=ENSO_OPTIONS,
            )

        sub_olap1 = olap1[
            (olap1["estacion_nombre"].isin(sel_stations))
            & (olap1["anio"] >= sel_years[0])
            & (olap1["anio"] <= sel_years[1])
            & (olap1["epoca_hidrologica"].isin(sel_epocas))
        ].copy()

        sub_olap2 = olap2[
            (olap2["estacion_nombre"].isin(sel_stations))
            & (olap2["fase_enso"].isin(sel_enso))
            & (olap2["epoca_hidrologica"].isin(sel_epocas))
        ].copy()

        oc_ctrl1, oc_ctrl2 = st.columns(2)
        with oc_ctrl1:
            hm_station = st.selectbox(
                "Estación para Heatmap OLAP 1 (Año × Época)",
                options=sel_stations if sel_stations else STATION_COLOR_DOMAIN,
                index=min(2, max(len(sel_stations) - 1, 0)),
                key="hm_station",
            )
        with oc_ctrl2:
            olap2_metric = st.selectbox(
                "Métrica para Cubo OLAP 2 (Estación × ENSO)",
                options=["capacidad_relativa_pct", "nivel_medio_m", "lluvia_media_diaria_mm"],
                format_func=lambda m: {
                    "capacidad_relativa_pct": "Capacidad Relativa Media (%)",
                    "nivel_medio_m": "Nivel Medio (m s.n.m.)",
                    "lluvia_media_diaria_mm": "Lluvia Media Diaria (mm)",
                }[m],
                key="olap2_metric",
            )

        oc1, oc2 = st.columns(2)
        with oc1:
            hm_df = sub_olap1[sub_olap1["estacion_nombre"] == hm_station].copy()
            heatmap = (
                alt.Chart(hm_df)
                .mark_rect(cornerRadius=3)
                .encode(
                    x=alt.X("anio:O", title="Año", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y(
                        "epoca_hidrologica:N",
                        title="Época Hidrológica",
                        sort=EPOCA_OPTIONS,
                        axis=alt.Axis(labelLimit=180),
                    ),
                    color=alt.Color(
                        "nivel_medio_m:Q",
                        title="Nivel (m s.n.m.)",
                        scale=alt.Scale(scheme="blues"),
                    ),
                    tooltip=[
                        alt.Tooltip("anio:O", title="Año"),
                        alt.Tooltip("epoca_hidrologica:N", title="Época"),
                        alt.Tooltip("nivel_medio_m:Q", title="Nivel Medio (m s.n.m.)", format=".2f"),
                        alt.Tooltip("nivel_max_m:Q", title="Máximo (m s.n.m.)", format=".2f"),
                        alt.Tooltip("lluvia_acum_mm:Q", title="Lluvia Acumulada (mm)", format=".0f"),
                    ],
                )
                .properties(height=340, title=f"Cubo OLAP 1: Nivel Medio en {hm_station} (m s.n.m.)")
            )
            st.altair_chart(heatmap, width="stretch")

        with oc2:
            enso_agg = (
                sub_olap2.groupby(["estacion_nombre", "fase_enso"], as_index=False, observed=True)
                .agg(
                    capacidad_relativa_pct=("capacidad_relativa_pct", "mean"),
                    nivel_medio_m=("nivel_medio_m", "mean"),
                    lluvia_media_diaria_mm=("lluvia_media_diaria_mm", "mean"),
                )
            )
            olap2_label = {
                "capacidad_relativa_pct": "Capacidad Relativa Media (%)",
                "nivel_medio_m": "Nivel Medio (m s.n.m.)",
                "lluvia_media_diaria_mm": "Lluvia Media Diaria (mm)",
            }[olap2_metric]
            enso_bars = (
                alt.Chart(enso_agg)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X(
                        "estacion_nombre:N",
                        title="Estación",
                        sort=STATION_COLOR_DOMAIN,
                        axis=alt.Axis(labelAngle=0, labelLimit=140),
                    ),
                    xOffset="fase_enso:N",
                    y=alt.Y(f"{olap2_metric}:Q", title=olap2_label),
                    color=alt.Color(
                        "fase_enso:N",
                        title="Fase ENSO",
                        scale=alt.Scale(domain=ENSO_COLOR_DOMAIN, range=ENSO_COLOR_RANGE),
                    ),
                    tooltip=[
                        alt.Tooltip("estacion_nombre:N", title="Estación"),
                        alt.Tooltip("fase_enso:N", title="Fase ENSO"),
                        alt.Tooltip("capacidad_relativa_pct:Q", title="Capacidad (%)", format=".1f"),
                        alt.Tooltip("nivel_medio_m:Q", title="Nivel Medio (m)", format=".2f"),
                        alt.Tooltip("lluvia_media_diaria_mm:Q", title="Lluvia diaria (mm)", format=".2f"),
                    ],
                )
                .properties(height=340, title=f"Cubo OLAP 2: {olap2_label} por Fase ENSO")
            )
            st.altair_chart(enso_bars, width="stretch")

        filtered_df = df[
            (df["estacion_nombre"].isin(sel_stations))
            & (df["anio"] >= sel_years[0])
            & (df["anio"] <= sel_years[1])
            & (df["epoca_hidrologica"].isin(sel_epocas))
            & (df["fase_enso"].isin(sel_enso))
        ].sort_values(["ds", "unique_id"], ascending=[False, True])

        t_olap1, t_olap2, t_raw = st.tabs([
            "📋 Cubo OLAP 1 (Año × Época)",
            "📋 Cubo OLAP 2 (ENSO × Trimestre)",
            f"🔍 Dataset Limpio ({len(filtered_df):,} filas filtradas)",
        ])
        with t_olap1:
            olap1_view = sub_olap1[
                [
                    "anio", "estacion_nombre", "region_tramo", "epoca_hidrologica",
                    "registros", "nivel_medio_m", "nivel_max_m", "nivel_min_m",
                    "lluvia_acum_mm", "oni_medio",
                ]
            ].sort_values(["anio", "estacion_nombre", "epoca_hidrologica"], ascending=[False, True, True])
            st.dataframe(
                olap1_view,
                width="stretch",
                hide_index=True,
                height=300,
                column_config={
                    "anio": st.column_config.NumberColumn("Año", format="%d", width="small"),
                    "estacion_nombre": st.column_config.TextColumn("Estación", width="medium"),
                    "region_tramo": st.column_config.TextColumn("Tramo", width="medium"),
                    "epoca_hidrologica": st.column_config.TextColumn("Época Hidrológica", width="medium"),
                    "registros": st.column_config.NumberColumn("Días", format="%d", width="small"),
                    "nivel_medio_m": st.column_config.NumberColumn("Nivel Medio", format="%.2f m"),
                    "nivel_max_m": st.column_config.NumberColumn("Máximo", format="%.2f m"),
                    "nivel_min_m": st.column_config.NumberColumn("Mínimo", format="%.2f m"),
                    "lluvia_acum_mm": st.column_config.NumberColumn("Lluvia Acum.", format="%.0f mm"),
                    "oni_medio": st.column_config.NumberColumn("ONI Medio", format="%.2f °C"),
                },
            )

        with t_olap2:
            olap2_view = sub_olap2[
                [
                    "fase_enso", "estacion_nombre", "trimestre", "epoca_hidrologica",
                    "registros", "nivel_medio_m", "nivel_std_m",
                    "lluvia_media_diaria_mm", "capacidad_relativa_pct",
                ]
            ].sort_values(["fase_enso", "estacion_nombre", "trimestre"])
            st.dataframe(
                olap2_view,
                width="stretch",
                hide_index=True,
                height=300,
                column_config={
                    "fase_enso": st.column_config.TextColumn("Fase ENSO", width="medium"),
                    "estacion_nombre": st.column_config.TextColumn("Estación", width="medium"),
                    "trimestre": st.column_config.TextColumn("Trimestre", width="small"),
                    "epoca_hidrologica": st.column_config.TextColumn("Época Hidrológica", width="medium"),
                    "registros": st.column_config.NumberColumn("Días", format="%d", width="small"),
                    "nivel_medio_m": st.column_config.NumberColumn("Nivel Medio", format="%.2f m"),
                    "nivel_std_m": st.column_config.NumberColumn("Desv. Est.", format="%.2f m"),
                    "lluvia_media_diaria_mm": st.column_config.NumberColumn("Lluvia Media", format="%.2f mm/d"),
                    "capacidad_relativa_pct": st.column_config.ProgressColumn(
                        "Capacidad Relativa",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
                },
            )

        with t_raw:
            display_cols = [
                "ds", "estacion_nombre", "y", "target_y_t14",
                "delta_y_3d", "lluvia_local_14d", "lluvia_cuenca_14d", "oni",
                "epoca_hidrologica", "fase_enso", "nivel_minmax_pct", "quality",
            ]
            st.dataframe(
                filtered_df[display_cols].head(500),
                width="stretch",
                hide_index=True,
                height=300,
                column_config={
                    "ds": st.column_config.DateColumn("Fecha", format="YYYY-MM-DD"),
                    "estacion_nombre": st.column_config.TextColumn("Estación", width="medium"),
                    "y": st.column_config.NumberColumn("Nivel y(t)", format="%.2f m"),
                    "target_y_t14": st.column_config.NumberColumn("Objetivo y(t+14)", format="%.2f m"),
                    "delta_y_3d": st.column_config.NumberColumn("Cambio 3d", format="%+.2f m"),
                    "lluvia_local_14d": st.column_config.NumberColumn("Lluvia Local 14d", format="%.1f mm"),
                    "lluvia_cuenca_14d": st.column_config.NumberColumn("Lluvia Cuenca 14d", format="%.1f mm"),
                    "oni": st.column_config.NumberColumn("ONI", format="%.2f °C"),
                    "epoca_hidrologica": st.column_config.TextColumn("Época", width="medium"),
                    "fase_enso": st.column_config.TextColumn("Fase ENSO", width="medium"),
                    "nivel_minmax_pct": st.column_config.ProgressColumn(
                        "Capacidad (%)",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
                    "quality": st.column_config.TextColumn("Calidad"),
                },
            )

    # =========================================================================
    # PESTAÑA 3: VISUALIZACIONES INTERACTIVAS + SELECTOR LIMPIO DE 6 FIGURAS
    # =========================================================================
    with tabs[2]:
        v1, v2, v3 = st.columns([1.1, 1.1, 1.0])
        with v1:
            viz_station = st.selectbox(
                "Estación (o Todas)",
                options=["ALL", *list(STATION_LABELS.keys())],
                format_func=lambda k: "Todas las estaciones" if k == "ALL" else STATION_LABELS[k],
                index=3,  # Ciudad Bolívar por defecto
                key="viz_station",
            )
        with v2:
            viz_years = st.slider("Ventana Temporal", 2009, 2024, (2016, 2024), key="viz_yr")
        with v3:
            viz_metric = st.selectbox(
                "Variable a Explorar",
                options=list(VARIABLE_META.keys()),
                format_func=lambda m: VARIABLE_META[m],
                key="viz_metric",
            )

        sub_viz = df[(df["anio"] >= viz_years[0]) & (df["anio"] <= viz_years[1])].copy()
        if viz_station != "ALL":
            sub_viz = sub_viz[sub_viz["unique_id"] == viz_station]

        sub_viz = sub_viz.sort_values("ds")
        y_label = VARIABLE_META[viz_metric]
        active_stations = [s for s in STATION_COLOR_DOMAIN if s in set(sub_viz["estacion_nombre"])]
        active_colors = [
            STATION_COLOR_RANGE[STATION_COLOR_DOMAIN.index(s)] for s in active_stations
        ]

        explorer_chart = (
            alt.Chart(sub_viz)
            .mark_line(strokeWidth=1.8)
            .encode(
                x=alt.X(
                    "ds:T",
                    title="Fecha (Año-Mes)",
                    axis=alt.Axis(format="%Y-%m", labelAngle=0, tickCount=8),
                ),
                y=alt.Y(f"{viz_metric}:Q", title=y_label),
                color=alt.Color(
                    "estacion_nombre:N",
                    title="Estación",
                    legend=alt.Legend(orient="bottom") if len(active_stations) > 1 else None,
                    scale=alt.Scale(domain=active_stations, range=active_colors),
                ),
                tooltip=[
                    alt.Tooltip("ds:T", title="Fecha", format="%Y-%m-%d"),
                    alt.Tooltip("estacion_nombre:N", title="Estación"),
                    alt.Tooltip(f"{viz_metric}:Q", title=y_label, format=".2f"),
                    alt.Tooltip("epoca_hidrologica:N", title="Época"),
                    alt.Tooltip("fase_enso:N", title="ENSO"),
                ],
            )
            .properties(height=360, title=f"Explorador Interactivo: {y_label} ({viz_years[0]}–{viz_years[1]})")
            .interactive()
        )
        st.altair_chart(explorer_chart, width="stretch")

        st.divider()
        st.subheader("Galería de Figuras del Estudio (Actividad 7)")
        figs_list = mongo_doc["visualizaciones_generadas"]
        fig_options = [f"Fig {idx + 1}: {item['id'].upper()}" for idx, item in enumerate(figs_list)]
        fig_labels = {
            0: "Fig 1 · Serie Temporal",
            1: "Fig 2 · Épocas",
            2: "Fig 3 · ENSO",
            3: "Fig 4 · Propagación",
            4: "Fig 5 · Heatmap OLAP",
            5: "Fig 6 · Modelo ML",
        }
        chosen_idx = st.segmented_control(
            "Selecciona una figura para inspeccionarla:",
            options=list(range(len(figs_list))),
            format_func=lambda i: fig_labels.get(i, fig_options[i]),
            default=0,
        )
        if chosen_idx is None:
            chosen_idx = 0

        f_item = figs_list[chosen_idx]
        img_file = PROJECT_ROOT / f_item["archivo"]
        st.markdown(f"**{f_item['titulo']}** · *{f_item['tipo']}*")
        st.info(f"**Insight clave:** {f_item['insight']}")
        if img_file.exists():
            st.image(str(img_file), width="stretch")

    # =========================================================================
    # PESTAÑA 4: MODELO ML (GRÁFICOS DE DESEMPEÑO + SIMULADOR SIMPLIFICADO)
    # =========================================================================
    with tabs[3]:
        p_col_sel, _ = st.columns([1.0, 2.0])
        with p_col_sel:
            eval_station = st.selectbox(
                "Estación en Evaluación (Test 2023–2024)",
                options=list(STATION_LABELS.keys()),
                format_func=lambda k: STATION_LABELS[k],
                index=2,  # Ciudad Bolívar
                key="eval_station",
            )

        sub_preds = preds_test[preds_test["unique_id"] == eval_station].sort_values("ds").copy()
        preds_long = sub_preds.melt(
            id_vars=["ds"],
            value_vars=["target_y_t14", "pred_ganador", "pred_naive"],
            var_name="Serie",
            value_name="Nivel_m",
        )
        serie_map = {
            "target_y_t14": "Real y(t+14)",
            "pred_ganador": f"Predicción {artifact['summary']['modelo_ganador']}",
            "pred_naive": "Línea Base (Naive)",
        }
        preds_long["Serie"] = preds_long["Serie"].map(serie_map)

        pred_chart = (
            alt.Chart(preds_long)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X(
                    "ds:T",
                    title="Fecha de Emisión del Pronóstico (Año-Mes)",
                    axis=alt.Axis(format="%Y-%m", labelAngle=0, tickCount=8),
                ),
                y=alt.Y("Nivel_m:Q", title="Nivel a +14 días (m s.n.m.)"),
                color=alt.Color(
                    "Serie:N",
                    legend=alt.Legend(orient="bottom"),
                    scale=alt.Scale(
                        domain=[
                            "Real y(t+14)",
                            f"Predicción {artifact['summary']['modelo_ganador']}",
                            "Línea Base (Naive)",
                        ],
                        range=["#56B4E9", "#D55E00", "#AAAAAA"],
                    ),
                ),
                strokeDash=alt.StrokeDash(
                    "Serie:N",
                    scale=alt.Scale(
                        domain=[
                            "Real y(t+14)",
                            f"Predicción {artifact['summary']['modelo_ganador']}",
                            "Línea Base (Naive)",
                        ],
                        range=[[1, 0], [1, 0], [4, 4]],
                    ),
                ),
                tooltip=[
                    alt.Tooltip("ds:T", title="Fecha", format="%Y-%m-%d"),
                    alt.Tooltip("Serie:N", title="Serie"),
                    alt.Tooltip("Nivel_m:Q", title="Nivel (m s.n.m.)", format=".2f"),
                ],
            )
            .properties(
                height=340,
                title=f"Real vs. Predicho a +14 Días — {STATION_LABELS[eval_station]} (m s.n.m., 2023–2024)",
            )
            .interactive()
        )
        st.altair_chart(pred_chart, width="stretch")

        mc1, mc2 = st.columns(2)
        with mc1:
            st_df = pd.DataFrame(mod["metricas_por_estacion"])
            st_long = st_df.melt(
                id_vars=["estacion"],
                value_vars=["mae_baseline_m", "mae_modelo_m"],
                var_name="Tipo",
                value_name="MAE_m",
            )
            st_long["Tipo"] = st_long["Tipo"].map({
                "mae_baseline_m": "Línea Base (Naive)",
                "mae_modelo_m": artifact["summary"]["modelo_ganador"],
            })
            st_chart = (
                alt.Chart(st_long)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X(
                        "estacion:N",
                        title="Estación",
                        sort=STATION_COLOR_DOMAIN,
                        axis=alt.Axis(labelAngle=0, labelLimit=130),
                    ),
                    xOffset="Tipo:N",
                    y=alt.Y("MAE_m:Q", title="Error MAE (m)"),
                    color=alt.Color(
                        "Tipo:N",
                        title="Modelo",
                        legend=alt.Legend(orient="bottom"),
                        scale=alt.Scale(
                            domain=["Línea Base (Naive)", artifact["summary"]["modelo_ganador"]],
                            range=["#AAAAAA", "#D55E00"],
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip("estacion:N", title="Estación"),
                        alt.Tooltip("Tipo:N", title="Modelo"),
                        alt.Tooltip("MAE_m:Q", title="MAE (m)", format=".4f"),
                    ],
                )
                .properties(height=320, title="Error MAE por Estación (metros): Ganador vs. Naive")
            )
            st.altair_chart(st_chart, width="stretch")

        with mc2:
            imp_df = pd.DataFrame(mod["variables_importantes"])
            imp_chart = (
                alt.Chart(imp_df)
                .mark_bar(cornerRadiusEnd=5, color="#0072B2")
                .encode(
                    x=alt.X("importancia_pct:Q", title="Importancia Relativa (%)"),
                    y=alt.Y(
                        "variable:N",
                        sort="-x",
                        title="Variable Predictora",
                        axis=alt.Axis(labelOverlap=False, labelLimit=160),
                    ),
                    tooltip=[
                        alt.Tooltip("variable:N", title="Variable"),
                        alt.Tooltip("importancia_pct:Q", title="Importancia (%)", format=".2f"),
                        alt.Tooltip("interpretacion_negocio:N", title="Significado"),
                    ],
                )
                .properties(height=320, title="Top 10 Variables Más Influyentes — XGBoost (%)")
            )
            st.altair_chart(imp_chart, width="stretch")

        with st.expander("📋 Desempeño detallado por Modelo y por Estación (Tarjetas y Tablas)", expanded=False):
            tab_m, tab_s = st.tabs(["🏆 Comparación de Modelos", "📍 Desempeño por Estación"])

            with tab_m:
                comp_tbl = pd.DataFrame(mod["comparacion_modelos"])
                winner_name = artifact["summary"]["modelo_ganador"]
                m_cards = st.columns(len(comp_tbl))
                for idx, row in comp_tbl.iterrows():
                    with m_cards[idx]:
                        with st.container(border=True):
                            tag = " 🏆" if row["modelo"] == winner_name else ""
                            st.markdown(f"**{row['modelo']}{tag}**")
                            st.caption(row["tipo"])
                            delta_txt = (
                                f"-{row['mejora_vs_baseline_pct']:.2f}% vs Naive"
                                if row["mejora_vs_baseline_pct"] > 0
                                else "Referencia base"
                            )
                            st.metric("MAE", f"{row['mae_m']:.4f} m", delta_txt)
                            st.caption(f"RMSE: `{row['rmse_m']:.4f} m` · R²: `{row['r2']:.4f}`")

                st.dataframe(
                    comp_tbl,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "modelo": st.column_config.TextColumn("Modelo", width="medium"),
                        "tipo": st.column_config.TextColumn("Enfoque Algorítmico", width="large"),
                        "mae_m": st.column_config.NumberColumn("MAE", format="%.4f m"),
                        "rmse_m": st.column_config.NumberColumn("RMSE", format="%.4f m"),
                        "r2": st.column_config.NumberColumn("R²", format="%.4f"),
                        "mejora_vs_baseline_pct": st.column_config.ProgressColumn(
                            "Mejora vs. Naive",
                            format="%.2f%%",
                            min_value=0,
                            max_value=100,
                        ),
                    },
                )

            with tab_s:
                s_cards = st.columns(len(st_df))
                for idx, (_, row) in enumerate(st_df.iterrows()):
                    with s_cards[idx]:
                        with st.container(border=True):
                            st.markdown(f"**{row['estacion']}**")
                            st.caption(f"{int(row['registros_test'])} días de prueba")
                            st.metric(
                                "MAE Ganador",
                                f"{row['mae_modelo_m']:.4f} m",
                                f"-{row['mejora_mae_pct']:.2f}% vs Naive ({row['mae_baseline_m']:.3f} m)",
                            )
                            st.caption(f"RMSE: `{row['rmse_modelo_m']:.4f} m` · R²: `{row['r2_modelo']:.4f}`")

                st.dataframe(
                    st_df.drop(columns=["unique_id"], errors="ignore"),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "estacion": st.column_config.TextColumn("Estación"),
                        "registros_test": st.column_config.NumberColumn("Obs. Test", format="%d"),
                        "mae_baseline_m": st.column_config.NumberColumn("MAE Naive", format="%.4f m"),
                        "mae_modelo_m": st.column_config.NumberColumn("MAE Ganador", format="%.4f m"),
                        "rmse_modelo_m": st.column_config.NumberColumn("RMSE Ganador", format="%.4f m"),
                        "r2_modelo": st.column_config.NumberColumn("R²", format="%.4f"),
                        "mejora_mae_pct": st.column_config.ProgressColumn(
                            "Mejora MAE",
                            format="%.2f%%",
                            min_value=0,
                            max_value=100,
                        ),
                    },
                )

        st.divider()
        st.subheader("🔮 Simulador de Pronóstico a +14 Días")

        s1, s2, s3 = st.columns([1.0, 1.0, 1.3])
        with s1:
            form_station = st.selectbox(
                "Estación Objetivo",
                options=list(STATION_LABELS.keys()),
                format_func=lambda k: STATION_LABELS[k],
                index=2,
                key="sim_station",
            )
        with s2:
            model_choice = st.selectbox(
                "Modelo Entrenado",
                options=list(artifact["all_pipelines"].keys()),
                index=2,
                key="sim_model",
            )
        with s3:
            scenario_choice = st.radio(
                "Escenario Rápido",
                options=[
                    "Último registro",
                    "🌊 Crecida (Aguas Altas / La Niña)",
                    "☀️ Estiaje (Aguas Bajas / El Niño)",
                ],
                horizontal=True,
                key="sim_scenario",
            )

        defaults, scenario_note = _get_scenario_values(
            df, artifact["defaults_by_station"], form_station, scenario_choice
        )
        st.caption(f"Valores iniciales cargados desde: **{scenario_note}**")

        # Variable aguas arriba más representativa según la estación
        if form_station == "caicara":
            up_key, up_label = "ayacucho_lag_7", "Nivel Puerto Ayacucho hace 7d (m)"
        else:
            up_key, up_label = "caicara_lag_7", "Nivel Caicara del Orinoco hace 7d (m)"

        key_prefix = f"sim_{form_station}_{scenario_choice}"

        with st.form("prediction_form"):
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                in_y = st.number_input(
                    "1. Nivel actual en la estación y(t) [m]",
                    value=round(float(defaults["y"]), 2),
                    step=0.1,
                    key=f"{key_prefix}_y",
                )
                in_delta_3d = st.number_input(
                    "2. Cambio últimos 3 días Δy(3d) [m]",
                    value=round(float(defaults["delta_y_3d"]), 3),
                    step=0.05,
                    key=f"{key_prefix}_d3",
                )
            with fc2:
                in_upstream = st.number_input(
                    f"3. {up_label}",
                    value=round(float(defaults[up_key]), 2),
                    step=0.1,
                    key=f"{key_prefix}_up",
                )
                in_lluvia_cuenca = st.number_input(
                    "4. Lluvia media en la cuenca 14d [mm]",
                    value=round(float(defaults["lluvia_cuenca_14d"]), 1),
                    step=5.0,
                    key=f"{key_prefix}_rain_c",
                )
            with fc3:
                epoca_idx = EPOCA_OPTIONS.index(defaults["epoca_hidrologica"]) if defaults["epoca_hidrologica"] in EPOCA_OPTIONS else 3
                enso_idx = ENSO_OPTIONS.index(defaults["fase_enso"]) if defaults["fase_enso"] in ENSO_OPTIONS else 1
                in_epoca = st.selectbox(
                    "5. Época Hidrológica",
                    options=EPOCA_OPTIONS,
                    index=epoca_idx,
                    key=f"{key_prefix}_epoca",
                )
                in_enso = st.selectbox(
                    "Fase Climática ENSO",
                    options=ENSO_OPTIONS,
                    index=enso_idx,
                    key=f"{key_prefix}_enso",
                )

            with st.expander("⚙️ Ajustes avanzados (rezagos adicionales, lluvia local y calendario)", expanded=False):
                a1, a2, a3 = st.columns(3)
                with a1:
                    in_y_lag7 = st.number_input(
                        "Nivel hace 7 días y(t-7) [m]",
                        value=round(float(defaults["y_lag_7"]), 2),
                        step=0.1,
                        key=f"{key_prefix}_yl7",
                    )
                    in_y_lag14 = st.number_input(
                        "Nivel hace 14 días y(t-14) [m]",
                        value=round(float(defaults["y_lag_14"]), 2),
                        step=0.1,
                        key=f"{key_prefix}_yl14",
                    )
                    in_delta_1d = st.number_input(
                        "Cambio últimas 24 h Δy(1d) [m]",
                        value=round(float(defaults["delta_y_1d"]), 3),
                        step=0.05,
                        key=f"{key_prefix}_d1",
                    )
                with a2:
                    if up_key == "ayacucho_lag_7":
                        in_ayacucho_7 = in_upstream
                        in_caicara_7 = st.number_input(
                            "Nivel Caicara (t-7 días) [m]",
                            value=round(float(defaults["caicara_lag_7"]), 2),
                            step=0.1,
                            key=f"{key_prefix}_cai7",
                        )
                    else:
                        in_caicara_7 = in_upstream
                        in_ayacucho_7 = st.number_input(
                            "Nivel Ayacucho (t-7 días) [m]",
                            value=round(float(defaults["ayacucho_lag_7"]), 2),
                            step=0.1,
                            key=f"{key_prefix}_aya7",
                        )
                    in_ayacucho_14 = st.number_input(
                        "Nivel Ayacucho (t-14 días) [m]",
                        value=round(float(defaults["ayacucho_lag_14"]), 2),
                        step=0.1,
                        key=f"{key_prefix}_aya14",
                    )
                    in_cb_1 = st.number_input(
                        "Nivel Ciudad Bolívar (t-1 día) [m]",
                        value=round(float(defaults["ciudad_bolivar_lag_1"]), 2),
                        step=0.1,
                        key=f"{key_prefix}_cb1",
                    )
                with a3:
                    in_lluvia_loc = st.number_input(
                        "Lluvia local acumulada 14d [mm]",
                        value=round(float(defaults["lluvia_local_14d"]), 1),
                        step=5.0,
                        key=f"{key_prefix}_rain_l",
                    )
                    in_oni = st.number_input(
                        "Índice NOAA ONI (°C)",
                        value=round(float(defaults["oni"]), 2),
                        step=0.1,
                        key=f"{key_prefix}_oni",
                    )
                    in_doy = st.slider(
                        "Día del Año (1–365)",
                        1,
                        365,
                        int(defaults.get("doy", 220)),
                        key=f"{key_prefix}_doy",
                    )

            submitted = st.form_submit_button("Calcular Pronóstico a +14 Días", type="primary")

        if submitted:
            sin_doy_val = float(np.sin(2.0 * np.pi * in_doy / 365.25))
            cos_doy_val = float(np.cos(2.0 * np.pi * in_doy / 365.25))
            delta_7d_val = float(in_y - in_y_lag7)

            input_df = pd.DataFrame([{
                "y": in_y,
                "y_lag_7": in_y_lag7,
                "y_lag_14": in_y_lag14,
                "delta_y_1d": in_delta_1d,
                "delta_y_3d": in_delta_3d,
                "delta_y_7d": delta_7d_val,
                "ayacucho_lag_7": in_ayacucho_7,
                "ayacucho_lag_14": in_ayacucho_14,
                "caicara_lag_7": in_caicara_7,
                "ciudad_bolivar_lag_1": in_cb_1,
                "lluvia_local_14d": in_lluvia_loc,
                "lluvia_cuenca_14d": in_lluvia_cuenca,
                "oni": in_oni,
                "sin_doy": sin_doy_val,
                "cos_doy": cos_doy_val,
                "unique_id": form_station,
                "epoca_hidrologica": in_epoca,
                "fase_enso": in_enso,
            }])

            chosen_pipe = artifact["all_pipelines"][model_choice]
            pred_t14 = float(chosen_pipe.predict(input_df)[0])
            expected_change = pred_t14 - in_y

            res_c1, res_c2, res_c3 = st.columns(3)
            res_c1.metric(
                f"Nivel Actual y(t) en {STATION_LABELS[form_station]}",
                f"{in_y:.2f} m s.n.m.",
            )
            res_c2.metric(
                f"Pronóstico a +14 Días ({model_choice})",
                f"{pred_t14:.2f} m s.n.m.",
                f"{expected_change:+.2f} m en 2 semanas",
                delta_color="off",
            )
            res_c3.metric(
                "Línea Base (Naive)",
                f"{in_y:.2f} m s.n.m.",
                "Sin cambio (0.00 m)",
                delta_color="off",
            )

            sim_df = pd.DataFrame([
                {
                    "Horizonte": "1. Nivel Actual (Hoy)",
                    "Nivel_m": in_y,
                    "Etiqueta": f"{in_y:.2f} m",
                },
                {
                    "Horizonte": "2. Pronóstico (+14 días)",
                    "Nivel_m": pred_t14,
                    "Etiqueta": f"{pred_t14:.2f} m ({expected_change:+.2f} m)",
                },
            ])
            max_sim = max(in_y, pred_t14, 1.0) * 1.32
            sim_bars = (
                alt.Chart(sim_df)
                .mark_bar(cornerRadiusEnd=6)
                .encode(
                    x=alt.X(
                        "Nivel_m:Q",
                        title="Nivel del Río (m s.n.m.)",
                        scale=alt.Scale(domain=[0, max_sim]),
                    ),
                    y=alt.Y(
                        "Horizonte:N",
                        title="",
                        sort=["1. Nivel Actual (Hoy)", "2. Pronóstico (+14 días)"],
                        scale=alt.Scale(paddingInner=0.35),
                        axis=alt.Axis(labelLimit=180, labelOverlap=False),
                    ),
                    color=alt.Color(
                        "Horizonte:N",
                        legend=None,
                        scale=alt.Scale(
                            domain=["1. Nivel Actual (Hoy)", "2. Pronóstico (+14 días)"],
                            range=["#56B4E9", "#FF7B29"],
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip("Horizonte:N", title="Horizonte"),
                        alt.Tooltip("Nivel_m:Q", title="Nivel (m s.n.m.)", format=".2f"),
                    ],
                )
            )
            sim_text = (
                alt.Chart(sim_df)
                .mark_text(align="left", baseline="middle", dx=8, fontWeight="bold")
                .encode(
                    x="Nivel_m:Q",
                    y=alt.Y("Horizonte:N", sort=["1. Nivel Actual (Hoy)", "2. Pronóstico (+14 días)"]),
                    text="Etiqueta:N",
                    color=alt.Color(
                        "Horizonte:N",
                        legend=None,
                        scale=alt.Scale(
                            domain=["1. Nivel Actual (Hoy)", "2. Pronóstico (+14 días)"],
                            range=["#56B4E9", "#FF7B29"],
                        ),
                    ),
                )
            )
            sim_chart = (
                (sim_bars + sim_text)
                .properties(
                    height=175,
                    title=f"Comparación: Nivel Actual vs. Pronóstico +14 Días en {STATION_LABELS[form_station]} (m s.n.m.)",
                )
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(sim_chart, width="stretch")

    # =========================================================================
    # PESTAÑA 5: INSIGHTS (MONGODB) + GLOSARIO + REPORTE PDF
    # =========================================================================
    with tabs[4]:
        top_left, top_right = st.columns([2.2, 1.0])
        with top_left:
            st.subheader("Insights y Recomendaciones (MongoDB)")
            st.caption(
                f"Documento `_id: {mongo_doc.get('_id')}` leído desde `{mongo_backend}` "
                "(Base: `orinoco_bigdata` · Colección: `analisis_crispdm`)."
            )
        with top_right:
            if not PDF_REPORT_PATH.exists():
                from src.generate_pdf import generate_pdf_from_mongodb
                generate_pdf_from_mongodb()
            pdf_bytes = PDF_REPORT_PATH.read_bytes()
            st.download_button(
                label="📥 Descargar Reporte PDF Automático",
                data=pdf_bytes,
                file_name="reporte_bigdata_orinoco.pdf",
                mime="application/pdf",
                type="primary",
                width="stretch",
            )

        ins_col, rec_col = st.columns(2)
        with ins_col:
            st.markdown("##### 🔍 Insights Clave (clic para ampliar)")
            for ins in mongo_doc["insights"]:
                badge = _impact_badge(ins["impacto_negocio"])
                with st.expander(f"[{ins['id']}] {badge} · {ins['titulo']}", expanded=False):
                    st.caption(f"Tema: {ins['tema']}")
                    st.write(ins["descripcion"])

        with rec_col:
            st.markdown("##### 🎯 Recomendaciones Accionables (clic para ampliar)")
            for rec in mongo_doc["recomendaciones"]:
                with st.expander(f"[{rec['id']}] {rec['destinatario']}", expanded=False):
                    st.markdown(f"**Acción:** {rec['recomendacion']}")
                    st.success(f"**Beneficio:** {rec['beneficio_esperado']}")

        st.divider()
        with st.expander("📖 Diccionario / Glosario Hidrometeorológico de Variables (18 predictores + Target)", expanded=False):
            glos_df = _build_glossary_dataframe().copy()
            glos_df["Variable"] = glos_df["Variable"].map(lambda v: f"`{v}`")
            st.table(glos_df, border="horizontal", width="stretch", height=440, hide_index=True)

        with st.expander("🗄️ Consultas NoSQL a MongoDB (JSON)", expanded=False):
            q_info = store_and_query_mongodb(mongo_doc)
            st.markdown("**Consulta 1 (`find_one` métricas del modelo):**")
            st.json(q_info["consulta_1_metricas_modelo"])
            st.markdown("**Consulta 2 (`aggregate` insights Crítico/Alto):**")
            st.json(q_info["consulta_2_insights_criticos"])
            st.markdown("**Consulta 3 (`find_one` calidad y recomendaciones):**")
            st.json(q_info["consulta_3_calidad_y_recomendaciones"])


if __name__ == "__main__":
    main()
