"""Generador y ejecutor automático del Notebook maestro del Proyecto Final de Big Data.

Construye notebooks/Armas_Eduardo_BigData_ProyectoFinal.ipynb con todas las secciones
exigidas en el PDF y lo ejecuta de principio a fin en un kernel limpio (nbclient) para
que todas las celdas queden ejecutadas mostrando sus resultados, tablas y gráficas inline.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOK_PATH = NOTEBOOKS_DIR / "Armas_Eduardo_BigData_ProyectoFinal.ipynb"


def build_and_execute_notebook() -> Path:
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    nb = new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {
        "name": "python",
        "version": "3.12",
    }

    cells = []

    # =========================================================================
    # CELDA 0: PORTADA, TÍTULO DEL TEMA Y DATASET
    # =========================================================================
    cells.append(
        new_markdown_cell(
            """# Proyecto Final — Gerencia y Almacenamiento en Big Data
## Pronóstico Hidrológico Multi-Estación del Río Orinoco a +14 Días mediante Pipeline CRISP-DM, Machine Learning, MongoDB y Docker

- **Institución:** Fundación Universitaria Los Libertadores
- **Asignatura:** Gerencia y Almacenamiento en Big Data
- **Autor:** Eduardo Armas
- **Dataset Principal:** Niveles Hidrométricos Diarios del Río Orinoco (`Puerto Ayacucho`, `Caicara del Orinoco`, `Ciudad Bolívar`, `Palúa`) integrados con Precipitación Diaria Satelital **NASA POWER (MERRA-2)** en 15 puntos de la cuenca y el Índice Climático **NOAA CPC Oceanic Niño Index (ONI)** (2009–2024; **271.196 registros crudos integrados**).

---"""
        )
    )

    # =========================================================================
    # SECCIÓN 1: BUSINESS UNDERSTANDING Y SELECCIÓN DEL DATASET
    # =========================================================================
    cells.append(
        new_markdown_cell(
            """## 1. Business Understanding y Selección del Dataset

### 1.1 Descripción de las Fuentes de Datos y Justificación del Volumen
El proyecto aborda un problema real de hidrología operacional y gestión de riesgo de desastres en la cuenca del **río Orinoco** (tercer río más caudaloso del mundo). Integramos tres fuentes públicas, oficiales y verificables:

1. **Niveles Limnimétricos Diarios del Río Orinoco (`DATOS RIO ORINOCO.xlsx` / `dataset_combinado.csv`):** Series diarias medidas en 4 estaciones ordenadas de aguas arriba hacia aguas abajo: **Puerto Ayacucho → Caicara del Orinoco → Ciudad Bolívar → Palúa**. En la ventana oficial validada `2009-01-01` a `2024-12-31` comprende **5.844 días × 4 estaciones = 23.376 registros válidos** (23.736 registros crudos incluyendo el trimestre inicial de 2025, y 74.732 registros en el archivo histórico desde 1974).
2. **Precipitación Diaria Satelital NASA POWER (`PRECTOTCORR` - MERRA-2):** 15 estaciones satelitales distribuidas en las subcuencas del Alto/Medio Orinoco, Apure, Meta, Guaviare, Ventuari, Caura y Bajo Caroní (`1981-01-01` a `2025-12-31`), aportando **16.436 días × 15 puntos = 246.540 registros meteorológicos diarios**.
3. **Índice Climático NOAA CPC Oceanic Niño Index (`noaa_oni.ascii.txt`):** **920 registros trimestrales móviles** desde 1950 que cuantifican la anomalía térmica del Pacífico ecuatorial (El Niño / La Niña).

> **Justificación técnica escrita sobre el volumen del dataset (< 500.000 registros):**
> La guía establece un volumen preferible superior a 500.000 registros o su justificación escrita cuando el dominio real tenga menos. En hidrología fluvial oficial, las reglas limnimétricas registran **una medición diaria** (frecuencia `1D`), pues la inercia hidráulica de una cuenca de $989.000\\text{ km}^2$ varía pocos centímetros por hora. Al integrar las 15 estaciones satelitales diarias de NASA POWER, el índice NOAA ONI y las 4 estaciones limnimétricas del río Orinoco, el repositorio procesa **271.196 registros crudos reales** (y más de **345.000 registros** contabilizando el histórico desde 1974). Superar artificialmente los 500.000 registros requeriría interpolar niveles a frecuencia horaria o generar ruido sintético, lo cual está explícitamente prohibido por la restricción de **no usar datos sintéticos**.

### 1.2 Pregunta de Negocio y Preguntas Específicas
- **Pregunta Principal de Negocio:** *¿Con qué precisión podemos anticipar el nivel del río Orinoco con **14 días de anticipación** ($t+14$) en las estaciones estratégicas de navegación y control de inundaciones (Ciudad Bolívar y Palúa) a partir de la propagación aguas arriba, la precipitación en la cuenca y el fenómeno El Niño/La Niña (ENSO), para emitir alertas tempranas hidrologísticas?*
- **Preguntas Específicas que responde el análisis:**
  1. **Dinámica espacio-temporal y ENSO:** *¿Cómo se propaga la onda de crecida desde Puerto Ayacucho y Caicara hacia Ciudad Bolívar y Palúa, y cómo varía la capacidad relativa del río según la Época Hidrológica y la Fase ENSO (La Niña, Neutro, El Niño)?*
  2. **Valor predictivo del Machine Learning:** *¿En qué porcentaje reducen el error de pronóstico a 14 días (MAE y RMSE) los modelos supervisados (Ridge, Random Forest y XGBoost) frente a una Línea Base de persistencia (`Naive`, que asume $\\hat{y}_{t+14} = y_t$)?*
  3. **Interpretabilidad operativa:** *¿Cuáles son las variables hidrométricas y climáticas de mayor importancia para predecir el nivel a dos semanas y qué recomendaciones operativas se derivan para Protección Civil y la navegación fluvial?*

### 1.3 Tarea de Machine Learning y Criterio de Éxito
- **Tarea elegida:** **Predicción (Regresión supervisada)** de la variable continua `target_y_t14` (nivel del río en metros sobre el nivel del mar exactamente 14 días después, $y_{t+14}$).
- **Criterio de éxito:** Superar a la Línea Base (`Naive Persistence`) logrando una **reducción del MAE y RMSE de al menos 45%** y un coeficiente de determinación **$R^2 > 0.97$** en el período de prueba reservado (`2023–2024`), garantizando cero fuga de información (*data leakage*)."""
        )
    )

    cells.append(
        new_code_cell(
            """import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image as IPImage, display

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.download_data import ensure_raw_data

ingest_summary = ensure_raw_data()
print("=== 1. INVENTARIO Y METADATOS DE DESCARGA EN data/raw/ ===")
print(json.dumps(ingest_summary, indent=2, ensure_ascii=False))"""
        )
    )

    # =========================================================================
    # SECCIÓN 2: DECISIÓN DE ARQUITECTURA: PANDAS O SPARK
    # =========================================================================
    cells.append(
        new_markdown_cell(
            """## 2. Decisión de Arquitectura: `pandas` vs. `Spark`

Retomando los criterios de la **Actividad 4**, la elección entre procesamiento en un solo nodo (`pandas` + `PyArrow`) y procesamiento distribuido (`Apache Spark`) debe basarse en mediciones empíricas de volumen, huella en memoria RAM, latencia de entrada/salida y sobrecarga de planificación (*scheduling overhead*):

| Criterio de Arquitectura (Actividad 4) | Medición Real en Nuestro Dataset | Decisión Técnica |
|---|---|---|
| **Volumen total de registros crudos** | `271.196` filas integradas (`246.540` NASA POWER + `23.736` niveles + `920` ONI) | Cabe holgadamente en la RAM de un contenedor estándar (< 50 MB). |
| **Peso en disco (Crudo vs. Parquet)** | `~4.43 MB` en CSV/XLSX crudos $\\rightarrow$ `1.53 MB` en Parquet comprimido (`snappy`) | El motor columnar C++ de `PyArrow` lee y escribe el Parquet en milisegundos. |
| **Optimización de tipos en `pandas`** | Conversión de `float64` $\\rightarrow$ `float32` y `object` $\\rightarrow$ `category` reduce la RAM del DataFrame enriquecido en **75,4%** (de `17,88 MB` a `4,40 MB`). | Huella de memoria mínima sin necesidad de particionamiento RDD/DataFrame distribuido. |
| **Sobrecarga de JVM en PySpark** | Iniciar una `SparkSession` local tarda `~4–8 s` y consume `>600 MB` de RAM solo para la JVM, mientras que el **ETL completo en `pandas` termina en `< 5 segundos`**. | **Se elige `pandas` + `PyArrow Parquet`** por menor latencia, menor consumo de RAM en Docker y vectorización nativa C/Cython. |"""
        )
    )

    cells.append(
        new_code_cell(
            """from src.etl_pipeline import run_etl_pipeline, PROCESSED_DIR

etl_results = run_etl_pipeline()
bench_df = pd.DataFrame([etl_results["benchmark_arquitectura"]]).T
bench_df.columns = ["Valor Medido"]
print("=== 2. BENCHMARK CUANTITATIVO DE ARQUITECTURA (PANDAS + PYARROW) ===")
display(bench_df)"""
        )
    )

    # =========================================================================
    # SECCIÓN 3: ETL CON CRISP-DM (DATA UNDERSTANDING + DATA PREPARATION)
    # =========================================================================
    cells.append(
        new_markdown_cell(
            """## 3. ETL con CRISP-DM (Data Understanding + Data Preparation)

### 3.1 Exploración Inicial y Tabla de Problemas de Calidad Detectados
Antes de aplicar cualquier transformación, auditamos las fuentes crudas mediante reglas de calidad específicas del dominio hidrológico:
1. **Filas fuera de la ventana válida (`2025-01-01` a `2025-03-31`):** El archivo `DATOS RIO ORINOCO.xlsx` incluye fechas hasta marzo de 2025 con celdas vacías a partir del 25 de febrero (**360 registros** fuera del período oficial 2009–2024).
2. **Nulos en años bisiestos (`missing`):** La estación `palua` carece de medición el **29 de febrero de 2012, 2016 y 2020** (**3 registros nulos**).
3. **Sensores congelados / Rachas planas (`suspect_flat`):** Un río caudaloso no permanece exactamente en el mismo milímetro durante 10 o más días seguidos. Se detectaron **12 días idénticos en Puerto Ayacucho** (`2019-03-07` al `2019-03-18`) y **19 días idénticos en Ciudad Bolívar** (`2020-03-24` al `2020-04-11`), sumando **31 registros de nulos disfrazados**.
4. **Salto diario atípico (`suspect_jump`):** En `palua` el `2021-10-29` se registra un salto aislado de $+1,06\\text{ m}$ en 24 horas (**1 registro**, etiquetado para trazabilidad sin alterar la observación).
5. **Granularidad dispersa en NASA POWER:** **246.540 registros** en 15 archivos CSV con encabezado de 9 líneas y código centinela `-999` para datos faltantes.
6. **Riesgo de fuga temporal en NOAA ONI y escalas heterogéneas:** Las estaciones tienen alturas base muy distintas (Ayacucho $\\approx 45,9\\text{ m}$ vs. Palúa $\\approx 5,8\\text{ m}$) y el índice trimestral ONI se publica con 3 meses de retraso oficial.

### 3.2 Aplicación de las 4 Técnicas ETL y Creación de Variables Derivadas
1. **Limpieza:** Recorte al período oficial `2009–2024` (de `23.736` a `23.376` registros), conversión de las rachas planas (`31` filas) a `NaN` e **interpolación lineal temporal** junto con los `3` días bisiestos (**34 valores imputados**, pasando de **34 anomalías/huecos a 0 nulos**).
2. **Normalización:** Estandarización de fechas ISO (`ds`), optimización de tipos (`float32`, `category`) y cálculo de `nivel_zscore` y `nivel_minmax_pct` por estación **usando únicamente media, desviación, mínimo y máximo del conjunto de entrenamiento (`2009–2022`)** para evitar *data leakage*.
3. **Agregación:** Consolidación de los `246.540` registros diarios de las 15 estaciones de NASA POWER en precipitación media por subcuenca (`lluvia_cuenca_orinoco_mm`, `lluvia_cuenca_caroni_mm`) y acumulados móviles de 7 y 14 días (`lluvia_cuenca_7d`, `lluvia_cuenca_14d`), además de 2 cubos OLAP.
4. **Enriquecimiento:** Cruce con lluvia NASA POWER, índice NOAA ONI desplazado 3 meses y creación de **más de 10 variables derivadas** para el modelo:
   - **Velocidades y aceleración del río:** `delta_y_1d`, `delta_y_3d`, `delta_y_7d`, `media_movil_7d`.
   - **Rezagos de propagación aguas arriba:** `ayacucho_lag_7`, `ayacucho_lag_14`, `caicara_lag_4`, `caicara_lag_7`, `ciudad_bolivar_lag_1`.
   - **Armónicos de Fourier del ciclo anual:** `sin_doy`, `cos_doy`, `sin_doy_2`, `cos_doy_2`.
   - **Dimensiones OLAP:** `epoca_hidrologica`, `fase_enso`, `region_tramo`, `trimestre`."""
        )
    )

    cells.append(
        new_code_cell(
            """print("=== 3.1 ESTADÍSTICAS DESCRIPTIVAS POR ESTACIÓN ANTES DE LIMPIAR ===")
stats_before_df = pd.DataFrame(etl_results["estadisticas_descriptivas"]).T
display(stats_before_df)

print("\\n=== 3.2 TABLA DE PROBLEMAS DE CALIDAD Y CONTEOS ANTES / DESPUÉS (4 TÉCNICAS ETL) ===")
problemas_df = pd.DataFrame(etl_results["calidad_datos"]["problemas_detectados"])
display(problemas_df)

df_clean = pd.read_parquet(PROCESSED_DIR / "orinoco_limpio.parquet")
print(f"\\nDataset limpio guardado en data/processed/orinoco_limpio.parquet: {df_clean.shape[0]:,} filas x {df_clean.shape[1]} columnas.")
print("Nulos restantes en la variable nivel (y):", int(df_clean["y"].isna().sum()))
display(df_clean[["ds", "unique_id", "estacion_nombre", "y", "target_y_t14", "delta_y_3d", "lluvia_cuenca_14d", "oni", "epoca_hidrologica", "fase_enso", "nivel_zscore", "quality", "imputed"]].head(10))"""
        )
    )

    # =========================================================================
    # SECCIÓN 4: ANÁLISIS EXPLORATORIO, CUBOS OLAP Y VISUALIZACIONES
    # =========================================================================
    cells.append(
        new_markdown_cell(
            """## 4. Análisis Exploratorio, Agregaciones OLAP y Visualizaciones

Se construyeron **2 agregaciones multidimensionales tipo OLAP** que alimentan los filtros interactivos (*slice, dice, drill-down*) del dashboard en Streamlit:
1. **Cubo OLAP 1 (`Año × Estación × Región × Época Hidrológica`):** 256 celdas con nivel medio, máximo, mínimo, Z-score y lluvia acumulada.
2. **Cubo OLAP 2 (`Fase ENSO × Estación × Trimestre × Época Hidrológica`):** 60 celdas que contrastan el comportamiento del río bajo episodios de *La Niña*, *Neutro* y *El Niño*."""
        )
    )

    cells.append(
        new_code_cell(
            """olap1_df = pd.read_parquet(PROCESSED_DIR / "olap_anio_estacion_epoca.parquet")
olap2_df = pd.read_parquet(PROCESSED_DIR / "olap_enso_estacion_trimestre.parquet")

print(f"=== CUBO OLAP 1: Año × Estación × Época Hidrológica ({len(olap1_df)} celdas) ===")
display(olap1_df.head(8))

print(f"\\n=== CUBO OLAP 2: Fase ENSO × Estación × Trimestre ({len(olap2_df)} celdas) ===")
display(olap2_df.head(9))"""
        )
    )

    # =========================================================================
    # SECCIÓN 5: MACHINE LEARNING (MODELING + EVALUATION)
    # =========================================================================
    cells.append(
        new_markdown_cell(
            """## 5. Machine Learning (Modeling + Evaluation)

### 5.1 Prevención de Fuga de Información (*Data Leakage*) y Partición Temporal
En series de tiempo con horizonte de pronóstico $h = 14$ días, una validación aleatoria (*random k-fold*) filtraría el futuro al entrenamiento. Aplicamos tres barreras estrictas contra el *data leakage*:
1. **Partición cronológica con embargo de 14 días:**
   - **Entrenamiento (`Train`):** `2009-01-29` a `2022-12-17` (`20.340` registros). Su último valor objetivo $y_{t+14}$ cae el `2022-12-31`, sin tocar jamás un solo día de 2023.
   - **Prueba (`Test`):** `2023-01-01` a `2024-12-17` (`2.868` registros evaluados a horizonte $t+14$).
2. **Transformadores encapsulados en `Pipeline`:** `StandardScaler` y `OneHotEncoder` se ajustan (`fit`) exclusivamente sobre `X_train`.
3. **Modelado físico del incremento a 14 días (`DeltaHorizonPipeline`):** Cada regresor estima el cambio esperado $\\widehat{\\Delta y}_{t+14} = y_{t+14} - y_t$ y reconstruye el nivel $\\hat{y}_{t+14} = y_t + \\widehat{\\Delta y}_{t+14}$, permitiendo que los árboles de decisión comparen la dinámica de ascenso/descenso entre estaciones con distinta altitud sobre el nivel del mar."""
        )
    )

    cells.append(
        new_code_cell(
            """from src.ml_pipeline import run_ml_pipeline, MODEL_ARTIFACT_PATH
from src.etl_pipeline import generate_eda_visualizations

ml_results = run_ml_pipeline()

print("=== 5.1 COMPARACIÓN DE MODELOS CONTRA LÍNEA BASE (TEST 2023-2024) ===")
comp_models_df = pd.DataFrame(ml_results["comparacion_modelos"])
display(comp_models_df)

print(f"\\nModelo Ganador seleccionado: {ml_results['modelo_ganador']} -> Artefacto guardado en: {MODEL_ARTIFACT_PATH.relative_to(PROJECT_ROOT)}")

print("\\n=== 5.2 DESEMPEÑO DEL MODELO GANADOR DESGLOSADO POR ESTACIÓN ===")
station_metrics_df = pd.DataFrame(ml_results["metricas_por_estacion"])
display(station_metrics_df)

print("\\n=== 5.3 INTERPRETACIÓN DEL MODELO: VARIABLES MÁS IMPORTANTES EN LENGUAJE DE NEGOCIO ===")
imp_vars_df = pd.DataFrame(ml_results["variables_importantes"])
display(imp_vars_df)"""
        )
    )

    cells.append(
        new_markdown_cell(
            """### 5.4 Visualización de las 6 Gráficas del Estudio (Principios de la Actividad 7)
Todas las gráficas aplican la paleta accesible *Okabe-Ito* (apta para daltonismo), alto *data-ink ratio* (sin bordes superiores/derechos redundantes), títulos con *insight* cuantitativo y ejes rotulados en español con unidades físicas."""
        )
    )

    cells.append(
        new_code_cell(
            """figures_meta = generate_eda_visualizations(ml_results)
for fig_item in figures_meta:
    print(f"\\n--- {fig_item['id'].upper()}: {fig_item['titulo']} ---")
    print(f"Tipo: {fig_item['tipo']} | Insight: {fig_item['insight']}")
    display(IPImage(filename=str(PROJECT_ROOT / fig_item["archivo"]), width=820))"""
        )
    )

    # =========================================================================
    # SECCIÓN 6: INSIGHTS EN MONGODB Y 3 CONSULTAS NOSQL
    # =========================================================================
    cells.append(
        new_markdown_cell(
            """## 6. Almacenamiento Híbrido e Insights en MongoDB

Siguiendo la arquitectura híbrida de la **Actividad 1**, los datos masivos tabulares residen en **Parquet** (`data/processed/orinoco_limpio.parquet`) mientras que los metadatos, el diagnóstico de calidad, las agregaciones OLAP, las métricas del modelo, los **6 insights** y las **4 recomendaciones estratégicas** se almacenan como documento JSON/BSON en **MongoDB** (`orinoco_bigdata.analisis_crispdm`).

A continuación insertamos el documento, ejecutamos **3 consultas a MongoDB** y cerramos explícitamente la conexión a la base de datos."""
        )
    )

    cells.append(
        new_code_cell(
            """from src.mongodb_store import build_analysis_document, store_and_query_mongodb

analysis_doc = build_analysis_document(
    ingest_info=ingest_summary,
    etl_info=etl_results,
    ml_info=ml_results,
    figs_info=figures_meta,
    autores=["Eduardo Armas - Fundación Universitaria Los Libertadores"],
)

mongo_queries = store_and_query_mongodb(analysis_doc)
print(f"Backend MongoDB utilizado: {mongo_queries['backend']}")
print(f"Base de datos: {mongo_queries['database']} | Colección: {mongo_queries['collection']} | _id: {mongo_queries['document_id']}")

print("\\n=== CONSULTA MONGODB #1: Recuperar métricas del modelo ganador vs. Línea Base ===")
print(json.dumps(mongo_queries["consulta_1_metricas_modelo"], indent=2, ensure_ascii=False))

print("\\n=== CONSULTA MONGODB #2: Pipeline de agregación ($unwind + $match) filtrando Insights de impacto Crítico/Alto ===")
display(pd.DataFrame(mongo_queries["consulta_2_insights_criticos"]))

print("\\n=== CONSULTA MONGODB #3: Resumen de calidad de datos resuelta y Recomendaciones de Negocio ===")
display(pd.DataFrame(mongo_queries["consulta_3_calidad_y_recomendaciones"]["recomendaciones"]))"""
        )
    )

    # =========================================================================
    # SECCIÓN 7: GENERACIÓN AUTOMÁTICA DEL REPORTE PDF DESDE MONGODB
    # =========================================================================
    cells.append(
        new_markdown_cell(
            """## 7. Generación Automática del Reporte PDF desde MongoDB y Cierre de Conexiones

Finalmente, generamos de forma **100% automática** el reporte ejecutivo en formato PDF (`reports/reporte_bigdata_orinoco.pdf`) leyendo el documento recién consultado desde MongoDB mediante `ReportLab`, y verificamos el cierre limpio de la conexión a la base de datos."""
        )
    )

    cells.append(
        new_code_cell(
            """from src.generate_pdf import generate_pdf_from_mongodb
from src.mongodb_store import get_mongo_connection

pdf_file = generate_pdf_from_mongodb()
print(f"Reporte PDF generado automáticamente desde MongoDB: {pdf_file.relative_to(PROJECT_ROOT)}")
print(f"Tamaño del archivo PDF: {pdf_file.stat().st_size / 1024:.2f} KB")

# Verificación final y cierre explícito de conexión a base de datos (Requisito técnico página 6)
client_check, backend_name = get_mongo_connection()
client_check.close()
print(f"Conexión a base de datos ({backend_name}) cerrada correctamente al finalizar el notebook.")"""
        )
    )

    nb.cells = cells
    with NOTEBOOK_PATH.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print("Ejecutando todas las celdas del notebook en un kernel limpio...")
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOKS_DIR)}},
    )
    client.execute()

    with NOTEBOOK_PATH.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"Notebook ejecutado y guardado con todas sus salidas en: {NOTEBOOK_PATH}")
    return NOTEBOOK_PATH


if __name__ == "__main__":
    build_and_execute_notebook()
