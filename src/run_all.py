"""Orquestador integral del pipeline de Big Data (Ingesta -> ETL -> ML -> Gráficas -> MongoDB -> PDF)."""

from __future__ import annotations

from typing import Any

from src.download_data import ensure_raw_data
from src.etl_pipeline import generate_eda_visualizations, run_etl_pipeline
from src.generate_pdf import generate_pdf_from_mongodb
from src.ml_pipeline import run_ml_pipeline
from src.mongodb_store import build_analysis_document, store_and_query_mongodb


def execute_full_pipeline() -> dict[str, Any]:
    """Ejecuta de punta a punta las fases CRISP-DM, persiste en MongoDB y genera el PDF."""
    print("[1/5] Verificando e ingiriendo datos crudos en data/raw/...")
    ingest_info = ensure_raw_data()

    print("[2/5] Ejecutando ETL CRISP-DM con pandas y construyendo cubos OLAP...")
    etl_info = run_etl_pipeline()

    print("[3/5] Entrenando y evaluando modelos de Machine Learning vs. Línea Base...")
    ml_info = run_ml_pipeline()

    print("[4/5] Generando las 6 visualizaciones con principios de la Actividad 7...")
    figs_info = generate_eda_visualizations(ml_info)

    print("[5/5] Almacenando documento e insights en MongoDB y generando Reporte PDF...")
    doc = build_analysis_document(ingest_info, etl_info, ml_info, figs_info)
    mongo_info = store_and_query_mongodb(doc)
    pdf_path = generate_pdf_from_mongodb()

    print(f"Pipeline finalizado. PDF disponible en: {pdf_path}")
    return {
        "ingest": ingest_info,
        "etl": etl_info,
        "ml": ml_info,
        "figures": figs_info,
        "mongodb": mongo_info,
        "pdf_path": str(pdf_path),
    }


if __name__ == "__main__":
    execute_full_pipeline()
