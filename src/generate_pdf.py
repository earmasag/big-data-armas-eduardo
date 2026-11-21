"""Fase 7: Generación automática del Reporte PDF a partir de MongoDB con ReportLab.

Lee el documento almacenado en MongoDB (colección analisis_crispdm) y genera
reports/reporte_bigdata_orinoco.pdf con portada, resumen del dataset, tabla de
problemas de calidad antes/después, tablas de agregación OLAP, las 6 gráficas
incrustadas, resultados comparativos del modelo de ML, insights y recomendaciones.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.download_data import PROJECT_ROOT
from src.mongodb_store import load_analysis_from_mongodb

REPORTS_DIR = PROJECT_ROOT / "reports"
PDF_REPORT_PATH = REPORTS_DIR / "reporte_bigdata_orinoco.pdf"


def _register_academic_fonts() -> tuple[str, str]:
    """Registra Arial (o fallback compatible) y retorna (font_regular, font_bold)."""
    candidates = [
        (
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/arialbd.ttf"),
            Path("C:/Windows/Fonts/ariali.ttf"),
            Path("C:/Windows/Fonts/arialbi.ttf"),
        ),
        (
            Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"),
            Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf"),
            Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Italic.ttf"),
            Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Bold_Italic.ttf"),
        ),
        (
            Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
            Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
            Path("/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf"),
            Path("/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf"),
        ),
    ]
    for reg, bold, italic, bold_italic in candidates:
        if reg.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont("ArialAcademic", str(reg)))
            pdfmetrics.registerFont(TTFont("ArialAcademic-Bold", str(bold)))
            if italic.exists():
                pdfmetrics.registerFont(TTFont("ArialAcademic-Italic", str(italic)))
            if bold_italic.exists():
                pdfmetrics.registerFont(TTFont("ArialAcademic-BoldItalic", str(bold_italic)))
            pdfmetrics.registerFontFamily(
                "ArialAcademic",
                normal="ArialAcademic",
                bold="ArialAcademic-Bold",
                italic="ArialAcademic-Italic" if italic.exists() else "ArialAcademic",
                boldItalic="ArialAcademic-BoldItalic" if bold_italic.exists() else "ArialAcademic-Bold",
            )
            return "ArialAcademic", "ArialAcademic-Bold"
    return "Helvetica", "Helvetica-Bold"


def _styled_table(
    data: list[list[Any]],
    col_widths: list[float] | None = None,
    font_regular: str = "Helvetica",
    font_bold: str = "Helvetica-Bold",
) -> Table:
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1B365D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), font_bold),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
            ("TOPPADDING", (0, 0), (-1, 0), 6),
            ("BACKGROUND", (0, 1), (-1, -1), colors.white),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#1A202C")),
            ("FONTNAME", (0, 1), (-1, -1), font_regular),
            ("FONTSIZE", (0, 1), (-1, -1), 8.5),
            ("LINEABOVE", (0, 0), (-1, 0), 1.0, colors.HexColor("#1B365D")),
            ("LINEBELOW", (0, 0), (-1, 0), 1.0, colors.HexColor("#1B365D")),
            ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.HexColor("#1B365D")),
            ("GRID", (0, 1), (-1, -1), 0.3, colors.HexColor("#CBD5E0")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 1), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
        ])
    )
    return tbl


def generate_pdf_from_mongodb(output_path: Path = PDF_REPORT_PATH) -> Path:
    """Construye el reporte PDF automáticamente a partir del documento almacenado en MongoDB."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    doc_data, backend_label = load_analysis_from_mongodb()
    font_regular, font_bold = _register_academic_fonts()

    styles = getSampleStyleSheet()
    cover_institution = ParagraphStyle(
        "CoverInstitution",
        parent=styles["Normal"],
        fontName=font_bold,
        fontSize=14,
        leading=20,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1B365D"),
        spaceAfter=6,
    )
    cover_course = ParagraphStyle(
        "CoverCourse",
        parent=styles["Normal"],
        fontName=font_regular,
        fontSize=12,
        leading=18,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=12,
    )
    cover_title = ParagraphStyle(
        "CoverTitle",
        parent=styles["Title"],
        fontName=font_bold,
        fontSize=18,
        leading=24,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#111827"),
        spaceAfter=14,
    )
    cover_subtitle = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName=font_regular,
        fontSize=12,
        leading=18,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=12,
    )
    cover_meta = ParagraphStyle(
        "CoverMeta",
        parent=styles["Normal"],
        fontName=font_regular,
        fontSize=12,
        leading=18,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1A202C"),
        spaceAfter=4,
    )
    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Heading1"],
        fontName=font_bold,
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#1B365D"),
        spaceBefore=14,
        spaceAfter=8,
    )
    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Heading2"],
        fontName=font_bold,
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#2C5282"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyAcademic",
        parent=styles["Normal"],
        fontName=font_regular,
        fontSize=12,
        leading=18,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#1A202C"),
        spaceAfter=8,
    )
    cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName=font_regular,
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1A202C"),
    )

    story: list[Any] = []
    meta = doc_data["metadata"]
    ds_meta = meta["dataset"]
    bench = meta["benchmark_arquitectura"]
    cal = doc_data["calidad_datos"]
    mod = doc_data["modelo"]

    # =========================================================================
    # PORTADA ACADÉMICA SENCILLA (PÁGINA 1)
    # =========================================================================
    story.append(Spacer(1, 1.3 * inch))
    story.append(Paragraph("FUNDACIÓN UNIVERSITARIA LOS LIBERTADORES", cover_institution))
    story.append(Paragraph("Gerencia y Almacenamiento en Big Data", cover_course))
    story.append(Spacer(1, 1.4 * inch))
    story.append(Paragraph(meta.get("proyecto", "Pipeline Integral de Big Data — Río Orinoco"), cover_title))
    story.append(
        Paragraph(
            "Informe Técnico de Proyecto Final: Pipeline CRISP-DM, Almacenamiento Híbrido "
            "en MongoDB y Modelado Predictivo de Machine Learning",
            cover_subtitle,
        )
    )
    story.append(Spacer(1, 2.2 * inch))
    story.append(Paragraph(f"<b>Autor:</b> {', '.join(meta['autores'])}", cover_meta))
    story.append(Paragraph(f"<b>Fecha:</b> {str(meta['fecha_analisis'])[:10]}", cover_meta))
    story.append(PageBreak())

    # =========================================================================
    # 1. RESUMEN DEL DATASET Y CALIDAD DE DATOS (CRISP-DM)
    # =========================================================================
    story.append(Paragraph("1. Resumen del Dataset y Calidad de Datos (CRISP-DM)", h1_style))
    story.append(
        Paragraph(
            "Este informe presenta los resultados del pipeline integral de Big Data aplicado al "
            "monitoreo hidrológico del río Orinoco, integrando almacenamiento híbrido en MongoDB, "
            "procesamiento analítico multidimensional (OLAP) y modelado predictivo de Machine Learning "
            "a un horizonte de 14 días.",
            body_style,
        )
    )

    meta_rows = [
        ["Atributo del Proyecto", "Valor Registrado en MongoDB"],
        ["ID del Documento MongoDB", str(doc_data.get("_id", "orinoco_crispdm_2026"))],
        ["Motor NoSQL de Origen", backend_label],
        ["Fecha de Generación", meta["fecha_analisis"]],
        ["Autores", ", ".join(meta["autores"])],
        ["Fuente Oficial de Datos", Paragraph(ds_meta["fuente"], cell_style)],
        ["Registros Crudos Integrados", Paragraph(f"{ds_meta['registros_crudos_integrados']:,} registros (Niveles + 15 estaciones NASA POWER + NOAA ONI)", cell_style)],
        ["Registros Limpios (2009-2024)", f"{ds_meta['registros']:,} registros diarios x {ds_meta['columnas']} columnas"],
        ["Motor ETL y Justificación", Paragraph(f"{bench['herramienta_elegida']} — ETL en {bench['tiempo_etl_completo_seg']} s, Parquet de {bench['peso_parquet_procesado_mb']} MB (-{bench['reduccion_ram_pct']}% RAM)", cell_style)],
        ["Modelo Ganador (+14 días)", Paragraph(f"{mod['algoritmo']} | MAE: {mod['metricas']['mae_m']} m | RMSE: {mod['metricas']['rmse_m']} m | R²: {mod['metricas']['r2']}", cell_style)],
    ]
    story.append(_styled_table(meta_rows, col_widths=[2.1 * inch, 4.7 * inch], font_regular=font_regular, font_bold=font_bold))
    story.append(Spacer(1, 0.14 * inch))

    story.append(Paragraph("Estadísticas Descriptivas Previas a la Limpieza:", h2_style))
    story.append(
        Paragraph(
            f"Antes de limpiar se auditaron los registros crudos detectando {cal['nulos_antes']} nulos en "
            f"años bisiestos, {cal['rachas_planas_corregidas']} días con sensores congelados (rachas planas), "
            f"{cal['registros_eliminados']} filas fuera de la ventana válida 2009–2024 y "
            f"{cal['centinelas_nasa_corregidos']} valores centinela (-999) en precipitación satelital. "
            f"Tras aplicar las 4 técnicas ETL quedaron {cal['nulos_despues']} nulos.",
            body_style,
        )
    )

    stats_rows = [["Estación", "Registros", "Nulos Antes", "Media (m)", "DE (m)", "Mín (m)", "Mediana (m)", "Máx (m)"]]
    for st, st_vals in doc_data["estadisticas_descriptivas"].items():
        stats_rows.append([
            st,
            str(st_vals["registros"]),
            str(st_vals["nulos_directos"]),
            f"{st_vals['media_m']:.2f}",
            f"{st_vals['std_m']:.2f}",
            f"{st_vals['min_m']:.2f}",
            f"{st_vals['mediana_m']:.2f}",
            f"{st_vals['max_m']:.2f}",
        ])
    story.append(_styled_table(stats_rows, col_widths=[1.15 * inch, 0.75 * inch, 0.8 * inch, 0.8 * inch, 0.75 * inch, 0.75 * inch, 0.9 * inch, 0.8 * inch], font_regular=font_regular, font_bold=font_bold))
    story.append(Spacer(1, 0.14 * inch))

    story.append(Paragraph("Diagnóstico y Conteos Antes / Después de las 4 Técnicas ETL:", h2_style))
    prob_rows = [["#", "Problema Detectado", "Técnica ETL", "Conteo Antes", "Conteo Después"]]
    for p in cal["problemas_detectados"]:
        prob_rows.append([
            str(p["id"]),
            Paragraph(p["problema"], cell_style),
            Paragraph(p["tecnica_etl"], cell_style),
            Paragraph(p["antes"], cell_style),
            Paragraph(p["despues"], cell_style),
        ])
    story.append(_styled_table(prob_rows, col_widths=[0.3 * inch, 2.1 * inch, 1.4 * inch, 1.45 * inch, 1.55 * inch], font_regular=font_regular, font_bold=font_bold))

    story.append(PageBreak())

    # =========================================================================
    # 2. TABLAS DE AGREGACIÓN MULTIDIMENSIONAL (OLAP)
    # =========================================================================
    story.append(Paragraph("2. Agregaciones Multidimensionales Tipo OLAP", h1_style))
    for ag in doc_data["agregaciones"]:
        story.append(Paragraph(f"{ag['nombre_cubo']} ({ag['total_celdas']} celdas agregadas)", h2_style))
        if "resumen_epoca_estacion" in ag:
            olap_rows = [["Estación", "Época Hidrológica", "Nivel Medio (m)", "Nivel Máx (m)", "Lluvia Media (mm/d)"]]
            for r in ag["resumen_epoca_estacion"][:12]:
                olap_rows.append([
                    r["estacion_nombre"],
                    r["epoca_hidrologica"],
                    f"{r['nivel_medio_m']:.2f}",
                    f"{r['nivel_max_m']:.2f}",
                    f"{r['lluvia_diaria_mm']:.2f}",
                ])
            story.append(_styled_table(olap_rows, col_widths=[1.4 * inch, 1.7 * inch, 1.2 * inch, 1.2 * inch, 1.3 * inch], font_regular=font_regular, font_bold=font_bold))
            story.append(Spacer(1, 0.12 * inch))
        elif "resumen_enso_estacion" in ag:
            olap2_rows = [["Estación", "Fase ENSO", "Días", "Nivel Medio (m)", "Capacidad (%)", "Lluvia (mm/d)"]]
            for r in ag["resumen_enso_estacion"]:
                olap2_rows.append([
                    r["estacion_nombre"],
                    r["fase_enso"],
                    str(r["dias"]),
                    f"{r['nivel_medio_m']:.2f}",
                    f"{r['capacidad_media_pct']:.1f}%",
                    f"{r['lluvia_diaria_mm']:.2f}",
                ])
            story.append(_styled_table(olap2_rows, col_widths=[1.4 * inch, 1.35 * inch, 0.75 * inch, 1.1 * inch, 1.1 * inch, 1.1 * inch], font_regular=font_regular, font_bold=font_bold))
            story.append(Spacer(1, 0.12 * inch))

    # =========================================================================
    # 3. RESULTADOS DEL MODELO DE MACHINE LEARNING
    # =========================================================================
    story.append(Paragraph("3. Resultados de Machine Learning (Modeling & Evaluation)", h1_style))
    story.append(Paragraph(f"<b>Prevención de Fuga de Información (Data Leakage):</b> {mod['prevencion_data_leakage']}", body_style))

    comp_rows = [["Modelo Evaluado", "Familia / Arquitectura", "MAE (m)", "RMSE (m)", "R²", "Mejora vs. Baseline"]]
    for m in mod["comparacion_modelos"]:
        comp_rows.append([
            Paragraph(m["modelo"], cell_style),
            Paragraph(m["tipo"], cell_style),
            f"{m['mae_m']:.4f}",
            f"{m['rmse_m']:.4f}",
            f"{m['r2']:.4f}",
            f"+{m['mejora_vs_baseline_pct']:.2f}%",
        ])
    story.append(_styled_table(comp_rows, col_widths=[1.45 * inch, 1.85 * inch, 0.8 * inch, 0.85 * inch, 0.75 * inch, 1.1 * inch], font_regular=font_regular, font_bold=font_bold))
    story.append(Spacer(1, 0.14 * inch))

    story.append(Paragraph("Desempeño del Modelo Ganador por Estación (Test 2023–2024):", h2_style))
    st_rows = [["Estación", "Obs. Test", "MAE Baseline (m)", "MAE Modelo (m)", "RMSE Modelo (m)", "R²", "Mejora MAE"]]
    for s in mod["metricas_por_estacion"]:
        st_rows.append([
            s["estacion"],
            str(s["registros_test"]),
            f"{s['mae_baseline_m']:.4f}",
            f"{s['mae_modelo_m']:.4f}",
            f"{s['rmse_modelo_m']:.4f}",
            f"{s['r2_modelo']:.4f}",
            f"+{s['mejora_mae_pct']:.2f}%",
        ])
    story.append(_styled_table(st_rows, col_widths=[1.4 * inch, 0.75 * inch, 1.05 * inch, 1.0 * inch, 1.0 * inch, 0.7 * inch, 0.9 * inch], font_regular=font_regular, font_bold=font_bold))
    story.append(Spacer(1, 0.14 * inch))

    story.append(Paragraph("Variables Más Importantes en Lenguaje de Negocio:", h2_style))
    imp_rows = [["Variable", "Importancia (%)", "Interpretación Hidrológica / de Negocio"]]
    for v in mod["variables_importantes"][:8]:
        imp_rows.append([
            v["variable"],
            f"{v['importancia_pct']:.2f}%",
            Paragraph(v["interpretacion_negocio"], cell_style),
        ])
    story.append(_styled_table(imp_rows, col_widths=[1.4 * inch, 1.05 * inch, 4.35 * inch], font_regular=font_regular, font_bold=font_bold))

    story.append(PageBreak())

    # =========================================================================
    # 4. VISUALIZACIONES GENERADAS (MÍNIMO 4 EXIGIDAS, SE INCLUYEN LAS 6)
    # =========================================================================
    story.append(Paragraph("4. Visualizaciones del Análisis Exploratorio y del Modelo", h1_style))
    for i, fig_meta in enumerate(doc_data["visualizaciones_generadas"], start=1):
        img_path = PROJECT_ROOT / fig_meta["archivo"]
        story.append(Paragraph(f"Figura {i}: {fig_meta['titulo']}", h2_style))
        story.append(Paragraph(f"<b>Insight:</b> {fig_meta['insight']}", body_style))
        if img_path.exists():
            story.append(Image(str(img_path), width=6.4 * inch, height=2.65 * inch))
        story.append(Spacer(1, 0.1 * inch))
        if i % 2 == 0 and i < len(doc_data["visualizaciones_generadas"]):
            story.append(PageBreak())

    story.append(PageBreak())

    # =========================================================================
    # 5. INSIGHTS Y RECOMENDACIONES ALMACENADOS EN MONGODB
    # =========================================================================
    story.append(Paragraph("5. Insights y Recomendaciones Estratégicas (desde MongoDB)", h1_style))
    story.append(Paragraph("Hallazgos Clave (Insights):", h2_style))
    ins_rows = [["ID", "Tema", "Impacto", "Hallazgo y Evidencia Cuantitativa"]]
    for ins in doc_data["insights"]:
        ins_rows.append([
            ins["id"],
            Paragraph(ins["tema"], cell_style),
            ins["impacto_negocio"],
            Paragraph(f"<b>{ins['titulo']}:</b> {ins['descripcion']}", cell_style),
        ])
    story.append(_styled_table(ins_rows, col_widths=[0.55 * inch, 1.15 * inch, 0.65 * inch, 4.45 * inch], font_regular=font_regular, font_bold=font_bold))
    story.append(Spacer(1, 0.16 * inch))

    story.append(Paragraph("Recomendaciones Accionables de Negocio:", h2_style))
    rec_rows = [["ID", "Destinatario", "Acción Recomendada", "Beneficio Esperado"]]
    for rec in doc_data["recomendaciones"]:
        rec_rows.append([
            rec["id"],
            Paragraph(rec["destinatario"], cell_style),
            Paragraph(rec["recomendacion"], cell_style),
            Paragraph(rec["beneficio_esperado"], cell_style),
        ])
    story.append(_styled_table(rec_rows, col_widths=[0.55 * inch, 1.45 * inch, 2.8 * inch, 2.0 * inch], font_regular=font_regular, font_bold=font_bold))

    def _draw_footer(canvas_obj: Any, doc_obj: Any) -> None:
        canvas_obj.saveState()
        canvas_obj.setFont(font_regular, 9)
        canvas_obj.setFillColor(colors.HexColor("#4A5568"))
        canvas_obj.drawCentredString(letter[0] / 2.0, 0.45 * inch, f"Página {doc_obj.page}")
        canvas_obj.restoreState()

    pdf = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="Reporte Final Big Data - Pronóstico Río Orinoco",
        author=", ".join(meta["autores"]),
    )
    pdf.build(story, onLaterPages=_draw_footer)
    return output_path


if __name__ == "__main__":
    out = generate_pdf_from_mongodb()
    print(f"=== Reporte PDF generado exitosamente en: {out} ({out.stat().st_size / 1024:.1f} KB) ===")
