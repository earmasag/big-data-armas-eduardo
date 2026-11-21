"""Script de descarga e ingesta de datos crudos en data/raw/ con rutas relativas.

Fuentes oficiales integradas:
1. Niveles limnimétricos diarios del río Orinoco (4 estaciones en cascada:
   Ayacucho, Caicara, Ciudad Bolívar y Palúa) - DATOS RIO ORINOCO.xlsx y dataset_combinado.csv.
2. NASA POWER (MERRA-2 PRECTOTCORR): precipitación diaria satelital (1981-2025)
   en 15 puntos de la cuenca del Orinoco y Bajo Caroní (246.540 registros).
   API oficial: https://power.larc.nasa.gov/api/temporal/daily/point
3. NOAA CPC Oceanic Niño Index (ONI): anomalías trimestrales SST El Niño / La Niña (1950-2025).
   URL oficial: https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
"""

from __future__ import annotations

import shutil
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PARENT_REPO_ROOT = PROJECT_ROOT.parent

NASA_POWER_API = "https://power.larc.nasa.gov/api/temporal/daily/point"
NOAA_ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
ORINOCO_REPO_URL = (
    "https://github.com/Cesar2926/Proyecto-IA-Orinoco/tree/main/data/raw"
)

NASA_POINTS: dict[str, dict[str, float]] = {
    "ayacucho": {"lat": 5.66, "lon": -67.62},
    "caicara": {"lat": 7.61, "lon": -66.15},
    "ciudad_bolivar": {"lat": 8.09, "lon": -63.56},
    "palua": {"lat": 8.32, "lon": -62.75},
    "caroni_kavanayen": {"lat": 5.62, "lon": -61.78},
    "caroni_uriman": {"lat": 5.35, "lon": -62.75},
    "caroni_canaima": {"lat": 6.24, "lon": -62.85},
    "paragua_la_paragua": {"lat": 6.83, "lon": -63.33},
    "guri_embalse": {"lat": 7.50, "lon": -62.85},
    "apure_san_fernando": {"lat": 7.89, "lon": -67.47},
    "meta_orocue": {"lat": 4.79, "lon": -71.34},
    "meta_puerto_carreno": {"lat": 6.19, "lon": -67.49},
    "guaviare_san_jose": {"lat": 2.57, "lon": -72.64},
    "ventuari_manapiare": {"lat": 5.33, "lon": -66.05},
    "caura_maripa": {"lat": 7.41, "lon": -65.18},
}


def build_nasa_url(lat: float, lon: float) -> str:
    """Construye la URL oficial del endpoint diario de NASA POWER."""
    query = {
        "parameters": "PRECTOTCORR",
        "community": "AG",
        "latitude": lat,
        "longitude": lon,
        "start": "19810101",
        "end": "20251231",
        "format": "CSV",
    }
    return f"{NASA_POWER_API}?{urllib.parse.urlencode(query)}"


def ensure_raw_data(force_download: bool = False) -> dict[str, Any]:
    """Garantiza que todos los archivos crudos estén en data/raw/ usando rutas relativas.

    Si los archivos existen en el repositorio base local (../data/raw y ../data/external),
    los sincroniza a data/raw/ para ejecución inmediata y reproducible sin depender de
    latencia de red; en caso contrario, los descarga desde las APIs públicas oficiales
    de NASA POWER y NOAA CPC.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    parent_raw = PARENT_REPO_ROOT / "data" / "raw"
    parent_ext = PARENT_REPO_ROOT / "data" / "external"

    # 1. Niveles del río Orinoco (Excel oficial y CSV histórico)
    river_raw_base = (
        "https://raw.githubusercontent.com/Cesar2926/Proyecto-IA-Orinoco/main/data/raw"
    )
    river_files = ["DATOS RIO ORINOCO.xlsx", "dataset_combinado.csv"]
    for fname in river_files:
        target = RAW_DIR / fname
        source = parent_raw / fname
        if not target.exists() or force_download:
            if source.exists() and not force_download:
                shutil.copy2(source, target)
            else:
                url = f"{river_raw_base}/{urllib.parse.quote(fname)}"
                with urllib.request.urlopen(url, timeout=60) as resp:
                    target.write_bytes(resp.read())

    # 2. Índice climático NOAA ONI
    oni_target = RAW_DIR / "noaa_oni.ascii.txt"
    oni_source = parent_ext / "noaa_oni.ascii.txt"
    if not oni_target.exists() or force_download:
        if oni_source.exists() and not force_download:
            shutil.copy2(oni_source, oni_target)
        else:
            with urllib.request.urlopen(NOAA_ONI_URL, timeout=60) as resp:
                oni_target.write_bytes(resp.read())

    # 3. 15 estaciones de precipitación diaria NASA POWER (246.540 registros diarios)
    for point_name, coords in NASA_POINTS.items():
        fname = f"nasa_power_{point_name}.csv"
        target = RAW_DIR / fname
        source = parent_ext / fname
        if not target.exists() or force_download:
            if source.exists() and not force_download:
                shutil.copy2(source, target)
            else:
                url = build_nasa_url(coords["lat"], coords["lon"])
                with urllib.request.urlopen(url, timeout=120) as resp:
                    target.write_bytes(resp.read())
                time.sleep(0.5)

    files = sorted(p for p in RAW_DIR.glob("*") if p.is_file() and p.name != ".gitkeep")
    total_bytes = sum(p.stat().st_size for p in files)

    return {
        "fecha_descarga": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "directorio_relativo": "data/raw/",
        "archivos_descargados": len(files),
        "peso_total_disco_mb": round(total_bytes / (1024 * 1024), 3),
        "urls_origen": {
            "niveles_orinoco": ORINOCO_REPO_URL,
            "nasa_power_api": NASA_POWER_API,
            "noaa_cpc_oni": NOAA_ONI_URL,
        },
        "detalle_fuentes": {
            "niveles_excel_2009_2025": "data/raw/DATOS RIO ORINOCO.xlsx (5.934 días x 4 estaciones = 23.736 celdas crudas; 23.376 válidas 2009-2024)",
            "niveles_historico_1974_2025": "data/raw/dataset_combinado.csv (18.683 días x 4 estaciones = 74.732 registros)",
            "nasa_power_15_estaciones": "15 archivos CSV (1981-2025, 16.436 días x 15 estaciones = 246.540 registros diarios)",
            "noaa_oni_trimestral": "data/raw/noaa_oni.ascii.txt (920 períodos trimestrales móviles desde 1950)",
        },
    }


if __name__ == "__main__":
    summary = ensure_raw_data()
    print("=== Ingesta completada en data/raw/ ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
