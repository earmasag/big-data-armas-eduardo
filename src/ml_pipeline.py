"""Fase 5 CRISP-DM: Modeling & Evaluation para pronóstico hidrológico a +14 días.

Entrena y compara:
1. Línea Base (Naive Persistence: y_{t+14} = y_t)
2. Ridge Regression (Modelo lineal regularizado L2 con StandardScaler + OneHotEncoder)
3. Random Forest Regressor (Ensamble Bagging de árboles sobre incremento hidrodinámico)
4. XGBoost Regressor (Gradient Boosting de árboles sobre incremento hidrodinámico)

Guarda el artefacto entrenado en models/modelo_orinoco.joblib para inferencia directa
en el dashboard de Streamlit sin reentrenar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from src.download_data import PROJECT_ROOT
from src.etl_pipeline import PROCESSED_DIR, STATION_LABELS

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_ARTIFACT_PATH = MODELS_DIR / "modelo_orinoco.joblib"

NUMERIC_FEATURES = [
    "y",
    "y_lag_7",
    "y_lag_14",
    "delta_y_1d",
    "delta_y_3d",
    "delta_y_7d",
    "ayacucho_lag_7",
    "ayacucho_lag_14",
    "caicara_lag_7",
    "ciudad_bolivar_lag_1",
    "lluvia_local_14d",
    "lluvia_cuenca_14d",
    "oni",
    "sin_doy",
    "cos_doy",
]

CATEGORICAL_FEATURES = [
    "unique_id",
    "epoca_hidrologica",
    "fase_enso",
]

FEATURE_BUSINESS_LABELS = {
    "y": "Nivel actual del río en la estación y(t) [Base hidrométrica local]",
    "y_lag_7": "Nivel en la estación hace 7 días y(t-7) [Memoria semanal]",
    "y_lag_14": "Nivel en la estación hace 14 días y(t-14) [Memoria quincenal]",
    "delta_y_1d": "Velocidad de ascenso en las últimas 24 horas [m/día]",
    "delta_y_3d": "Cambio de nivel en los últimos 3 días [Tendencia corta]",
    "delta_y_7d": "Cambio de nivel en los últimos 7 días [Aceleración de la onda]",
    "ayacucho_lag_7": "Nivel aguas arriba en Puerto Ayacucho (t-7 días)",
    "ayacucho_lag_14": "Nivel aguas arriba en Puerto Ayacucho (t-14 días)",
    "caicara_lag_7": "Nivel aguas arriba en Caicara del Orinoco (t-7 días)",
    "ciudad_bolivar_lag_1": "Nivel en Ciudad Bolívar (t-1 día) [Tránsito hacia Palúa]",
    "lluvia_local_14d": "Precipitación local acumulada en 14 días (NASA POWER, mm)",
    "lluvia_cuenca_14d": "Precipitación media en la cuenca del Orinoco en 14 días (mm)",
    "oni": "Índice climático El Niño / La Niña (NOAA ONI con lag de publicación)",
    "sin_doy": "Componente estacional anual Seno(Día del Año)",
    "cos_doy": "Componente estacional anual Coseno(Día del Año)",
    "unique_id": "Estación limnimétrica del tramo fluvial",
    "epoca_hidrologica": "Época hidrológica del año",
    "fase_enso": "Fase climática ENSO",
}


class DeltaHorizonPipeline(BaseEstimator, RegressorMixin):
    """Pipeline que modela la variación a 14 días Δy = y(t+14) - y(t) y reconstruye y(t+14).

    Permite que los modelos basados en árboles (Random Forest y XGBoost) concentren
    sus particiones en la dinámica física de ascenso/descenso sin verse distorsionados
    por la diferencia de altitud entre estaciones (Ayacucho ~46 m vs Palúa ~6 m).
    """

    def __init__(self, preprocessor: ColumnTransformer, regressor: Any) -> None:
        self.preprocessor = preprocessor
        self.regressor = regressor

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> DeltaHorizonPipeline:
        self.prep_ = clone(self.preprocessor)
        self.reg_ = clone(self.regressor)
        X_trans = self.prep_.fit_transform(X)
        y_base = X["y"].to_numpy(dtype=np.float64)
        delta_target = np.asarray(y, dtype=np.float64) - y_base
        self.reg_.fit(X_trans, delta_target)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_trans = self.prep_.transform(X)
        y_base = X["y"].to_numpy(dtype=np.float64)
        delta_pred = self.reg_.predict(X_trans)
        return y_base + delta_pred


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, baseline_mae: float) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mejora_pct = float(100.0 * (1.0 - mae / baseline_mae)) if baseline_mae > 0 else 0.0
    return {
        "mae_m": round(mae, 4),
        "rmse_m": round(rmse, 4),
        "r2": round(r2, 4),
        "mejora_vs_baseline_pct": round(mejora_pct, 2),
    }


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def run_ml_pipeline() -> dict[str, Any]:
    """Entrena Línea Base + 3 modelos de regresión sin data leakage y guarda el artefacto."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = PROCESSED_DIR / "orinoco_limpio.parquet"
    df = pd.read_parquet(parquet_path)

    required_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["target_y_t14", "ds"]
    model_df = df.dropna(subset=required_cols).sort_values(["ds", "unique_id"]).reset_index(drop=True)

    # Partición temporal estricta con embargo de 14 días para evitar fuga del target t+14:
    # Train: observaciones hasta 2022-12-17 (su target t+14 llega hasta 2022-12-31)
    # Test:  observaciones desde 2023-01-01 hasta 2024-12-17
    train_df = model_df[model_df["ds"] <= "2022-12-17"].copy()
    test_df = model_df[model_df["ds"] >= "2023-01-01"].copy()

    X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_df["target_y_t14"].to_numpy(dtype=np.float64)

    X_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test_df["target_y_t14"].to_numpy(dtype=np.float64)

    # 1. Línea Base: Persistencia Naive (predice que en t+14 el río tendrá el nivel actual y_t)
    y_pred_naive = test_df["y"].to_numpy(dtype=np.float64)
    baseline_mae = float(mean_absolute_error(y_test, y_pred_naive))
    naive_metrics = _compute_metrics(y_test, y_pred_naive, baseline_mae)

    # 2. Modelo 1: Ridge Regression (Lineal regularizado L2)
    ridge_pipe = DeltaHorizonPipeline(
        preprocessor=_build_preprocessor(),
        regressor=Ridge(alpha=25.0, random_state=42),
    )
    ridge_pipe.fit(X_train, y_train)
    y_pred_ridge = ridge_pipe.predict(X_test)
    ridge_metrics = _compute_metrics(y_test, y_pred_ridge, baseline_mae)

    # 3. Modelo 2: Random Forest Regressor
    rf_pipe = DeltaHorizonPipeline(
        preprocessor=_build_preprocessor(),
        regressor=RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=6,
            random_state=42,
            n_jobs=-1,
        ),
    )
    rf_pipe.fit(X_train, y_train)
    y_pred_rf = rf_pipe.predict(X_test)
    rf_metrics = _compute_metrics(y_test, y_pred_rf, baseline_mae)

    # 4. Modelo 3: XGBoost Regressor
    xgb_pipe = DeltaHorizonPipeline(
        preprocessor=_build_preprocessor(),
        regressor=XGBRegressor(
            n_estimators=260,
            max_depth=4,
            learning_rate=0.035,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=3.0,
            random_state=42,
            n_jobs=-1,
        ),
    )
    xgb_pipe.fit(X_train, y_train)
    y_pred_xgb = xgb_pipe.predict(X_test)
    xgb_metrics = _compute_metrics(y_test, y_pred_xgb, baseline_mae)

    candidates = [
        {"modelo": "Línea Base (Naive)", "tipo": "Persistencia y(t+14) = y(t)", **naive_metrics, "pipe": None, "preds": y_pred_naive},
        {"modelo": "Ridge Regression", "tipo": "Lineal Regularizado L2", **ridge_metrics, "pipe": ridge_pipe, "preds": y_pred_ridge},
        {"modelo": "Random Forest", "tipo": "Ensamble Bagging (200 árboles)", **rf_metrics, "pipe": rf_pipe, "preds": y_pred_rf},
        {"modelo": "XGBoost Regressor", "tipo": "Gradient Boosting (260 árboles)", **xgb_metrics, "pipe": xgb_pipe, "preds": y_pred_xgb},
    ]

    ml_only = [c for c in candidates if c["pipe"] is not None]
    winner = min(ml_only, key=lambda c: c["mae_m"])
    winner_pipe: DeltaHorizonPipeline = winner["pipe"]
    winner_preds: np.ndarray = winner["preds"]

    # Métricas desglosadas por estación en el conjunto de prueba (2023-2024)
    test_eval_df = test_df[["ds", "unique_id", "estacion_nombre", "y", "target_y_t14"]].copy()
    test_eval_df["pred_naive"] = y_pred_naive
    test_eval_df["pred_ridge"] = y_pred_ridge
    test_eval_df["pred_rf"] = y_pred_rf
    test_eval_df["pred_xgb"] = y_pred_xgb
    test_eval_df["pred_ganador"] = winner_preds

    metrics_by_station: list[dict[str, Any]] = []
    for st, group in test_eval_df.groupby("unique_id", observed=True):
        st_true = group["target_y_t14"].to_numpy()
        st_naive = group["pred_naive"].to_numpy()
        st_win = group["pred_ganador"].to_numpy()
        st_base_mae = float(mean_absolute_error(st_true, st_naive))
        st_win_metrics = _compute_metrics(st_true, st_win, st_base_mae)
        metrics_by_station.append({
            "unique_id": str(st),
            "estacion": STATION_LABELS.get(str(st), str(st)),
            "registros_test": int(len(group)),
            "mae_baseline_m": round(st_base_mae, 4),
            "mae_modelo_m": st_win_metrics["mae_m"],
            "rmse_modelo_m": st_win_metrics["rmse_m"],
            "r2_modelo": st_win_metrics["r2"],
            "mejora_mae_pct": st_win_metrics["mejora_vs_baseline_pct"],
        })

    # Importancia de variables sobre la dinámica de cambio a 14 días (usando XGBoost para desglose no lineal rico)
    prep_step: ColumnTransformer = xgb_pipe.prep_
    reg_step = xgb_pipe.reg_
    raw_feat_names = list(prep_step.get_feature_names_out())
    raw_importances = np.asarray(reg_step.feature_importances_, dtype=float)
    raw_importances = raw_importances / max(raw_importances.sum(), 1e-12)

    grouped_imp: dict[str, float] = {}
    for fname, val in zip(raw_feat_names, raw_importances):
        clean_name = fname.replace("num__", "").replace("cat__", "")
        matched_col = clean_name
        for cat_col in CATEGORICAL_FEATURES:
            if clean_name.startswith(cat_col + "_"):
                matched_col = cat_col
                break
        grouped_imp[matched_col] = grouped_imp.get(matched_col, 0.0) + float(val)

    top_features = sorted(grouped_imp.items(), key=lambda kv: kv[1], reverse=True)
    variables_importantes = [
        {
            "variable": col,
            "importancia_pct": round(imp * 100.0, 2),
            "interpretacion_negocio": FEATURE_BUSINESS_LABELS.get(col, col),
        }
        for col, imp in top_features[:10]
    ]

    # Valores por defecto por estación (última observación del dataset) para el formulario de Streamlit
    defaults_by_station: dict[str, dict[str, Any]] = {}
    for st, group in model_df.groupby("unique_id", observed=True):
        last_row = group.sort_values("ds").iloc[-1]
        defaults_by_station[str(st)] = {
            col: (
                float(last_row[col])
                if col in NUMERIC_FEATURES
                else str(last_row[col])
            )
            for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES
        }
        defaults_by_station[str(st)]["ultima_fecha"] = str(last_row["ds"].date())
        defaults_by_station[str(st)]["real_t14"] = float(last_row["target_y_t14"])

    cb_test = test_eval_df[test_eval_df["unique_id"] == "ciudad_bolivar"].sort_values("ds")
    serie_cb = [
        {
            "ds": str(r["ds"].date()),
            "nivel_actual_t": round(float(r["y"]), 3),
            "real": round(float(r["target_y_t14"]), 3),
            "naive": round(float(r["pred_naive"]), 3),
            "pred_ganador": round(float(r["pred_ganador"]), 3),
        }
        for _, r in cb_test.iterrows()
    ]

    comparison_table = [
        {
            "modelo": c["modelo"],
            "tipo": c["tipo"],
            "mae_m": c["mae_m"],
            "rmse_m": c["rmse_m"],
            "r2": c["r2"],
            "mejora_vs_baseline_pct": c["mejora_vs_baseline_pct"],
        }
        for c in candidates
    ]

    ml_summary = {
        "tarea": "Predicción (Regresión supervisada multi-estación a horizonte t + 14 días)",
        "variable_objetivo": "target_y_t14 (Nivel hidrométrico del río Orinoco en m s.n.m. a 14 días)",
        "prevencion_data_leakage": (
            "Partición cronológica estricta con embargo de 14 días: Entrenamiento 2009-01-29 a 2022-12-17 "
            f"({len(train_df)} registros) y Prueba 2023-01-01 a 2024-12-17 ({len(test_df)} registros). "
            "StandardScaler y OneHotEncoder encapsulados en Pipeline ajustados exclusivamente sobre X_train; "
            "ONI desplazado 3 meses por retraso oficial de publicación."
        ),
        "registros_train": int(len(train_df)),
        "registros_test": int(len(test_df)),
        "modelo_ganador": winner["modelo"],
        "algoritmo_ganador": winner["tipo"],
        "metricas_ganador": {
            "mae_m": winner["mae_m"],
            "rmse_m": winner["rmse_m"],
            "r2": winner["r2"],
            "mejora_vs_baseline_pct": winner["mejora_vs_baseline_pct"],
        },
        "linea_base": {
            "modelo": "Línea Base (Naive Persistence)",
            "mae_m": naive_metrics["mae_m"],
            "rmse_m": naive_metrics["rmse_m"],
            "r2": naive_metrics["r2"],
        },
        "comparacion_modelos": comparison_table,
        "metricas_por_estacion": metrics_by_station,
        "variables_importantes": variables_importantes,
        "defaults_por_estacion": defaults_by_station,
        "serie_test_ciudad_bolivar": serie_cb,
    }

    artifact_payload = {
        "pipeline": winner_pipe,
        "all_pipelines": {
            "Ridge Regression": ridge_pipe,
            "Random Forest": rf_pipe,
            "XGBoost Regressor": xgb_pipe,
        },
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_business_labels": FEATURE_BUSINESS_LABELS,
        "defaults_by_station": defaults_by_station,
        "summary": ml_summary,
    }
    joblib.dump(artifact_payload, MODEL_ARTIFACT_PATH)
    test_eval_df.to_parquet(PROCESSED_DIR / "predicciones_test_2023_2024.parquet", index=False)

    return ml_summary


if __name__ == "__main__":
    summary = run_ml_pipeline()
    print("=== Entrenamiento y Evaluación ML completados ===")
    print("Modelo ganador:", summary["modelo_ganador"], "->", summary["metricas_ganador"])
    for row in summary["comparacion_modelos"]:
        print(" ", row)
