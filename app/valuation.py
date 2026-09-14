"""Modelo de valor de mercado (detector de gangas) y fichas de jugador/equipo.

Portado de MODELIZACION.ipynb (celdas 68-85) sin cambiar la logica.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

# hiperparametros hallados por busqueda aleatoria (RandomizedSearchCV, 40 iters,
# 5-fold CV) sobre las features de rendimiento: sube el R2 de 0.732 (GB) a 0.742
HIPERPARAMS_XGB = dict(
    n_estimators=900, max_depth=3, learning_rate=0.02,
    subsample=0.6, colsample_bytree=1.0, reg_lambda=1, min_child_weight=8,
)


def construir_features_gangas(df):
    excluir = {
        "MARKET VALUE", "HIGHEST MARKET VALUE", "LOG_MARKET_VALUE",
        "MARKET_VALUE_DROP", "MARKET_VALUE_TREND_PCT",
        "PLAYER_ID", "TEAM_ID", "LEAGUE_ID", "NAME", "TEAM", "LEAGUE",
        "MAIN POSITION", "POSITION 2", "POSITION 3", "NATIONALITY",
        "BIRTH DATE", "CONTRACT UNTIL", "TEAM JOINED", "PREVIOUS TEAM",
        "FOOT", "ROL", "SCORE_ROL", "SCORE_ROL_AJUSTADO",
        "CONTRACT_OPPORTUNITY", "MONTHS_TO_CONTRACT_END",
        # salidas de una ejecucion anterior del propio modelo de valor: si el
        # dataframe ya las tiene (p.ej. datos.pkl ya enriquecido), usarlas como
        # feature es fuga de datos (el modelo se predice a si mismo)
        "VALOR_PREDICHO", "LOG_RESIDUO", "DIFERENCIA_VALOR", "RATIO_VALOR", "TENDENCIA",
        "VALOR_REALISTA",
    }

    crudas_con_z = [c for c in df.columns if f"{c}_Z_LIGA" in df.columns]
    excluir.update(crudas_con_z)

    features = [c for c in df.columns
                if c not in excluir and pd.api.types.is_numeric_dtype(df[c])]

    print(f"  Features seleccionadas: {len(features)}")
    return features


def evaluar_modelos_cv(df, k=5, semilla=42):
    features = construir_features_gangas(df)

    X = df[features].fillna(0)
    y = df["LOG_MARKET_VALUE"]

    cv = KFold(n_splits=k, shuffle=True, random_state=semilla)

    modelos = {
        "RandomForest": RandomForestRegressor(
            n_estimators=400, min_samples_leaf=3, random_state=semilla, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=400, max_depth=3, learning_rate=0.05, random_state=semilla),
        "XGBoost": XGBRegressor(
            **HIPERPARAMS_XGB, random_state=semilla, n_jobs=-1),
    }

    predicciones = {}
    for nombre, modelo in modelos.items():
        pred_oof_log = cross_val_predict(modelo, X, y, cv=cv, n_jobs=-1)

        r2 = r2_score(y, pred_oof_log)
        mae_log = mean_absolute_error(y, pred_oof_log)
        mae_millones = mean_absolute_error(np.expm1(y), np.expm1(pred_oof_log))

        print(f"\n  {nombre}  (validacion cruzada {k}-fold)")
        print(f"    R2:              {r2:.3f}")
        print(f"    MAE (log):       {mae_log:.3f}")
        print(f"    MAE (millones):  {mae_millones:.2f} M")

        predicciones[nombre] = {
            "pred_oof_log": pred_oof_log,
            "features": features,
            "r2": r2, "mae_log": mae_log, "mae_millones": mae_millones}

    return predicciones


def construir_features_valor_realista(df):
    """Igual que construir_features_gangas, pero permitiendo el valor de mercado
    historico. Uso exclusivo del modelo 'realista' (ver evaluar_modelo_realista):
    no debe usarse para el detector de gangas, porque ancla la prediccion al
    historico y diluye la señal de infra/sobrevaloracion por rendimiento."""
    features = construir_features_gangas(df)
    if "HIGHEST MARKET VALUE" in df.columns and "HIGHEST MARKET VALUE" not in features:
        features = features + ["HIGHEST MARKET VALUE"]
    return features


def evaluar_modelo_realista(df, k=5, semilla=42):
    """Modelo aparte para dar una estimacion de valor mas 'realista' (ficha del
    jugador), no para el detector de gangas. Incluye el valor de mercado
    historico maximo, lo que sube mucho la precision (R2 ~0.92 vs ~0.74) porque
    el valor actual esta muy anclado al pico anterior; por eso se mantiene fuera
    del modelo de rendimiento que alimenta analizar_valor/gangas_similares."""
    features = construir_features_valor_realista(df)
    X = df[features].fillna(0)
    y = df["LOG_MARKET_VALUE"]

    cv = KFold(n_splits=k, shuffle=True, random_state=semilla)
    modelo = XGBRegressor(**HIPERPARAMS_XGB, random_state=semilla, n_jobs=-1)
    pred_oof_log = cross_val_predict(modelo, X, y, cv=cv, n_jobs=-1)

    r2 = r2_score(y, pred_oof_log)
    mae_log = mean_absolute_error(y, pred_oof_log)
    mae_millones = mean_absolute_error(np.expm1(y), np.expm1(pred_oof_log))

    print(f"\n  Valor realista (con historico)  (validacion cruzada {k}-fold)")
    print(f"    R2:              {r2:.3f}")
    print(f"    MAE (millones):  {mae_millones:.2f} M")

    return {"pred_oof_log": pred_oof_log, "features": features,
            "r2": r2, "mae_log": mae_log, "mae_millones": mae_millones}


def incorporar_valor_realista(df, prediccion_realista):
    df = df.copy()
    df["VALOR_REALISTA"] = np.expm1(prediccion_realista["pred_oof_log"]).round(2)
    print("  VALOR_REALISTA incorporado al dataframe")
    return df


def incorporar_predicciones(df, prediccion_cv):
    df = df.copy()
    pred_log = prediccion_cv["pred_oof_log"]
    df["VALOR_PREDICHO"] = np.expm1(pred_log).round(2)
    df["LOG_RESIDUO"] = (pred_log - df["LOG_MARKET_VALUE"].to_numpy()).round(3)
    df["DIFERENCIA_VALOR"] = (df["VALOR_PREDICHO"] - df["MARKET VALUE"]).round(2)
    df["RATIO_VALOR"] = (df["VALOR_PREDICHO"] / df["MARKET VALUE"]).round(2)
    df["TENDENCIA"] = np.select(
        [df["LOG_RESIDUO"] > 0.20, df["LOG_RESIDUO"] < -0.20],
        ["Infravalorado (posible subida)", "Sobrevalorado (posible bajada)"],
        default="En linea con su valor")
    print("  Predicciones incorporadas al dataframe")
    return df


def analizar_valor(df, prediccion_cv, n=20,
                   direccion="infravalorados",
                   metrica="log_residuo",        # "absoluto" | "ratio" | "log_residuo"
                   valor_min=None, valor_max=None,
                   edad_max=None, solo_contrato_acabando=False):

    out = df.copy()
    pred_log = prediccion_cv["pred_oof_log"]
    real_log = df["LOG_MARKET_VALUE"].to_numpy()

    out["VALOR_PREDICHO"] = np.expm1(pred_log).round(2)
    out["VALOR_REAL"] = out["MARKET VALUE"]

    out["DIFERENCIA"] = (out["VALOR_PREDICHO"] - out["VALOR_REAL"]).round(2)
    out["RATIO"] = (out["VALOR_PREDICHO"] / out["VALOR_REAL"]).round(2)
    out["LOG_RESIDUO"] = (pred_log - real_log).round(3)

    if valor_min is not None:
        out = out[out["VALOR_REAL"] >= valor_min]
    if valor_max is not None:
        out = out[out["VALOR_REAL"] <= valor_max]
    if edad_max is not None:
        out = out[out["AGE"] <= edad_max]
    if solo_contrato_acabando:
        out = out[out["CONTRACT_OPPORTUNITY"] == 1]

    col = {"absoluto": "DIFERENCIA", "ratio": "RATIO",
           "log_residuo": "LOG_RESIDUO"}[metrica]

    if direccion == "infravalorados":
        resultado = out.nlargest(n, col)
    else:
        resultado = out.nsmallest(n, col)

    cols = ["NAME", "TEAM", "LEAGUE", "MAIN POSITION", "AGE",
            "VALOR_REAL", "VALOR_PREDICHO", "DIFERENCIA", "RATIO", "LOG_RESIDUO"]
    return resultado[cols].reset_index(drop=True)


def ficha_jugador(df, nombre_jugador):
    idx = df.index[df["NAME"] == nombre_jugador]
    if len(idx) == 0:
        print(f"  No encontrado: {nombre_jugador}")
        return None
    j = df.loc[idx[0]]

    mismos_rol = df[df["ROL"] == j["ROL"]]
    pct_score = (mismos_rol["SCORE_ROL_AJUSTADO"] < j["SCORE_ROL_AJUSTADO"]).mean() * 100

    ficha = {
        "Nombre": j["NAME"],
        "Equipo": j["TEAM"],
        "Liga": j["LEAGUE"],
        "Posicion": j["MAIN POSITION"],
        "Rol": j["ROL"],
        "Edad": int(j["AGE"]),
        "Minutos": int(j["MINS"]),
        "Valor real (M)": j["MARKET VALUE"],
        "Valor predicho por rendimiento (M)": j["VALOR_PREDICHO"],
        "Valor estimado realista (M)": j["VALOR_REALISTA"] if "VALOR_REALISTA" in df.columns else None,
        "Diferencia (M)": j["DIFERENCIA_VALOR"],
        "Ratio": j["RATIO_VALOR"],
        "Situacion": j["TENDENCIA"],
        "Score de rol": round(j["SCORE_ROL_AJUSTADO"], 3),
        "Percentil en su rol": f"{pct_score:.0f}%",
        "Contrato hasta": j["CONTRACT UNTIL"],
        "Meses contrato": int(j["MONTHS_TO_CONTRACT_END"]) if pd.notna(j["MONTHS_TO_CONTRACT_END"]) else None,
    }
    return pd.Series(ficha)


def analizar_equipo(df, nombre_equipo, ordenar_por="DIFERENCIA_VALOR"):
    plantilla = df[df["TEAM"] == nombre_equipo].copy()
    if plantilla.empty:
        print(f"  Equipo no encontrado: {nombre_equipo}")
        return None

    cols = ["NAME", "MAIN POSITION", "ROL", "AGE",
            "MARKET VALUE", "VALOR_PREDICHO", "DIFERENCIA_VALOR",
            "RATIO_VALOR", "TENDENCIA", "SCORE_ROL_AJUSTADO",
            "MONTHS_TO_CONTRACT_END"]

    resultado = plantilla[cols].sort_values(ordenar_por, ascending=False)

    print(f"  {nombre_equipo}: {len(plantilla)} jugadores")
    print(f"  Valor total real:     {plantilla['MARKET VALUE'].sum():.1f} M")
    print(f"  Valor total predicho: {plantilla['VALOR_PREDICHO'].sum():.1f} M")
    infra = (plantilla['TENDENCIA'].str.contains('Infra')).sum()
    sobre = (plantilla['TENDENCIA'].str.contains('Sobre')).sum()
    print(f"  Infravalorados: {infra}  |  Sobrevalorados: {sobre}")

    return resultado.reset_index(drop=True)
