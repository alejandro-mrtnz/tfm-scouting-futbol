"""Limpieza y feature engineering sobre el Excel de estadisticas de jugadores.

Portado de MODELIZACION.ipynb (celdas 1-58) sin cambiar la logica.
"""
import numpy as np
import pandas as pd

COLS_RENDIMIENTO = [
    # --- Ofensivo / finalizacion ---
    "GOALS_90",
    "ASSISTS_90",
    "SHOTS PER MATCH",
    "GOAL_CONVERSION",
    "xG/90",
    "xGDif",

    # --- Creacion ---
    "KEY PASSES",
    "CROSS",
    "DRIBBLE_OF",
    "TACKLE RECEIVED",
    "SPACE BALL",          # pases al hueco

    # --- Pase / posesion ---
    "SUCCESSFUL PASSES (%)",
    "LONG PASS",
    "PromeP",              # pases promedio por partido

    # --- Defensivo ---
    "TACKLE",
    "INTERCEPTION",
    "CLEARANCE",
    "BLOCKS",
    "DRIBBLE_DEF",
    "INTENTIONAL OFFSIDES",  # fueras de juego provocados
]

ROLES = {
    "Defensa central":      ["Defensa central"],
    "Lateral":              ["Lateral derecho", "Lateral izquierdo"],
    "Mediocentro defensivo":["Pivote", "Mediocentro"],
    "Media punta":          ["Mediocentro ofensivo", "Mediapunta",
                             "Interior derecho", "Interior izquierdo"],
    "Extremo":              ["Extremo derecho", "Extremo izquierdo"],
    "Delantero":            ["Delantero centro"],
}

METRICAS_ROL = {
    "Defensa central": {
        "INTERCEPTION_Z_LIGA": 2, "CLEARANCE_Z_LIGA": 1.5, "TACKLE_Z_LIGA": 1.5,
        "BLOCKS_Z_LIGA": 1.5, "INTENTIONAL OFFSIDES_Z_LIGA": 1,
        "SUCCESSFUL PASSES (%)_Z_LIGA": 1, "LONG PASS_Z_LIGA": 0.5,
    },
    "Lateral": {
        "CROSS_Z_LIGA": 1.5, "TACKLE_Z_LIGA": 1.5, "INTERCEPTION_Z_LIGA": 1.5,
        "KEY PASSES_Z_LIGA": 1, "DRIBBLE_OF_Z_LIGA": 1, "ASSISTS_90_Z_LIGA": 1,
        "DRIBBLE_DEF_Z_LIGA": 1,
    },
    "Mediocentro defensivo": {
        "TACKLE_Z_LIGA": 2, "INTERCEPTION_Z_LIGA": 2, "SUCCESSFUL PASSES (%)_Z_LIGA": 1.5,
        "PromeP_Z_LIGA": 1, "LONG PASS_Z_LIGA": 1, "KEY PASSES_Z_LIGA": 0.5,
    },
    "Media punta": {
        "KEY PASSES_Z_LIGA": 2, "ASSISTS_90_Z_LIGA": 1.5, "SPACE BALL_Z_LIGA": 1.5,
        "DRIBBLE_OF_Z_LIGA": 1, "GOALS_90_Z_LIGA": 1, "xG/90_Z_LIGA": 1, "PromeP_Z_LIGA": 0.5,
    },
    "Extremo": {
        "DRIBBLE_OF_Z_LIGA": 2, "ASSISTS_90_Z_LIGA": 1.5, "KEY PASSES_Z_LIGA": 1.5,
        "GOALS_90_Z_LIGA": 1, "xG/90_Z_LIGA": 1, "CROSS_Z_LIGA": 1, "TACKLE RECEIVED_Z_LIGA": 0.5,
    },
    "Delantero": {
        "GOALS_90_Z_LIGA": 2, "xG/90_Z_LIGA": 1.5, "GOAL_CONVERSION_Z_LIGA": 1.5,
        "SHOTS PER MATCH_Z_LIGA": 1, "xGDif_Z_LIGA": 1, "ASSISTS_90_Z_LIGA": 0.5,
    },
}


def cargar_datos(ruta, hoja=None):
    if ruta.endswith(".csv"):
        df = pd.read_csv(ruta)
    else:
        df = pd.read_excel(ruta, sheet_name=hoja)
    print(f"  Datos cargados: {df.shape[0]} filas x {df.shape[1]} columnas")
    return df


def rellenar_position_2(df):
    df = df.copy()
    n = df["POSITION 2"].isna().sum()
    df["POSITION 2"] = df["POSITION 2"].fillna(df["MAIN POSITION"])
    print(f"  POSITION 2 rellenados con MAIN POSITION: {n}")
    return df


def rellenar_position_3(df):
    df = df.copy()
    n = df["POSITION 3"].isna().sum()
    df["POSITION 3"] = df["POSITION 3"].fillna(df["POSITION 2"])
    print(f"  POSITION 3 rellenados con POSITION 2: {n}")
    return df


def rellenar_height_mediana(df):
    df = df.copy()
    mediana = df["HEIGHT"].median()
    n = df["HEIGHT"].isna().sum()
    df["HEIGHT"] = df["HEIGHT"].fillna(mediana)
    print(f"  HEIGHT rellenados con mediana ({mediana}): {n}")
    return df


def rellenar_foot(df):
    df = df.copy()
    n = df["FOOT"].isna().sum()
    df["FOOT"] = df["FOOT"].fillna("Derecho")
    print(f"  FOOT rellenados con 'Derecho': {n}")
    return df


def rellenar_contract_until(df):
    df = df.copy()
    n = df["CONTRACT UNTIL"].isna().sum()
    df["CONTRACT UNTIL"] = df["CONTRACT UNTIL"].fillna("30/06/2026")
    print(f"  CONTRACT UNTIL rellenados con '30/06/2026': {n}")
    return df


def rellenar_team_joined(df):
    df = df.copy()
    n = df["TEAM JOINED"].isna().sum()
    df["TEAM JOINED"] = df["TEAM JOINED"].fillna("30/06/2025")
    print(f"  TEAM JOINED rellenados con '30/06/2025': {n}")
    return df


def rellenar_previous_team(df):
    df = df.copy()
    n = df["PREVIOUS TEAM"].isna().sum()
    df["PREVIOUS TEAM"] = df["PREVIOUS TEAM"].fillna("unknown")
    print(f"  PREVIOUS TEAM rellenados con 'unknown': {n}")
    return df


def filtrar_mins(df, minimo=500):
    df = df.copy()
    antes = df.shape[0]
    df = df[df["MINS"] >= minimo]
    print(f"  Filas eliminadas (MINS < {minimo}): {antes - df.shape[0]}")
    print(f"  Filas restantes: {df.shape[0]}")
    return df


def calcular_diferencia_market_value(df):
    df = df.copy()

    df["MARKET VALUE"] = pd.to_numeric(df["MARKET VALUE"], errors="coerce")
    df["HIGHEST MARKET VALUE"] = pd.to_numeric(df["HIGHEST MARKET VALUE"], errors="coerce")

    df["MARKET_VALUE_DROP"] = (df["HIGHEST MARKET VALUE"] - df["MARKET VALUE"]).round(2)

    df["MARKET_VALUE_TREND_PCT"] = np.where(
        df["HIGHEST MARKET VALUE"] > 0,
        ((df["MARKET VALUE"] - df["HIGHEST MARKET VALUE"]) / df["HIGHEST MARKET VALUE"]) * 100,
        np.nan,
    ).round(2)

    print("  MARKET_VALUE_DROP y MARKET_VALUE_TREND_PCT calculados")
    return df


def calcular_start_pct(df):
    df = df.copy()

    df["STARTS"] = pd.to_numeric(df["STARTS"], errors="coerce")
    df["BENCH"] = pd.to_numeric(df["BENCH"], errors="coerce")

    total = df["STARTS"] + df["BENCH"]
    df["START_PCT"] = np.where(
        total > 0,
        (df["STARTS"] / total) * 100,
        np.nan,
    ).round(2)

    print("  START_PCT calculado")
    return df


def calcular_available_min_pct(df):
    df = df.copy()

    df["STARTS"] = pd.to_numeric(df["STARTS"], errors="coerce")
    df["BENCH"] = pd.to_numeric(df["BENCH"], errors="coerce")
    df["MINS"] = pd.to_numeric(df["MINS"], errors="coerce")

    min_disponibles = (df["STARTS"] + df["BENCH"]) * 90
    df["AVAILABLE_MIN_PCT"] = (np.where(
        min_disponibles > 0,
        df["MINS"] / min_disponibles,
        np.nan,
    ).round(2)) * 100

    print("  AVAILABLE_MIN_PCT calculado")
    return df


def calcular_situacion_contractual(df, fecha_referencia="30/06/2026"):
    df = df.copy()

    ref = pd.to_datetime(fecha_referencia, format="%d/%m/%Y")
    contract = pd.to_datetime(df["CONTRACT UNTIL"], format="%d/%m/%Y", errors="coerce")

    df["MONTHS_TO_CONTRACT_END"] = (
        (contract.dt.year - ref.year) * 12 + (contract.dt.month - ref.month)
    )

    df["CONTRACT_OPPORTUNITY"] = np.where(
        df["MONTHS_TO_CONTRACT_END"].notna(),
        (df["MONTHS_TO_CONTRACT_END"] <= 12).astype(int),
        np.nan,
    )

    print("  MONTHS_TO_CONTRACT_END y CONTRACT_OPPORTUNITY calculados")
    return df


def calcular_contribuciones_90(df):
    df = df.copy()

    df["GOALS"] = pd.to_numeric(df["GOALS"], errors="coerce")
    df["ASSIST"] = pd.to_numeric(df["ASSIST"], errors="coerce")
    df["MINS"] = pd.to_numeric(df["MINS"], errors="coerce")

    df["GOALS_90"] = np.where(df["MINS"] > 0, 90 * df["GOALS"] / df["MINS"], np.nan).round(2)
    df["ASSISTS_90"] = np.where(df["MINS"] > 0, 90 * df["ASSIST"] / df["MINS"], np.nan).round(2)
    df["GOAL_CONTRIBUTIONS_90"] = np.where(
        df["MINS"] > 0,
        90 * (df["GOALS"] + df["ASSIST"]) / df["MINS"],
        np.nan,
    ).round(2)

    print("  GOALS_90, ASSISTS_90 y GOAL_CONTRIBUTIONS_90 calculados")
    return df


def calcular_goal_conversion(df):
    df = df.copy()

    df["GOALS"] = pd.to_numeric(df["GOALS"], errors="coerce")
    df["SHOTS"] = pd.to_numeric(df["SHOTS"], errors="coerce")

    df["GOAL_CONVERSION"] = np.where(
        df["SHOTS"] > 0,
        df["GOALS"] / df["SHOTS"],
        0,
    ).round(2)

    print("  GOAL_CONVERSION calculado")
    return df


def calcular_defensive_actions(df):
    df = df.copy()

    cols_def = ["TACKLE", "INTERCEPTION", "CLEARANCE", "BLOCKS", "DRIBBLE_DEF"]
    for c in cols_def:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["DEFENSIVE_ACTIONS"] = df[cols_def].sum(axis=1, skipna=True).round(2)

    print(f"  DEFENSIVE_ACTIONS calculado (suma de {', '.join(cols_def)})")
    return df


def calcular_chance_creation(df):
    df = df.copy()

    cols_creacion = ["KEY PASSES", "CROSS"]
    for c in cols_creacion:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["CHANCE_CREATION"] = df[cols_creacion].sum(axis=1, skipna=True).round(2)

    print(f"  CHANCE_CREATION calculado (suma de {', '.join(cols_creacion)})")
    return df


def calcular_dribbling_threat(df):
    df = df.copy()

    cols = ["DRIBBLE_OF", "TACKLE RECEIVED"]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["DRIBBLING_THREAT"] = df[cols].sum(axis=1, skipna=True).round(2)

    print(f"  DRIBBLING_THREAT calculado (suma de {', '.join(cols)})")
    return df


def excluir_porteros(df):
    df = df.copy()
    antes = df.shape[0]
    df = df[df["MAIN POSITION"] != "Portero"]
    print(f"  Porteros eliminados: {antes - df.shape[0]}")
    print(f"  Jugadores de campo restantes: {df.shape[0]}")
    return df


def excluir_market_value_cero(df):
    df = df.copy()
    df["MARKET VALUE"] = pd.to_numeric(df["MARKET VALUE"], errors="coerce")
    antes = df.shape[0]
    df = df[df["MARKET VALUE"] > 0]
    print(f"  Jugadores con valor 0 o nulo eliminados: {antes - df.shape[0]}")
    print(f"  Jugadores restantes: {df.shape[0]}")
    return df


def eliminar_columnas(df, columnas):
    df = df.copy()
    existentes = [c for c in columnas if c in df.columns]
    df = df.drop(columns=existentes)
    print(f"  Columnas eliminadas: {', '.join(existentes)}")
    return df


def codificar_posiciones_ponderadas(df, peso_main=1.0, peso_pos2=0.6, peso_pos3=0.3):
    df = df.copy()

    posiciones = pd.unique(df[["MAIN POSITION", "POSITION 2", "POSITION 3"]].values.ravel())
    posiciones = [p for p in posiciones if pd.notna(p)]

    for pos in posiciones:
        nombre = "POS_" + pos.upper().replace(" ", "_")
        df[nombre] = 0.0
        df.loc[df["POSITION 3"] == pos, nombre] = peso_pos3
        df.loc[df["POSITION 2"] == pos, nombre] = peso_pos2
        df.loc[df["MAIN POSITION"] == pos, nombre] = peso_main

    print(f"  Posiciones ponderadas: {len(posiciones)} columnas (main={peso_main}, pos2={peso_pos2}, pos3={peso_pos3})")
    return df


def one_hot_foot(df):
    df = df.copy()
    for valor in ["Derecho", "Izquierdo", "Ambidiestro"]:
        df["FOOT_" + valor.upper()] = (df["FOOT"] == valor).astype(int)
    print("  One-hot de FOOT: 3 columnas")
    return df


def one_hot_liga(df, col_id="LEAGUE_ID"):
    df = df.copy()

    ligas = sorted(df[col_id].dropna().unique())
    for liga in ligas:
        df[f"LEAGUE_{liga}"] = (df[col_id] == liga).astype(int)

    print(f"  One-hot de liga: {len(ligas)} columnas (a partir de {col_id})")
    return df


def crear_peak_age(df):
    df = df.copy()
    df["PEAK_AGE"] = df["AGE"].between(24, 28).astype(int)
    print(f"  PEAK_AGE creado ({df['PEAK_AGE'].sum()} jugadores en pico)")
    return df


def crear_young_prospect(df):
    df = df.copy()
    df["AGE"] = pd.to_numeric(df["AGE"], errors="coerce")
    df["YOUNG_PROSPECT"] = (df["AGE"] <= 23).astype(int)
    print(f"  YOUNG_PROSPECT creado ({df['YOUNG_PROSPECT'].sum()} jovenes promesas)")
    return df


def normalizar_por_liga(df, columnas):
    df = df.copy()
    for col in columnas:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[f"{col}_Z_LIGA"] = df.groupby("LEAGUE")[col].transform(
            lambda x: (x - x.mean()) / x.std(ddof=0)
        )
    print(f"  Normalizadas por liga (z-score): {len(columnas)} columnas")
    return df


def capar_zscores(df, limite=3.0):
    df = df.copy()
    cols_z = [c for c in df.columns if c.endswith("_Z_LIGA")]
    df[cols_z] = df[cols_z].clip(lower=-limite, upper=limite)
    print(f"  Z-scores capados a +/-{limite}: {len(cols_z)} columnas")
    return df


def crear_log_market_value(df):
    df = df.copy()
    df["MARKET VALUE"] = pd.to_numeric(df["MARKET VALUE"], errors="coerce")
    df["LOG_MARKET_VALUE"] = np.log1p(df["MARKET VALUE"])
    print(f"  LOG_MARKET_VALUE creado (skew {df['LOG_MARKET_VALUE'].skew():.2f})")
    return df


def asignar_rol(df, roles=ROLES):
    df = df.copy()
    mapa = {}
    for rol, posiciones in roles.items():
        for pos in posiciones:
            mapa[pos] = rol
    df["ROL"] = df["MAIN POSITION"].map(mapa)
    print("  Roles asignados:")
    print(df["ROL"].value_counts().to_string().replace("\n", "\n    "))
    return df


def calcular_score_rol(df, metricas_rol=METRICAS_ROL):
    df = df.copy()
    df["SCORE_ROL"] = np.nan
    for rol, pesos_dict in metricas_rol.items():
        mask = df["ROL"] == rol
        cols = [c for c in pesos_dict if c in df.columns]
        pesos = np.array([pesos_dict[c] for c in cols])
        valores = df.loc[mask, cols].fillna(0).to_numpy()
        df.loc[mask, "SCORE_ROL"] = (valores * pesos).sum(axis=1) / pesos.sum()
    print("  SCORE_ROL calculado con pesos por rol")
    return df


def ponderar_score_por_minutos(df, min_referencia=2000):
    df = df.copy()
    factor = (df["MINS"] / min_referencia).clip(upper=1.0)
    df["SCORE_ROL_AJUSTADO"] = df["SCORE_ROL"] * factor
    print(f"  SCORE_ROL_AJUSTADO calculado (referencia {min_referencia} min)")
    return df


def pipeline_rellenar(config):
    print("Carga de datos")
    print("-" * 40)
    datos = cargar_datos(config["archivo_datos"], config["nombre_hoja"])

    print("\nRelleno de posiciones")
    print("-" * 40)
    datos = rellenar_position_2(datos)
    datos = rellenar_position_3(datos)

    print("\nRelleno de height")
    print("-" * 40)
    datos = rellenar_height_mediana(datos)

    print("\nRelleno de valores fijos")
    print("-" * 40)
    datos = rellenar_foot(datos)
    datos = rellenar_contract_until(datos)
    datos = rellenar_team_joined(datos)
    datos = rellenar_previous_team(datos)

    print("\nFiltrado por minutos jugados")
    print("-" * 40)
    datos = filtrar_mins(datos, minimo=900)

    print("\nExclusion de porteros y valores de mercado nulos")
    print("-" * 40)
    datos = excluir_porteros(datos)
    datos = excluir_market_value_cero(datos)

    print("\nCalculo de diferencia de valor de mercado")
    print("-" * 40)
    datos = calcular_diferencia_market_value(datos)

    print("\nCalculo de porcentaje de titularidades")
    print("-" * 40)
    datos = calcular_start_pct(datos)

    print("\nCalculo de porcentaje de minutos disponibles jugados")
    print("-" * 40)
    datos = calcular_available_min_pct(datos)

    print("\nCalculo de contribuciones por 90 minutos")
    print("-" * 40)
    datos = calcular_contribuciones_90(datos)

    print("\nCalculo de conversion de gol")
    print("-" * 40)
    datos = calcular_goal_conversion(datos)

    print("\nCalculo de situacion contractual")
    print("-" * 40)
    datos = calcular_situacion_contractual(datos, fecha_referencia="30/06/2026")

    print("\nCalculo de acciones defensivas")
    print("-" * 40)
    datos = calcular_defensive_actions(datos)

    print("\nCalculo de creacion de ocasiones")
    print("-" * 40)
    datos = calcular_chance_creation(datos)

    print("\nCalculo de amenaza con balon (regate + faltas recibidas)")
    print("-" * 40)
    datos = calcular_dribbling_threat(datos)

    print("\nCodificacion de posicion ponderada y pie")
    print("-" * 40)
    datos = codificar_posiciones_ponderadas(datos)
    datos = one_hot_foot(datos)
    print("\nCodificacion one-hot de liga")
    print("-" * 40)
    datos = one_hot_liga(datos, col_id="LEAGUE_ID")
    datos = crear_peak_age(datos)
    datos = crear_young_prospect(datos)

    print("\nNormalizacion de metricas por liga (z-score)")
    print("-" * 40)
    datos = normalizar_por_liga(datos, COLS_RENDIMIENTO)
    datos = capar_zscores(datos, limite=7.0)

    datos = crear_log_market_value(datos)

    print("\nEliminacion de columnas no usadas")
    print("-" * 40)
    datos = eliminar_columnas(datos, ["Despo", "RATING", "NATIONALITY", "PREVIOUS TEAM"])

    print("\nAsignacion de rol y score de rendimiento")
    print("-" * 40)
    datos = asignar_rol(datos)
    datos = calcular_score_rol(datos)
    datos = ponderar_score_por_minutos(datos, min_referencia=2000)

    print("\nPipeline completado")
    return datos
