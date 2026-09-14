"""Motor de similitud entre jugadores.

Portado de MODELIZACION.ipynb (celdas 63-64, 88) sin cambiar la logica.
"""
import numpy as np


def preparar_matriz_similitud(df, peso_rendimiento=0.5, peso_posicion=0.5):
    df = df.copy().reset_index(drop=True)

    cols_z = [c for c in df.columns if c.endswith("_Z_LIGA")]
    cols_pos = [c for c in df.columns if c.startswith("POS_")]

    def estandarizar(M):
        return (M - M.mean(axis=0)) / (M.std(axis=0) + 1e-9)

    X_rend = estandarizar(df[cols_z].fillna(0).to_numpy()) * (peso_rendimiento ** 0.5)
    X_pos = estandarizar(df[cols_pos].fillna(0).to_numpy()) * (peso_posicion ** 0.5)

    X = np.hstack([X_rend, X_pos])
    print(f"  Matriz lista: {X.shape[0]} jugadores x {X.shape[1]} dims "
          f"(rend={len(cols_z)} pos={len(cols_pos)}, pesos {peso_rendimiento}/{peso_posicion})")
    return df, X


def recomendar(df, X, nombre_jugador, n=10,
               presupuesto_max=None,
               edad_min=None, edad_max=None,
               ligas_incluir=None, ligas_excluir=None,
               solo_contrato_acabando=False,
               misma_posicion=False):

    idx = df.index[df["NAME"] == nombre_jugador]
    if len(idx) == 0:
        print(f"  No encontrado: {nombre_jugador}")
        return None
    idx = idx[0]

    distancias = np.sqrt(((X - X[idx]) ** 2).sum(axis=1))

    res = df.copy()
    res["DISTANCIA"] = distancias
    res = res[res.index != idx]

    if presupuesto_max is not None:
        res = res[res["MARKET VALUE"] <= presupuesto_max]
    if edad_min is not None:
        res = res[res["AGE"] >= edad_min]
    if edad_max is not None:
        res = res[res["AGE"] <= edad_max]
    if ligas_incluir is not None:
        res = res[res["LEAGUE"].isin(ligas_incluir)]
    if ligas_excluir is not None:
        res = res[~res["LEAGUE"].isin(ligas_excluir)]
    if solo_contrato_acabando:
        res = res[res["CONTRACT_OPPORTUNITY"] == 1]
    if misma_posicion:
        res = res[res["MAIN POSITION"] == df.loc[idx, "MAIN POSITION"]]

    if res.empty:
        print("  Ningun jugador cumple los filtros. Prueba a relajarlos.")
        return None

    cols = ["NAME", "TEAM", "LEAGUE", "MAIN POSITION", "AGE",
            "MARKET VALUE", "CONTRACT_OPPORTUNITY", "DISTANCIA"]
    return res.nsmallest(n, "DISTANCIA")[cols].reset_index(drop=True)


def mejorar_jugador(df, X, nombre_jugador, n=10,
                    n_similares=60,
                    max_ratio_valor=3.0,
                    margen_mejora_max=None,
                    presupuesto_max=None,
                    solo_contrato_acabando=False):

    idx = df.index[df["NAME"] == nombre_jugador]
    if len(idx) == 0:
        print(f"  No encontrado: {nombre_jugador}")
        return None
    idx = idx[0]

    score_ref = df.loc[idx, "SCORE_ROL_AJUSTADO"]
    valor_ref = df.loc[idx, "MARKET VALUE"]
    rol_ref = df.loc[idx, "ROL"]

    distancias = np.sqrt(((X - X[idx]) ** 2).sum(axis=1))
    res = df.copy()
    res["DISTANCIA"] = distancias
    res = res[res.index != idx]
    res = res.nsmallest(n_similares, "DISTANCIA")

    res = res[res["ROL"] == rol_ref]

    res = res[res["SCORE_ROL_AJUSTADO"] > score_ref]
    res["MEJORA_SCORE"] = res["SCORE_ROL_AJUSTADO"] - score_ref

    if max_ratio_valor is not None and valor_ref > 0:
        res = res[res["MARKET VALUE"] <= valor_ref * max_ratio_valor]
    if presupuesto_max is not None:
        res = res[res["MARKET VALUE"] <= presupuesto_max]
    if margen_mejora_max is not None:
        res = res[res["MEJORA_SCORE"] <= margen_mejora_max]
    if solo_contrato_acabando:
        res = res[res["CONTRACT_OPPORTUNITY"] == 1]

    if res.empty:
        print("  Ningun jugador mejora al tuyo dentro de los limites. Prueba a relajarlos.")
        return None

    cols = ["NAME", "TEAM", "LEAGUE", "MAIN POSITION", "AGE", "MARKET VALUE",
            "SCORE_ROL_AJUSTADO", "MEJORA_SCORE", "DISTANCIA"]
    return res.nlargest(n, "MEJORA_SCORE")[cols].reset_index(drop=True)


def gangas_similares(df, X, nombre_jugador, n=10,
                     n_similares=80,
                     ratio_min=1.3,
                     presupuesto_max=None,
                     mismo_rol=True):

    idx = df.index[df["NAME"] == nombre_jugador]
    if len(idx) == 0:
        print(f"  No encontrado: {nombre_jugador}")
        return None
    idx = idx[0]

    distancias = np.sqrt(((X - X[idx]) ** 2).sum(axis=1))
    res = df.copy()
    res["DISTANCIA"] = distancias
    res = res[res.index != idx].nsmallest(n_similares, "DISTANCIA")

    if mismo_rol:
        res = res[res["ROL"] == df.loc[idx, "ROL"]]

    res = res[res["RATIO_VALOR"] >= ratio_min]

    if presupuesto_max is not None:
        res = res[res["MARKET VALUE"] <= presupuesto_max]

    if res.empty:
        print("  Ninguna ganga parecida dentro de los limites. Prueba a relajarlos.")
        return None

    cols = ["NAME", "TEAM", "LEAGUE", "MAIN POSITION", "AGE",
            "MARKET VALUE", "VALOR_PREDICHO", "RATIO_VALOR", "DISTANCIA"]
    return res.nlargest(n, "RATIO_VALOR")[cols].reset_index(drop=True)
