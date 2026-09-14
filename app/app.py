"""App Streamlit para testear el recomendador/mejorador/detector de gangas.

Lanzar con:  streamlit run app.py
Requiere haber ejecutado antes train.py (genera artifacts/ en la raíz).
"""
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from charts import metricas_por_defecto, preparar_datos_radar, render_radar_chart
from paths import ARTIFACTS_DIR
from pipeline import COLS_RENDIMIENTO
from similarity import gangas_similares, mejorar_jugador, recomendar
from valuation import analizar_equipo, analizar_valor, ficha_jugador

RUTA_SIN_FOTO = Path(__file__).resolve().parent / "assets" / "sin_foto.png"

COLUMNAS_LEGIBLES_DEFECTO = [
    "NAME", "TEAM", "LEAGUE", "MAIN POSITION", "ROL", "AGE", "HEIGHT", "FOOT",
    "MINS", "STARTS", "GOALS", "ASSIST", "GOALS_90", "ASSISTS_90",
    "SHOTS PER MATCH", "KEY PASSES", "SUCCESSFUL PASSES (%)", "TACKLE",
    "INTERCEPTION", "MARKET VALUE", "VALOR_PREDICHO", "CONTRACT UNTIL",
]

st.set_page_config(page_title="TFM - Scouting de jugadores", layout="wide")


def _normaliza(texto):
    texto = str(texto).lower()
    texto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in texto if not unicodedata.combining(c))


@st.cache_resource(show_spinner=False)
def cargar_artefactos():
    datos = pd.read_pickle(ARTIFACTS_DIR / "datos.pkl")
    X = np.load(ARTIFACTS_DIR / "X.npy")
    pred_oof_log = np.load(ARTIFACTS_DIR / "pred_oof_log.npy")
    with open(ARTIFACTS_DIR / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    predicciones = {"pred_oof_log": pred_oof_log}

    ruta_fotos = ARTIFACTS_DIR / "fotos.json"
    fotos = {}
    if ruta_fotos.exists():
        with open(ruta_fotos, encoding="utf-8") as f:
            fotos = json.load(f)

    return datos, X, meta, predicciones, fotos


def foto_de(df, nombre_jugador):
    fila = df.loc[df["NAME"] == nombre_jugador, "PLAYER_ID"]
    if len(fila) == 0:
        return str(RUTA_SIN_FOTO)
    url = fotos.get(str(int(fila.iloc[0])))
    return url if url else str(RUTA_SIN_FOTO)


def selector_jugador(df, key, placeholder="Escribe parte del nombre"):
    texto = st.text_input("Buscar jugador", key=f"buscar_{key}", placeholder=placeholder)
    if not texto:
        return None
    opciones = sorted([n for n in df["NAME"] if _normaliza(texto) in _normaliza(n)])[:50]
    if not opciones:
        st.warning("Sin coincidencias.")
        return None
    return st.selectbox("Jugador", opciones, key=f"select_{key}")


if not (ARTIFACTS_DIR / "datos.pkl").exists():
    st.error(
        "No se han encontrado los artefactos entrenados. "
        "Ejecuta `python app/train.py` desde la raíz del proyecto."
    )
    st.stop()

datos, X, meta, predicciones, fotos = cargar_artefactos()

with st.sidebar:
    st.header("TFM - Panel de control")
    st.metric("Jugadores en el dataset", meta["n_jugadores"])
    st.caption(f"Generado: {meta['generado']}")
    modelo = meta["modelo_elegido"]
    info_modelo = meta["modelos"][modelo]
    st.caption(
        f"Modelo de rendimiento ({modelo}): R2={info_modelo['r2']}  |  "
        f"MAE={info_modelo['mae_millones']} M"
    )
    if "modelo_realista" in meta:
        ir = meta["modelo_realista"]
        st.caption(f"Modelo realista (con historico): R2={ir['r2']}  |  MAE={ir['mae_millones']} M")

(tab_similares, tab_mejorar, tab_gangas, tab_ficha, tab_equipo, tab_mercado,
 tab_radar, tab_datos) = st.tabs(
    ["Jugadores similares", "Mejorar jugador", "Gangas parecidas",
     "Ficha de jugador", "Analisis de equipo", "Explorador de mercado",
     "Comparar jugadores", "Datos depurados"]
)

with tab_similares:
    st.subheader("Buscar jugadores con perfil similar")
    jugador = selector_jugador(datos, "similares")
    col1, col2, col3 = st.columns(3)
    n = col1.slider("Numero de resultados", 3, 30, 10, key="n_similares")
    presupuesto = col2.number_input("Presupuesto max (M, 0 = sin limite)", min_value=0.0, value=0.0, key="pres_similares")
    edad_max = col3.number_input("Edad max (0 = sin limite)", min_value=0, value=0, key="edad_similares")
    contrato = st.checkbox("Solo contrato acabandose", key="contrato_similares")

    if jugador and st.button("Buscar similares", type="primary"):
        r = recomendar(
            datos, X, jugador, n=n,
            presupuesto_max=presupuesto or None,
            edad_max=edad_max or None,
            solo_contrato_acabando=contrato,
        )
        if r is not None:
            st.dataframe(r, width='stretch')

with tab_mejorar:
    st.subheader("Encontrar un jugador que mejore a otro dentro de presupuesto")
    jugador = selector_jugador(datos, "mejorar")
    col1, col2, col3 = st.columns(3)
    n = col1.slider("Numero de resultados", 3, 30, 10, key="n_mejorar")
    presupuesto = col2.number_input("Presupuesto max (M, 0 = sin limite)", min_value=0.0, value=0.0, key="pres_mejorar")
    ratio_valor = col3.slider("Max. veces el valor actual", 1.0, 6.0, 3.0, 0.5, key="ratio_mejorar")
    contrato = st.checkbox("Solo contrato acabandose", key="contrato_mejorar")

    if jugador and st.button("Buscar mejoras", type="primary"):
        r = mejorar_jugador(
            datos, X, jugador, n=n,
            presupuesto_max=presupuesto or None,
            max_ratio_valor=ratio_valor,
            solo_contrato_acabando=contrato,
        )
        if r is not None:
            st.dataframe(r, width='stretch')

with tab_gangas:
    st.subheader("Jugadores parecidos que ademas son gangas segun el modelo")
    jugador = selector_jugador(datos, "gangas")
    col1, col2, col3 = st.columns(3)
    n = col1.slider("Numero de resultados", 3, 30, 10, key="n_gangas")
    presupuesto = col2.number_input("Presupuesto max (M, 0 = sin limite)", min_value=0.0, value=0.0, key="pres_gangas")
    ratio_min = col3.slider("Ratio minimo (predicho/real)", 1.0, 3.0, 1.3, 0.1, key="ratio_gangas")
    mismo_rol = st.checkbox("Mismo rol", value=True, key="rol_gangas")

    if jugador and st.button("Buscar gangas parecidas", type="primary"):
        r = gangas_similares(
            datos, X, jugador, n=n,
            ratio_min=ratio_min,
            presupuesto_max=presupuesto or None,
            mismo_rol=mismo_rol,
        )
        if r is not None:
            st.dataframe(r, width='stretch')

with tab_ficha:
    st.subheader("Ficha individual de un jugador")
    jugador = selector_jugador(datos, "ficha")
    if jugador:
        ficha = ficha_jugador(datos, jugador)
        if ficha is not None:
            col_foto, col_tabla = st.columns([1, 3])
            with col_foto:
                st.image(foto_de(datos, jugador), width=180)
            with col_tabla:
                st.table(ficha)

with tab_equipo:
    st.subheader("Analisis de plantilla")
    equipos = sorted(datos["TEAM"].dropna().unique())
    equipo = st.selectbox("Equipo", equipos, key="select_equipo")
    orden = st.selectbox(
        "Ordenar por",
        ["DIFERENCIA_VALOR", "RATIO_VALOR", "SCORE_ROL_AJUSTADO", "MARKET VALUE", "AGE"],
        key="orden_equipo",
    )
    if st.button("Analizar equipo", type="primary"):
        r = analizar_equipo(datos, equipo, ordenar_por=orden)
        if r is not None:
            st.dataframe(r, width='stretch')

with tab_mercado:
    st.subheader("Explorador de infra / sobrevalorados")
    col1, col2 = st.columns(2)
    direccion = col1.radio("Buscar", ["infravalorados", "sobrevalorados"], key="direccion_mercado")
    metrica = col2.radio("Metrica", ["absoluto", "ratio", "log_residuo"], key="metrica_mercado")

    col3, col4, col5, col6 = st.columns(4)
    n = col3.slider("Numero de resultados", 5, 50, 20, key="n_mercado")
    valor_min = col4.number_input("Valor min (M, 0 = sin limite)", min_value=0.0, value=0.0, key="vmin_mercado")
    valor_max = col5.number_input("Valor max (M, 0 = sin limite)", min_value=0.0, value=0.0, key="vmax_mercado")
    edad_max = col6.number_input("Edad max (0 = sin limite)", min_value=0, value=0, key="edad_mercado")

    if st.button("Explorar mercado", type="primary"):
        r = analizar_valor(
            datos, predicciones, n=n,
            direccion=direccion, metrica=metrica,
            valor_min=valor_min or None,
            valor_max=valor_max or None,
            edad_max=edad_max or None,
        )
        if r is not None:
            st.dataframe(r, width='stretch')

with tab_radar:
    st.subheader("Comparar dos jugadores")
    col_a, col_b = st.columns(2)
    with col_a:
        jugador_a = selector_jugador(datos, "radar_a", placeholder="Jugador A")
    with col_b:
        jugador_b = selector_jugador(datos, "radar_b", placeholder="Jugador B")

    if jugador_a and jugador_b:
        if jugador_a == jugador_b:
            st.warning("Elige dos jugadores distintos.")
        else:
            defecto = metricas_por_defecto(datos, jugador_a, jugador_b)
            metricas = st.multiselect(
                "Metricas a comparar (recomendado 5-10)",
                options=COLS_RENDIMIENTO,
                default=defecto,
                key="metricas_radar",
            )
            if len(metricas) >= 3 and st.button("Comparar", type="primary"):
                datos_radar = preparar_datos_radar(datos, jugador_a, jugador_b, metricas)
                if datos_radar is None:
                    st.warning("No se ha podido preparar la comparativa.")
                else:
                    col_foto_a, col_chart, col_foto_b = st.columns([1, 3, 1])
                    with col_foto_a:
                        st.image(foto_de(datos, jugador_a), width='stretch')
                        st.caption(jugador_a)
                    with col_chart:
                        fig = render_radar_chart(datos_radar, jugador_a, jugador_b)
                        st.plotly_chart(fig, width='stretch')
                    with col_foto_b:
                        st.image(foto_de(datos, jugador_b), width='stretch')
                        st.caption(jugador_b)
                    st.caption(
                        "El area del radar es el percentil (0-100) de cada metrica "
                        "sobre el conjunto total de jugadores filtrados (900+ minutos, "
                        "sin porteros). Pasa el raton sobre un punto para ver el valor real."
                    )
                    tabla = pd.DataFrame({
                        "Metrica": datos_radar["categorias"],
                        jugador_a: datos_radar["valores_a"],
                        jugador_b: datos_radar["valores_b"],
                    })
                    st.dataframe(tabla, width='stretch', hide_index=True)
            elif len(metricas) < 3:
                st.info("Selecciona al menos 3 metricas.")

with tab_datos:
    st.subheader("Datos depurados (post-limpieza y feature engineering)")
    st.caption(f"{datos.shape[0]} jugadores x {datos.shape[1]} columnas")

    col1, col2, col3 = st.columns(3)
    ligas_sel = col1.multiselect("Liga", sorted(datos["LEAGUE"].dropna().unique()), key="filtro_liga")
    roles_sel = col2.multiselect("Rol", sorted(datos["ROL"].dropna().unique()), key="filtro_rol")
    equipos_sel = col3.multiselect("Equipo", sorted(datos["TEAM"].dropna().unique()), key="filtro_equipo")

    col4, col5, col6 = st.columns(3)
    edad_min, edad_max = col4.slider(
        "Edad", int(datos["AGE"].min()), int(datos["AGE"].max()),
        (int(datos["AGE"].min()), int(datos["AGE"].max())), key="filtro_edad")
    mins_min, mins_max = col5.slider(
        "Minutos jugados", int(datos["MINS"].min()), int(datos["MINS"].max()),
        (int(datos["MINS"].min()), int(datos["MINS"].max())), key="filtro_mins")
    valor_min, valor_max = col6.slider(
        "Valor de mercado (M)", 0.0, float(datos["MARKET VALUE"].max()),
        (0.0, float(datos["MARKET VALUE"].max())), key="filtro_valor")

    nombre_buscar = st.text_input("Buscar por nombre", key="filtro_nombre")

    filtrado = datos.copy()
    if ligas_sel:
        filtrado = filtrado[filtrado["LEAGUE"].isin(ligas_sel)]
    if roles_sel:
        filtrado = filtrado[filtrado["ROL"].isin(roles_sel)]
    if equipos_sel:
        filtrado = filtrado[filtrado["TEAM"].isin(equipos_sel)]
    filtrado = filtrado[filtrado["AGE"].between(edad_min, edad_max)]
    filtrado = filtrado[filtrado["MINS"].between(mins_min, mins_max)]
    filtrado = filtrado[filtrado["MARKET VALUE"].between(valor_min, valor_max)]
    if nombre_buscar:
        filtrado = filtrado[filtrado["NAME"].apply(
            lambda n: _normaliza(nombre_buscar) in _normaliza(n))]

    mostrar_todas = st.checkbox("Mostrar todas las columnas", key="mostrar_todas_columnas")
    if mostrar_todas:
        columnas_mostrar = list(filtrado.columns)
    else:
        columnas_mostrar = st.multiselect(
            "Columnas a mostrar", options=list(datos.columns),
            default=COLUMNAS_LEGIBLES_DEFECTO, key="columnas_datos",
        )

    st.caption(f"Mostrando {filtrado.shape[0]} jugadores")
    st.dataframe(filtrado[columnas_mostrar], width='stretch', hide_index=True)

    st.download_button(
        "Descargar seleccion (CSV)",
        data=filtrado[columnas_mostrar].to_csv(index=False).encode("utf-8"),
        file_name="jugadores_filtrados.csv",
        mime="text/csv",
    )
