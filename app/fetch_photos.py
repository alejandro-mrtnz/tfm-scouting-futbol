"""Descarga (una sola vez) la URL de la foto de perfil de cada jugador desde
Transfermarkt y la guarda en artifacts/fotos.json como {PLAYER_ID: url_foto}.

No se ejecuta como parte de train.py ni de la app: es un paso manual y lento
(una peticion HTTP por jugador) que solo hay que repetir si quieres refrescar
las fotos. No descarga ni redistribuye las imagenes, solo guarda el enlace
(igual que hacen la mayoria de webs de scouting con fotos de Transfermarkt).

Uso:
    python fetch_photos.py            # todos los jugadores que falten
    python fetch_photos.py --limit 20 # prueba rapida con los primeros 20
"""
import json
import re
import sys
import time

import pandas as pd
import requests

from paths import ARTIFACTS_DIR, BASE_DIR

CSV_CRUDO = BASE_DIR / "transfermarkt_7_ligas_profiles_incremental_2025_26.csv"
SALIDA = ARTIFACTS_DIR / "fotos.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
PAUSA_SEGUNDOS = 0.6
REINTENTOS = 2
GUARDAR_CADA = 50


def extraer_url_foto(html, player_id):
    patron = rf'https://img\.a\.transfermarkt\.technology/portrait/header/{player_id}-\d+\.jpg\?lm=\d+'
    encontrado = re.search(patron, html)
    if encontrado:
        return encontrado.group(0)

    # fallback: variante "big" si no hay "header" para este jugador
    patron_big = rf'https://img\.a\.transfermarkt\.technology/portrait/big/{player_id}-\d+\.jpg\?lm=\d+'
    encontrado = re.search(patron_big, html)
    return encontrado.group(0) if encontrado else None


def obtener_foto(sesion, tm_url, player_id):
    for intento in range(REINTENTOS + 1):
        try:
            r = sesion.get(tm_url, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                return extraer_url_foto(r.text, player_id)
        except requests.RequestException:
            pass
        time.sleep(1.0)
    return None


def main(limite=None):
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    datos = pd.read_pickle(ARTIFACTS_DIR / "datos.pkl")
    crudo = pd.read_csv(CSV_CRUDO)
    mapa_url = dict(zip(crudo["tm_player_id"].astype("int64"), crudo["tm_url"]))

    fotos = {}
    if SALIDA.exists():
        with open(SALIDA, encoding="utf-8") as f:
            fotos = json.load(f)

    ids_objetivo = datos["PLAYER_ID"].dropna().astype("int64").unique().tolist()
    ids_pendientes = [pid for pid in ids_objetivo if not fotos.get(str(pid))]
    if limite is not None:
        ids_pendientes = ids_pendientes[:limite]

    print(f"Jugadores objetivo: {len(ids_objetivo)}  |  pendientes: {len(ids_pendientes)}")

    sesion = requests.Session()
    ok, fallidos = 0, 0
    for i, player_id in enumerate(ids_pendientes, start=1):
        tm_url = mapa_url.get(player_id)
        if not tm_url:
            fotos[str(player_id)] = None
            fallidos += 1
        else:
            url_foto = obtener_foto(sesion, tm_url, player_id)
            fotos[str(player_id)] = url_foto
            if url_foto:
                ok += 1
            else:
                fallidos += 1

        if i % 10 == 0 or i == len(ids_pendientes):
            print(f"  [{i}/{len(ids_pendientes)}] ok={ok} fallidos={fallidos}", flush=True)

        if i % GUARDAR_CADA == 0:
            with open(SALIDA, "w", encoding="utf-8") as f:
                json.dump(fotos, f)

        time.sleep(PAUSA_SEGUNDOS)

    with open(SALIDA, "w", encoding="utf-8") as f:
        json.dump(fotos, f)

    print(f"\nCompletado. Guardado en: {SALIDA}")
    print(f"Total en fichero: {len(fotos)}  |  con foto: {sum(1 for v in fotos.values() if v)}")


if __name__ == "__main__":
    limite = None
    if "--limit" in sys.argv:
        limite = int(sys.argv[sys.argv.index("--limit") + 1])
    main(limite)
