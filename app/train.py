"""Ejecuta el pipeline completo (limpieza -> features -> similitud -> modelo de valor)
y guarda los artefactos en artifacts/ para que la app Streamlit arranque al instante.

Uso:
    python train.py

Se puede relanzar en cualquier momento (por ejemplo tras actualizar el Excel de origen)
para recalcular todo desde cero.
"""
import json
from datetime import datetime

import numpy as np

from paths import ARTIFACTS_DIR, DATA_FILE
from pipeline import pipeline_rellenar
from similarity import preparar_matriz_similitud
from valuation import (
    evaluar_modelo_realista, evaluar_modelos_cv,
    incorporar_predicciones, incorporar_valor_realista,
)

CONFIG = {
    "archivo_datos": str(DATA_FILE),
    "nombre_hoja": 0,
}


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("1. LIMPIEZA Y FEATURE ENGINEERING")
    print("=" * 60)
    datos = pipeline_rellenar(CONFIG)

    print("\n" + "=" * 60)
    print("2. MATRIZ DE SIMILITUD")
    print("=" * 60)
    datos, X = preparar_matriz_similitud(datos, peso_rendimiento=0.5, peso_posicion=0.5)

    print("\n" + "=" * 60)
    print("3. MODELO DE VALOR DE MERCADO (validacion cruzada)")
    print("=" * 60)
    predicciones = evaluar_modelos_cv(datos, k=5)

    print("\n" + "=" * 60)
    print("4. INCORPORAR PREDICCIONES AL DATASET")
    print("=" * 60)
    modelo_elegido = "XGBoost"
    datos = incorporar_predicciones(datos, predicciones[modelo_elegido])

    print("\n" + "=" * 60)
    print("4b. MODELO DE VALOR REALISTA (con historico, solo para la ficha)")
    print("=" * 60)
    prediccion_realista = evaluar_modelo_realista(datos, k=5)
    datos = incorporar_valor_realista(datos, prediccion_realista)

    print("\n" + "=" * 60)
    print("5. GUARDANDO ARTEFACTOS")
    print("=" * 60)
    datos.to_pickle(ARTIFACTS_DIR / "datos.pkl")
    np.save(ARTIFACTS_DIR / "X.npy", X)
    np.save(ARTIFACTS_DIR / "pred_oof_log.npy", predicciones[modelo_elegido]["pred_oof_log"])

    meta = {
        "generado": datetime.now().isoformat(timespec="seconds"),
        # Evita guardar una ruta absoluta del equipo que ejecuta el entrenamiento.
        "archivo_datos": DATA_FILE.name,
        "n_jugadores": int(datos.shape[0]),
        "modelo_elegido": modelo_elegido,
        "modelos": {
            nombre: {
                "r2": round(float(info["r2"]), 4),
                "mae_millones": round(float(info["mae_millones"]), 3),
            }
            for nombre, info in predicciones.items()
        },
        "modelo_realista": {
            "r2": round(float(prediccion_realista["r2"]), 4),
            "mae_millones": round(float(prediccion_realista["mae_millones"]), 3),
        },
    }
    with open(ARTIFACTS_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"  Guardado en: {ARTIFACTS_DIR}")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print("\nListo. Ya puedes lanzar la app (streamlit run app.py).")


if __name__ == "__main__":
    main()
