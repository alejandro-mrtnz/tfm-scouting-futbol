"""Resuelve rutas de datos/artefactos tanto en modo script como empaquetado (.exe).

En modo congelado (PyInstaller), sys.executable es el propio .exe: usamos su
carpeta como raiz del proyecto (ahi deben vivir el Excel de datos y artifacts/).
En modo script, la raiz del proyecto es la carpeta que contiene app/ (un nivel
por encima de este fichero).
"""
import sys
from pathlib import Path


def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


BASE_DIR = get_base_dir()
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_FILE = BASE_DIR / "Estadisticas Jugadores 2025-2026.xlsx"
