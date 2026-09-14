# Datos del proyecto

## Política de publicación

Los datasets originales y los artefactos derivados no se versionan. Contienen información recopilada de fuentes de terceros y deben obtenerse y utilizarse de acuerdo con sus condiciones aplicables. Las reglas de `.gitignore` evitan que se añadan por accidente al repositorio.

El código, los notebooks, la documentación y las figuras sí forman parte de la publicación.

## Ficheros de entrada

El flujo de depuración (`DEPURADOR.ipynb`) espera estos ficheros en la raíz:

| Fichero | Contenido |
| --- | --- |
| `DATOS TRANSFERMARKET.xlsx` | Datos biográficos, contractuales y de valor de mercado. |
| `transfermarkt_7_ligas_profiles_incremental_2025_26.csv` | Posiciones y máximo histórico de valor. |
| `whoscored_todas_ligas_jugadores_2025_26.xlsx` | Estadísticas de rendimiento por bloques. |

La aplicación y `app/train.py` usan como entrada consolidada:

```text
Estadisticas Jugadores 2025-2026.xlsx
```

## Artefactos generados

Al ejecutar `python app/train.py` se crea `artifacts/` con:

- `datos.pkl`: tabla procesada y predicciones.
- `X.npy`: matriz usada para similitud.
- `pred_oof_log.npy`: predicciones *out-of-fold* del modelo seleccionado.
- `meta.json`: métricas y metadatos de la ejecución.
- `fotos.json`: mapa opcional de identificadores a URL, generado por `app/fetch_photos.py`.

Estos ficheros son regenerables y no se versionan en Git. La distribución para Windows los incluye dentro del ZIP de la Release porque el ejecutable los necesita para funcionar; los ficheros de entrada originales permanecen excluidos. El script de fotografías realiza solicitudes a una fuente externa; úsalo de manera responsable y revisa las condiciones de esa fuente antes de ejecutarlo.

## Comprobaciones antes de actualizar los datos

- Mantener los nombres de columnas esperados por `app/pipeline.py`.
- Revisar manualmente los emparejamientos ambiguos de jugadores y equipos.
- Documentar la fecha de extracción y la temporada.
- Reentrenar y comparar las métricas antes de sustituir resultados anteriores.
- No subir credenciales, cookies, cabeceras privadas ni datos personales no necesarios.
