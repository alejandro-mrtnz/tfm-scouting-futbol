# Ejecutar TFM Scouting en Windows

## Opción recomendada: versión precompilada

1. Abre la sección [Releases](https://github.com/alejandro-mrtnz/tfm-scouting-futbol/releases).
2. En la versión más reciente, descarga `TFM_Scouting-Windows-v1.0.0.zip`.
3. Descomprime todo el contenido. No ejecutes el programa directamente desde el ZIP.
4. Comprueba que la estructura resultante sea:

```text
TFM_Scouting-Windows-v1.0.0/
├── TFM_Scouting.exe
├── EJECUTAR_APLICACION.md
└── artifacts/
    ├── X.npy
    ├── datos.pkl
    ├── fotos.json
    ├── meta.json
    └── pred_oof_log.npy
```

5. Haz doble clic en `TFM_Scouting.exe`.
6. Espera unos segundos hasta que se abra la aplicación en el navegador. La interfaz se sirve únicamente desde el propio ordenador.

## Advertencia de Windows

El ejecutable no está firmado digitalmente. Windows SmartScreen puede mostrar una advertencia. Verifica que el ZIP procede de la Release oficial de este repositorio antes de seleccionar **Más información** y **Ejecutar de todas formas**.

## Si la aplicación no se abre

- Mantén `TFM_Scouting.exe` y `artifacts/` dentro de la misma carpeta.
- Extrae el ZIP antes de ejecutar el programa.
- Espera al menos 20 segundos durante el primer arranque.
- Comprueba que el antivirus no haya puesto el ejecutable en cuarentena.
- Cierra procesos anteriores de `TFM_Scouting.exe` desde el Administrador de tareas y vuelve a intentarlo.

## Alternativa desde el código fuente

Consulta la sección **Puesta en marcha** del [README](README.md) para instalar Python, regenerar los modelos e iniciar la aplicación con Streamlit.
