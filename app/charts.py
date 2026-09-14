"""Grafico radar (spider) interactivo para comparar dos jugadores en sus metricas clave."""
import plotly.graph_objects as go

from pipeline import METRICAS_ROL

# Paleta categorica validada (slots 1 y 2): pasa los 6 checks (banda de luminosidad,
# suelo de croma, separacion CVD >= 8, piso de vision normal >= 15, contraste >= 3:1).
COLOR_A = "#2a78d6"
COLOR_B = "#eb6834"
COLOR_GRID = "#e1e0d9"
COLOR_AXIS = "#c3c2b7"
COLOR_TEXT_PRIMARY = "#0b0b0b"
COLOR_TEXT_MUTED = "#898781"
COLOR_SURFACE = "#fcfcfb"

METRICAS_GENERICAS = [
    "GOALS_90", "ASSISTS_90", "SHOTS PER MATCH", "KEY PASSES",
    "SUCCESSFUL PASSES (%)", "DRIBBLE_OF", "TACKLE", "INTERCEPTION",
]


def _rgba(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def metricas_por_defecto(df, jugador_a, jugador_b, maximo=8):
    rol_a = df.loc[df["NAME"] == jugador_a, "ROL"]
    rol_b = df.loc[df["NAME"] == jugador_b, "ROL"]
    rol_a = rol_a.iloc[0] if len(rol_a) else None
    rol_b = rol_b.iloc[0] if len(rol_b) else None

    if rol_a is not None and rol_a == rol_b and rol_a in METRICAS_ROL:
        metricas = [c.replace("_Z_LIGA", "") for c in METRICAS_ROL[rol_a]]
    else:
        metricas = METRICAS_GENERICAS

    return [m for m in metricas if m in df.columns][:maximo]


def preparar_datos_radar(df, jugador_a, jugador_b, metricas):
    idx_a = df.index[df["NAME"] == jugador_a]
    idx_b = df.index[df["NAME"] == jugador_b]
    if len(idx_a) == 0 or len(idx_b) == 0:
        return None
    fila_a = df.loc[idx_a[0]]
    fila_b = df.loc[idx_b[0]]

    valores_a, valores_b, percentil_a, percentil_b = [], [], [], []
    for m in metricas:
        rango = df[m].rank(pct=True) * 100
        valores_a.append(float(fila_a[m]))
        valores_b.append(float(fila_b[m]))
        percentil_a.append(float(rango.loc[idx_a[0]]))
        percentil_b.append(float(rango.loc[idx_b[0]]))

    return {
        "categorias": list(metricas),
        "valores_a": valores_a, "valores_b": valores_b,
        "percentil_a": percentil_a, "percentil_b": percentil_b,
    }


def _trazo_jugador(nombre, categorias, percentiles, valores, color):
    categorias_cerrado = categorias + categorias[:1]
    percentiles_cerrado = percentiles + percentiles[:1]
    valores_cerrado = valores + valores[:1]

    return go.Scatterpolar(
        r=percentiles_cerrado,
        theta=categorias_cerrado,
        name=nombre,
        mode="lines+markers",
        line=dict(color=color, width=2),
        marker=dict(size=9, color=color, line=dict(color=COLOR_SURFACE, width=1.5)),
        fill="toself",
        fillcolor=_rgba(color, 0.12),
        customdata=valores_cerrado,
        hovertemplate=(
            "<b>%{theta}</b><br>"
            "Valor: %{customdata:.2f}<br>"
            "Percentil: %{r:.0f}"
            f"<extra>{nombre}</extra>"
        ),
    )


def render_radar_chart(datos_radar, nombre_a, nombre_b):
    categorias = datos_radar["categorias"]

    fig = go.Figure()
    fig.add_trace(_trazo_jugador(
        nombre_a, categorias, datos_radar["percentil_a"], datos_radar["valores_a"], COLOR_A))
    fig.add_trace(_trazo_jugador(
        nombre_b, categorias, datos_radar["percentil_b"], datos_radar["valores_b"], COLOR_B))

    fig.update_layout(
        polar=dict(
            bgcolor=COLOR_SURFACE,
            radialaxis=dict(
                range=[0, 100], tickvals=[20, 40, 60, 80, 100],
                gridcolor=COLOR_GRID, linecolor=COLOR_AXIS,
                tickfont=dict(color=COLOR_TEXT_MUTED, size=10),
            ),
            angularaxis=dict(
                gridcolor=COLOR_GRID, linecolor=COLOR_AXIS,
                tickfont=dict(color=COLOR_TEXT_PRIMARY, size=12),
            ),
        ),
        showlegend=True,
        legend=dict(font=dict(color=COLOR_TEXT_PRIMARY), orientation="h",
                    yanchor="bottom", y=1.05, xanchor="center", x=0.5),
        paper_bgcolor=COLOR_SURFACE,
        plot_bgcolor=COLOR_SURFACE,
        margin=dict(l=110, r=110, t=90, b=50),
        height=600,
    )
    return fig
