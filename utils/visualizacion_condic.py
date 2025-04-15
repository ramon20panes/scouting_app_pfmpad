# utils/visualizacion_condic.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
import io
import base64

def grafico_barras(df, metrica="Distancia", color="#d32f2f"):
    fig, ax = plt.subplots(figsize=(10, 4))
    df_plot = df[["Jornada", metrica]].dropna().sort_values("Jornada")
    ax.bar(df_plot["Jornada"].astype(str), df_plot[metrica], color=color)
    ax.set_title(f"{metrica} por Jornada", fontsize=12)
    ax.set_xlabel("Jornada")
    ax.set_ylabel(metrica)
    ax.grid(True, linestyle='--', alpha=0.3)
    st.pyplot(fig)

def radar_chart(df, jugador, jornada=None, color="#d32f2f"):
    columnas = [
        "Distancia", "Distancia_Explosiva", "MAX_Acc", "MAX_Dec",
        "Max_HR", "Sprints_Abs_Cnt", "HSR_Abs_Cnt", "Max_Speed"
    ]
    
    if jornada:
        df = df[df["Jornada"] == jornada]
    
    valores = df[columnas].mean().values if len(df) > 0 else [0]*len(columnas)
    valores = np.nan_to_num(valores)

    angles = np.linspace(0, 2 * np.pi, len(columnas), endpoint=False).tolist()
    valores = np.concatenate((valores, [valores[0]]))
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, valores, color=color, linewidth=2)
    ax.fill(angles, valores, color=color, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(columnas, fontsize=8)
    ax.set_title(f"Radar de métricas físicas: {jugador}", fontsize=11)
    st.pyplot(fig)

def scatter_plot(df, x_col="MAX_Acc", y_col="MAX_Dec", color="#d32f2f"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df[x_col], df[y_col], c=color, alpha=0.6, edgecolors="k")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} vs {x_col}")
    ax.grid(True, linestyle='--', alpha=0.3)
    st.pyplot(fig)
