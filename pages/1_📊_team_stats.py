import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
import sys
import matplotlib.pyplot as plt
import base64
from pathlib import Path 

from utils.stats_db import get_player_stats, get_metric_groups, get_ranking_by_metric
from utils.styles import load_all_styles

from utils.auth import check_auth, logout

from utils.export_pdf import export_to_pdf, dataframe_a_pdf_contenido

# Configuración de la página
st.set_page_config(
    page_title="Atlético de Madrid 24/25",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cargar estilos al principio del archivo
load_all_styles()

if not check_auth():
    st.warning("No estás autenticado. Por favor, inicia sesión.")
    st.stop()

# Agregar la ruta raíz al path para importar módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mostrar la sidebar explícitamente
st.markdown("""
    <style>
    [data-testid="stSidebar"][aria-expanded="false"],
    [data-testid="stSidebar"][aria-expanded="true"],
    div[data-testid="collapsedControl"] {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        height: auto !important;
        position: relative !important;
        z-index: 1 !important;
        margin: 0px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Añadir reducción de márgenes y espaciados
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 0rem;
}
div[data-testid="stVerticalBlock"] > div {
    margin-bottom: 0.3rem;
}
</style>
""", unsafe_allow_html=True)

def app():
    # Título y logo
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<h2 class="main-title">Métricas temporada</h2>', unsafe_allow_html=True)
        st.markdown("<h3>Rankings</h3>", unsafe_allow_html=True)
    
    with col2:
        # Cargar el logo del Atlético de Madrid
        try:
            logo = Image.open("assets/images/logos/atm.png")
            st.image(logo, width=100)
        except FileNotFoundError:
            st.warning("Logo no encontrado. Verifica la ruta: assets/images/logos/atm.png")    

    metric_groups = get_metric_groups()
    all_metrics = sorted(set(m for group in metric_groups.values() for m in group))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        rank1 = st.selectbox(" ", all_metrics, index=all_metrics.index("Centralidad toques") if "Centralidad toques" in all_metrics else 0, key="rank1", label_visibility="collapsed")
    
    with col2:
        rank2 = st.selectbox(" ", all_metrics, index=all_metrics.index("%Pases completados") if "%Pases completados" in all_metrics else 0, key="rank2", label_visibility="collapsed")
    
    with col3:
        rank3 = st.selectbox(" ", all_metrics, index=all_metrics.index("TKL+INT/600 toques rival") if "TKL+INT/600 toques rival" in all_metrics else 0, key="rank3", label_visibility="collapsed")
    
    with col4:
        rank4 = st.selectbox(" ", all_metrics, index=all_metrics.index("Goles") if "Goles" in all_metrics else 0, key="rank4", label_visibility="collapsed")

    df = get_player_stats()
    if not df.empty:
        col1, col2, col3, col4 = st.columns(4)
        min_minutes_ranking = 200  # Mínimo de minutos para aparecer en rankings
        
        # Mostrar cada ranking
        for i, (col, rank) in enumerate(zip([col1, col2, col3, col4], [rank1, rank2, rank3, rank4])):
            with col:
                st.markdown(f"<h5>Top {rank}</h5>", unsafe_allow_html=True)
                ranking_df = get_ranking_by_metric(rank, limit=5, min_minutes=min_minutes_ranking)
                
                # Mostrar ranking
                if not ranking_df.empty:
                    for j, (_, player) in enumerate(ranking_df.iterrows()):
                        value = player[rank]
                        # Formatear el valor según el tipo
                        if isinstance(value, float):
                            value_str = f"{value:.2f}"
                        else:
                            value_str = str(value)
                        
                        st.markdown(
                            f"<div style='display: flex; justify-content: space-between;'>"
                            f"<span><strong>{j+1}.</strong> {player['Jugador']}</span>"
                            f"<span>{value_str}</span></div>",
                            unsafe_allow_html=True
                        )
                else:
                    st.write("No hay datos disponibles")

    # NUEVO BLOQUE: tabla + filtros
    # Dos columnas iguales: métricas y minutos
    st.write("")

    col_metrics, col_minutes = st.columns([1, 1])

    with col_metrics:
        st.markdown("<h4>Grupo métricas:</h4>", unsafe_allow_html=True)
        selected_group = st.selectbox(
            " ", list(metric_groups.keys()),
            index=5, key="metric_group", label_visibility="collapsed"
        )

    with col_minutes:
        st.markdown("<h4>Rango minutos:</h4>", unsafe_allow_html=True)
        min_minutes, max_minutes = st.slider(
            " ", 0, 2700, (0, 2700),
            key="minutes_range", label_visibility="collapsed"
        )

    # Obtener datos según filtros
    df = get_player_stats(min_minutes=min_minutes, max_minutes=max_minutes)

    # 🧾 Mostrar la tabla completa debajo
    if not df.empty:
        base_columns = ["Jugador", "Posicion", "Edad", "Minutos"]
        selected_metrics = metric_groups[selected_group]
        display_columns = [col for col in base_columns + selected_metrics if col in df.columns]

        # Solo aplicar formato a columnas numéricas (métricas seleccionadas)
        format_cols = [col for col in selected_metrics if col in df.columns]

        # Eliminar valores None o NaN de las columnas numéricas
        for col in format_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Aplicar formato decimal solo a columnas numéricas
        styled_df = df[display_columns].style.format({col: "{:.2f}" for col in format_cols})

        # Mostrar tabla
        st.dataframe(
            styled_df.background_gradient(
                cmap="RdYlGn",
                subset=format_cols
            ),
            use_container_width=True,
            height=450
        )
        
    else:
        st.warning("No hay jugadores que cumplan con el criterio de minutos.")

    # --- BARRA INFERIOR CON BACK, NOMBRE, PDF, EXIT ---
    st.markdown("""<hr style='margin-top: 2rem; margin-bottom: 1rem;'>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([3, 2, 1])

    with col1:
        st.markdown("""
            <div style="text-align: center; font-size: 18px; font-weight: bold; color: #232E61;">
                Ramón González<br>Mod11 MPAD
            </div>
        """, unsafe_allow_html=True)

    with col2:
        generar_pdf = st.button("📄 Generar PDF", key="pdf_button", use_container_width=True)

        if generar_pdf:
            contenido_pdf = dataframe_a_pdf_contenido(df, display_columns)
            pdf_filename = "informe_temporada.pdf"

            export_to_pdf("Informe de LaLiga 24/25", contenido_pdf, autor="Ramón González", output_path=pdf_filename)

            if os.path.exists(pdf_filename):
                with open(pdf_filename, "rb") as f:
                    pdf_bytes = f.read()

                st.success("✅ Informe PDF generado correctamente.")
                st.download_button(
                    label="⬇️ Descargar PDF",
                    data=pdf_bytes,
                    file_name=pdf_filename,
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.error("❌ No se pudo generar el PDF.")

    with col3:
        if st.button("⏻ Exit", use_container_width=True):
            logout()

if __name__ == "__main__":
    app()