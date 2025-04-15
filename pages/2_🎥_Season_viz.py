import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import base64
from PIL import Image
from datetime import datetime

from utils.auth import check_auth, logout
from utils.styles import load_all_styles

# Visualizaciones
from utils.visualization_2 import create_bumpy_chart, create_match_timeline, plot_atletico_xg_differential
from data.api_handlers.football_data_api import load_teams_mapping, get_atletico_matches
from data.data_processing.understat_data import get_atletico_data
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

# Verificar autenticación
if not check_auth():
    st.warning("No estás autenticado. Por favor, inicia sesión.")
    st.stop()

# Ruta raíz al path para importar módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# CSS personalizado para mejorar la visibilidad en los selectbox y otros widgets
st.markdown("""
<style>
    /* Mejorar visibilidad en selectbox y inputs */
    .stSelectbox div[data-baseweb="select"] span,
    .stMultiSelect div[data-baseweb="select"] span {
        color: black !important;
        background-color: white !important;
    }
    
    .stSelectbox div[data-baseweb="select"] div,
    .stMultiSelect div[data-baseweb="select"] div {
        background-color: white !important;
    }
    
    .stSelectbox div[data-baseweb="select"] input,
    .stMultiSelect div[data-baseweb="select"] input {
        color: black !important;
    }
    
    .stSlider div[data-baseweb="slider"] div {
        background-color: white !important;
    }
    
    /* Para asegurar que los textos en los dropdowns sean visibles */
    div[role="listbox"] ul li {
        color: black !important;
    }
    
    /* Para los textos dentro de inputs numéricos */
    input[type="number"] {
        color: black !important;
    }
    
    /* Para rangos de selección */
    .stSlider [data-baseweb="slider"] {
        background-color: rgba(255, 255, 255, 0.8) !important;
    }
</style>
""", unsafe_allow_html=True)

# Variables globales para el PDF
matches_df = pd.DataFrame()
display_columns = []

# Sidebar
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

# Reducción de márgenes y espaciados
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
        st.markdown('<h2 class="main-title">Evolución temporada</h2>', unsafe_allow_html=True)
        st.markdown("<h4>Selección de visualización</h4>", unsafe_allow_html=True)
    
    with col2:
        # Cargar el logo del Atlético de Madrid
        try:
            logo = Image.open("assets/images/logos/atm.png")
            st.image(logo, width=100)
        except FileNotFoundError:
            try:
                # Intentar rutas alternativas
                alt_paths = [
                    "assets/escudos/atm.png",
                    "assets/logos/atm.png",
                    "assets/images/atm.png"
                ]
                for path in alt_paths:
                    if Path(path).exists():
                        logo = Image.open(path)
                        st.image(logo, width=100)
                        break
                else:
                    st.warning("Logo no encontrado. Verifica la ruta de los logos.")
            except Exception as e:
                st.warning(f"No se pudo cargar el logo: {str(e)}")
    
    # Reducir el espacio antes de los tabs
    st.markdown("""
    <style>
    div[data-testid="stHorizontalBlock"] {
        margin-bottom: -25px;
    }
    div[data-testid="stTabs"] {
        margin-top: -20px;
    }
    div[data-testid="stTabContent"] {
        padding-top: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Tabs para las diferentes visualizaciones
    tab1, tab2, tab3 = st.tabs(["Clasificación LaLiga", "Timeline Partidos", "Expected Goals (xG)"])
    
    # TAB 1 - Clasificación LaLiga
    with tab1:
        # Cargar los datos
        @st.cache_data(ttl=3600)
        def load_liga_positions():
            csv_path = Path("data/raw/liga_positions_24_25_pag2.csv")
            if csv_path.exists():
                return pd.read_csv(csv_path)
            else:
                # Rutas alternativas
                alt_path = Path("data/raw/master/liga_positions_24_25.csv")
                if alt_path.exists():
                    return pd.read_csv(alt_path)
        
        try:
            df_cla = load_liga_positions()
        
            # Valores por defecto si es la primera vez
            if "highlight_teams" not in st.session_state:
                st.session_state.highlight_teams = ["Club Atlético de Madrid", "Real Madrid CF", "FC Barcelona"]
    
            # Vrear el selector
            highlight_teams = st.multiselect("Equipos seleccionados: ",
                df_cla["Equipo"].tolist(),
                default=st.session_state.highlight_teams,
                key="teams_multiselect"
            )

            # Guarda la selección actual en session_state
            st.session_state.highlight_teams = highlight_teams
        
            # Gráfico
            fig, ax = create_bumpy_chart(df_cla, highlight_teams)
            st.pyplot(fig)
        
        except Exception as e:
            st.error(f"Error al cargar o procesar los datos: {str(e)}")

    with tab2:
        st.subheader("Timeline de Partidos")

        @st.cache_data(ttl=3600)
        def load_atletico_matches():    
            try:
                api_key = st.secrets["FOOTBALL_DATA_API_KEY"]
            except:
                from dotenv import load_dotenv
                load_dotenv()
                api_key = os.getenv("FOOTBALL_DATA_API_KEY")
                if not api_key:
                    try:
                        import toml
                        config = toml.load("streamlit/secrets.toml")
                        api_key = config.get("football_data_api_key", "")
                    except:
                        pass

            if not api_key:
                st.error("No se encontró la API Key para football-data.org. Verifica tus secrets o .env")
                return None

            return get_atletico_matches(api_key)

        try:
            matches_df = load_atletico_matches()

            if matches_df is not None and not matches_df.empty:
                display_columns = ["date", "homeTeam", "awayTeam", "score", "matchday"]
                team_mapping = load_teams_mapping()

                fig = create_match_timeline(matches_df, team_mapping)
                st.pyplot(fig)

                with st.expander("Ver datos detallados"):
                    st.dataframe(matches_df)

                # 💾 Guardamos en session_state para uso posterior en el botón final
                st.session_state.matches_df_export = matches_df
                st.session_state.columns_df_export = [col for col in display_columns if col in matches_df.columns]

            else:
                st.warning("No se pudieron cargar los datos de partidos. Verifica tu conexión a la API.")

        except Exception as e:
            st.error(f"Error al cargar o procesar los datos: {str(e)}")
            import traceback
            st.error(traceback.format_exc())    

    with tab3:
        st.subheader("Análisis xG por Jornada")
        
        with st.spinner("Cargando datos de xG desde Understat..."):
            df_expcGL, df1 = get_atletico_data()  # Esta función ya maneja internamente los casos de error
        
        # Gráfico
        fig = plot_atletico_xg_differential(df_expcGL, df1)
        st.pyplot(fig)
        
        # Datos en una tabla expandible
        with st.expander("Ver datos en tabla"):
            st.dataframe(df_expcGL.style.format({
                'xG': '{:.2f}', 
                'xGA': '{:.2f}',
                'xGdif': '{:.2f}',
                'npxG': '{:.2f}',
                'npxGA': '{:.2f}',
                'xpts': '{:.2f}',
                'npxGD': '{:.2f}'
            }))

    # Obtener el nombre de la página actual
    current_page = __file__.split('\\')[-1]

    # Inicializar y actualizar el historial
    if "page_history" not in st.session_state:
        st.session_state.page_history = []

    # Actualizar historial solo si es una página nueva
    if not st.session_state.page_history or st.session_state.page_history[-1] != current_page:
        st.session_state.page_history.append(current_page)

    # Firma, PDF, Back y Exit
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
            df_export = st.session_state.get("matches_df_export", pd.DataFrame())
            columnas = st.session_state.get("columns_df_export", [])

            if df_export.empty or not columnas:
                st.warning("No hay datos disponibles para generar el PDF.")
            else:
                contenido_pdf = dataframe_a_pdf_contenido(df_export, columnas)
                pdf_filename = "informe_progreso_liga.pdf"

                export_to_pdf("Progresión LaLiga 24/25", contenido_pdf, autor="Ramón González", output_path=pdf_filename)

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