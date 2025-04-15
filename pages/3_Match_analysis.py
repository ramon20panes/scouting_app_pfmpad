import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import traceback
import os
import sys
from utils.auth import check_auth, logout
from utils.styles import load_all_styles
from utils.visualization_3 import plot_team_metrics, fotmob_match_momentum_plot_atletico, pass_network_visualization
from utils.visualization_3 import preprocess_xg_data, plot_xg_timeline, plot_shot_map
from data.data_jornada.csv_lectura import load_partidos_master, load_match_stats, process_whoscored_event_data, get_passes_df, get_passes_between_df 
from data.data_jornada.url_mapeo import load_equipos_master
from utils.export_pdf import export_to_pdf, dataframe_a_pdf_contenido

# Configuración de la página
st.set_page_config(
    page_title="Match_Atlético de Madrid 24/25",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos al principio del archivo
load_all_styles()

# Autenticación
if not check_auth():
    st.warning("No estás autenticado. Por favor, inicia sesión.")
    st.stop()

# Ruta raíz al path para importar módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# CSS personalizado para mejorar la visibilidad en los selectbox y otros widgets
st.markdown("""
<style>
    /* Mejorar visibilidad en selectbox y inputs */
    .stSelectbox div[data-baseweb="select"] span {
        color: black !important;
        background-color: white !important;
    }
    
    .stSelectbox div[data-baseweb="select"] div {
        background-color: white !important;
    }
    
    .stSelectbox div[data-baseweb="select"] input {
        color: black !important;
    }
    
    .stSlider div[data-baseweb="slider"] div {
        background-color: white !important;
    }
    
    .stMultiSelect div[data-baseweb="select"] span {
        color: black !important;
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
        st.markdown('<h2 class="main-title">Análisis Post Partido</h2>', unsafe_allow_html=True)
        st.markdown("<h4>Elegir jornada</h4>", unsafe_allow_html=True)

    with col2:
        # Cargar el logo del Atlético de Madrid
        ESCUDO_PATH = Path("assets/images/logos/atm.png")
        try:
            st.image(ESCUDO_PATH, width=100)
        except FileNotFoundError:
            st.warning("Logo no encontrado. Verifica la ruta del escudo.")
        
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

    # Lógica principal
    try:
        with st.spinner("Cargando partidos..."):
            partidos_df = load_partidos_master()

        with st.spinner("Cargando equipos..."):
            equipos_df = load_equipos_master()

        if partidos_df.empty or equipos_df.empty:
            st.error("No se pudieron cargar los datos maestros")
            return
        
        st.session_state.partidos_df = partidos_df
        st.session_state.equipos_df = equipos_df

        # Selector de jornada con persistencia
        jornadas = partidos_df['formato_jornada'].tolist()
        if 'selected_jornada' not in st.session_state:
            st.session_state.selected_jornada = jornadas[0]  # Primera jornada por defecto

        selected_jornada = st.selectbox("Selecciona una jornada:", jornadas)

        # Obtener datos del partido
        partido_row = partidos_df[partidos_df['formato_jornada'] == selected_jornada]
        if partido_row.empty:
            st.error(f"No se encontró información para la jornada: {selected_jornada}")
            return
            
        partido_data = partido_row.iloc[0]
        equipo_local = partido_data['equipo_local']
        equipo_visitante = partido_data['equipo_visitante']
        
        st.write(f"Partido: {equipo_local} vs {equipo_visitante}")
        
        local_row = equipos_df[equipos_df['nombre'].str.strip() == equipo_local.strip()]
        visitante_row = equipos_df[equipos_df['nombre'].str.strip() == equipo_visitante.strip()]
        
        local_info = local_row.iloc[0].to_dict() if not local_row.empty else None
        visitante_info = visitante_row.iloc[0].to_dict() if not visitante_row.empty else None
                
        # Cargar estadísticas del partido
        partido_str = f"{equipo_local}-{equipo_visitante}"
        match_stats = load_match_stats(jornada=partido_data['Jornada'], partido=partido_str)

        if match_stats is not None and not match_stats.empty:
            if local_info and visitante_info:
                plot_team_metrics(match_stats, local_info, visitante_info)
            else:
                st.warning("No se pudo mostrar la tabla de estadísticas porque falta información de los equipos")
        else:
            st.warning(f"No se encontraron estadísticas para el partido: {partido_str}")
        
        matches_df = None  # ← Para que no dé error si aún no hemos entrado en la pestaña
        display_columns = []

        # SECCIÓN DE VISUALIZACIONES CON TABS
        st.header("Visualizaciones avanzadas")
            
        # Opciones de visualización como pestañas
            
        tab1, tab2, tab3, tab4 = st.tabs([
            "Dinámica del partido",
            "Redes de pases",                         
            "Expected Goals (xG)", 
            "Mapas de tiros"
        ])
            
        # Pestañas para seleccionar visualización

        with tab1:
            st.subheader("Match Momentum")

            # Verificar si estamos en una jornada problemática
            is_problematic_jornada = False  # Con esto, no tenemos jornadas problemáticas

            try:
                # Verificar si tenemos ID de FotMob
                if 'id_fotmob' in partido_data and not pd.isna(partido_data['id_fotmob']):
                    fotmob_id = str(partido_data['id_fotmob'])
                
                    with st.spinner("Cargando datos de momentum..."):
                        fig_mm, ax_mm = fotmob_match_momentum_plot_atletico(fotmob_id, debug=False)
                        st.pyplot(fig_mm)
                else:
                    st.warning(f"No hay ID de FotMob disponible para el partido: {partido_data['equipo_local']} vs {partido_data['equipo_visitante']}")
        
            except Exception as e:
                st.error(f"Error al generar el gráfico de momentum: {str(e)}")
                st.info("Es posible que este partido no tenga datos de momentum disponibles en FotMob.")

        # Mapa de redes de pase
        with tab2:
            st.subheader("Redes de pases")
            try:
                # Rutas de archivos
                jornada_formato = selected_jornada.replace(' ', '_')
                events_file = f"data/raw/parquet/{jornada_formato}_EventData_whoscored.parquet"
                players_file = f"data/raw/parquet/{jornada_formato}_PlayerData_whoscored.parquet"
                teams_file = "data/raw/master/equipos_master_laliga.csv"

                # Spinner mientras los datos se cargan
                with st.spinner("Cargando datos de eventos y jugadores..."):
                    # Procesar los datos - usar la función existente
                    df_red, dfp_red, team_info = process_whoscored_event_data(events_file, players_file, teams_file)
            
                    # Preparar datos para visualización
                    passes_df = get_passes_df(df_red)
                
                    # Colores
                    atleti_color = '#272E61'  # Azul oscuro para el Atlético de Madrid
                    rival_color = '#e60000'   # Rojo para el equipo rival
                
                    # Figura
                    fig, axs = plt.subplots(1, 2, figsize=(20, 14), facecolor="#d4d4d4")
                
                    # Procesar y visualizar ambos equipos
                    for i, team_name in enumerate([team_info['home_team_name'], team_info['away_team_name']]):
                        # Determinar si este equipo es el Atlético de Madrid
                        is_this_team_atleti = (team_info['is_atleti_home'] and i == 0) or (not team_info['is_atleti_home'] and i == 1)
                        team_color = atleti_color if is_this_team_atleti else rival_color
                    
                        # Calcular datos para este equipo usando la función existente
                        passes_between_df, average_locs_df = get_passes_between_df(team_name, passes_df, None, df_red)
                    
                        # Dibujar red de pases
                        pass_network_visualization(
                            ax=axs[i],
                            passes_between_df=passes_between_df,
                            average_locs_and_count_df=average_locs_df,
                            teamName=team_name,
                            passes_df=passes_df,
                            home_team=team_info['home_team_name'],
                            away_team=team_info['away_team_name'],
                            team_color=team_color,
                            jornada=selected_jornada  # Añadir el parámetro jornada
                        )
                
                    plt.tight_layout()
                    st.pyplot(fig)
                
            except Exception as e:
                import traceback
                st.error(f"Error al generar red de pases: {str(e)}")
                st.write(traceback.format_exc())

        # XG visualización
        with tab3:
            st.subheader("Expected Goals (xG)")
        
            with st.spinner("Cargando datos de xG..."):
                try:
                    # Obtener URL del partido desde partido_data
                    url_partido = partido_data.get('url_fbref')
                
                    if url_partido:
                        try:
                            # Cargar datos de fbref
                            df_processed = pd.read_html(url_partido, attrs={'id': 'shots_all'})[0]
                            # Verificar si tennemos datos válidos
                            if df_processed.empty or df_processed.shape[0] < 2:
                                st.warning(f"No hay suficientes datos de tiros para la jornada {selected_jornada}")
                            # Preprocesar datos de xG
                            else:
                                df_xG = preprocess_xg_data(df_processed)

                                if df_xG.empty:
                                    st.warning("No se pudieron procesar los datos xG correctamente")
                                else:
                                    # Crear figura de xG
                                    fig = plot_xg_timeline(df_xG)
                    
                                    # Mostrar figura
                                    st.pyplot(fig)

                        except Exception as e:
                            import traceback
                            st.error(f"Error al procesar datos de xG: {str(e)}")
                            st.code(traceback.format_exc())
                            st.warning(f"Estructura de datos no compatible para la jorndad {selected_jornada}")

                    else:
                        st.info("No se encontró URL de partido para cargar datos de xG")
            
                except Exception as e:
                    import traceback 
                    st.error(f"Error al cargar datos de xG: {str(e)}")
                    st.code(traceback.format_exc())
            
        # Representación de tiros de ambos equipos            
        with tab4:
            st.subheader("Mapas de tiros")
        
            with st.spinner("Cargando datos de tiros..."):
                try:
                    # Verificar si tenemos ID de Understat
                    if 'id_understat' in partido_data and not pd.isna(partido_data['id_understat']):
                        # Convertir a entero y luego a string para eliminar el ".0"
                        understat_id = str(int(float(partido_data['id_understat'])))
                    
                        # Función cacheada para obtener datos de tiros
                        @st.cache_data(ttl=3600)
                        def get_cached_shots(id_understat):
                            from data.data_processing.understat_data import get_shot_map
                            return get_shot_map(id_understat)
                    
                        # Obtener datos
                        shots_data = get_cached_shots(understat_id)
                    
                        if shots_data:
                            col1, col2 = st.columns(2)
                        
                            with col1:
                                local_shots_fig = plot_shot_map(shots_data['local'], partido_data['equipo_local'])
                                st.pyplot(local_shots_fig)
                        
                            with col2:
                                visitante_shots_fig = plot_shot_map(shots_data['visitante'], partido_data['equipo_visitante'])
                                st.pyplot(visitante_shots_fig)                                      
                                
                        else:
                            st.warning(f"No se pudieron obtener datos de tiros para el partido: {partido_data['equipo_local']} vs {partido_data['equipo_visitante']}")
                            st.info("Es posible que este partido no tenga datos disponibles en Understat.")
                    else:
                        st.warning("No hay ID de Understat disponible para este partido")
            
                except Exception as e:
                    import traceback
                    st.error(f"Error al cargar mapas de tiros: {str(e)}")
                    st.code(traceback.format_exc())

    except Exception as e:
        import traceback
        st.error(f"Error general en la aplicación: {str(e)}")
        if 'traceback' in globals():
            st.code(traceback.format_exc())
        else:
            import traceback
            st.code(traceback.format_exc())

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
            columnas_deseadas = ["date", "homeTeam", "awayTeam", "score", "matchday"]
            columnas_disponibles = [col for col in columnas_deseadas if col in matches_df.columns]

            if not columnas_disponibles:
                st.error("No hay columnas disponibles para exportar el PDF.")
            else:
                contenido_pdf = dataframe_a_pdf_contenido(matches_df, columnas_disponibles)
                pdf_filename = "informe_visualizaciones.pdf"

                export_to_pdf("Visualizaciones LaLiga 24/25", contenido_pdf, autor="Ramón González", output_path=pdf_filename)

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

        else:
            st.warning("⚠️ Primero accede a la pestaña 'Timeline Partidos' para generar el informe PDF.")

    with col3:
        if st.button("⏻ Exit", use_container_width=True):
            logout()    

if __name__ == "__main__":
    app()