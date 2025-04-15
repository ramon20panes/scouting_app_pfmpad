import streamlit as st
import pandas as pd
import os
import sys
from pathlib import Path
from PIL import Image
import plotly.express as px
from datetime import datetime

from utils.auth import check_auth, logout
from utils.styles import load_all_styles
from utils.datos_fisicos import (
    load_players_data, load_physical_data, calculate_trend, display_metric_with_trend, normalize_name,
    create_evolution_chart, create_comparison_chart, calculate_age, get_player_physical_data, map_name_to_database
)
from utils.export_pdf import export_to_pdf, dataframe_a_pdf_contenido
from utils.auth import check_player_access, get_user_name
from utils.auth import get_user_role, get_user_name

# Configuración de la página
st.set_page_config(
    page_title="Atlético de Madrid 24/25",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CaEstilos
load_all_styles()

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
    
    /* Estilos para tarjeta de jugador */
    .player-card {
        background-color: #fff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    /* Estilos para métricas con indicador de progreso */
    .metric-up {
        color: #4CAF50;
        font-weight: bold;
    }
    
    .metric-down {
        color: #F44336;
        font-weight: bold;
    }
    
    .metric-stable {
        color: #FFC107;
        font-weight: bold;
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
        st.markdown('<h2 class="main-title">Datos Físicos Individuales</h2>', unsafe_allow_html=True)
        st.markdown("<h4>Análisis personalizado por jugador</h4>", unsafe_allow_html=True)
    
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
    
    # Cargar datos de jugadores
    df_players = load_players_data()
    
    # Cargar datos físicos
    df_physical = load_physical_data()
    
    # Corrección específica para jugadores problemáticos
    if check_player_access():
        player_name = get_user_name()
        
        correcciones = {
            "Jose María Giménez": "José María Giménez",
            "Alexander Sorloth": "Alexander Sørloth",
            "Aleksander Sorloth": "Alexander Sørloth",
            "Clement Lenglet": "Clément Lenglet",
            "Jorge Resurrección": "Koke"
        }

        # Después de filtrar
        df_physical_filtered = df_physical[df_physical['nombre'] == player_name]
        
        # Primero intentar con el diccionario de correcciones
        if player_name in correcciones:
            corrected_name = correcciones[player_name]
            st.sidebar.info(f"Corrigiendo nombre: {player_name} → {corrected_name}")
            player_name = corrected_name
        # Si no hay corrección directa y tampoco hay resultados, intentar búsqueda flexible
        elif len(df_physical_filtered) == 0:
            # Normalizar el nombre del jugador (quitar acentos, minúsculas)
            player_name_norm = normalize_name(player_name)
            
            # Buscar la mejor coincidencia
            best_match = None
            best_score = 0
            
            for nombre_db in df_physical['nombre'].unique():
                nombre_db_norm = normalize_name(nombre_db)
                
                # Calcular similitud entre nombres
                if player_name_norm in nombre_db_norm or nombre_db_norm in player_name_norm:
                    # Calcular un puntaje simple basado en la longitud de la coincidencia
                    score = len(set(player_name_norm.split()) & set(nombre_db_norm.split()))
                    
                    if score > best_score:
                        best_score = score
                        best_match = nombre_db
            
            # Si encontramos una coincidencia, usarla
            if best_match:
                st.sidebar.info(f"Coincidencia aproximada encontrada: {player_name} → {best_match}")
                player_name = best_match
        
        # Ahora filtrar con el nombre corregido
        df_players = df_players[df_players['nombre_completo'] == player_name]
        df_physical = df_physical[df_physical['nombre'] == player_name]
        
        if df_physical.empty:
            st.error(f"❌ No se encontraron datos físicos para {player_name}.")
            st.stop()

    # Definir las métricas físicas disponibles una sola vez
    available_metrics = [
        'distancia_total', 'distancia_sprint', 'velocidad_max', 'aceleraciones', 'desaceleraciones', 'sprints', 
        'distancia_alta_intensidad', 'carga_cardio', 'impactos', 'distancia_por_min', 'distancia', 'distancia_explosiva',
        'hibd', 'max_acc', 'max_dec', 'avg_acc', 'avg_dec', 'high_acc_cnt', 'high_dec_cnt', 'high_acc_m', 'high_dec_m',
        'max_hr', 'avg_hr', 'high_hr_m', 'high_hr_cnt', 'sprints_abs_cnt', 'sprints_rel_cnt', 'hsr_abs_cnt', 'hsr_rel_cnt',
        'sprints_rel_m', 'sprints_abs_m', 'hsr_abs_m', 'hsr_rel_m', 'max_speed', 'avg_speed', 'total_impacts', 'steps_count',
        'step_balance', 'jumps_count', 'hmld', 'hml_cnt'
    ]

    metric_display_names = {
        'distancia_total': 'Distancia Total',
        'distancia_sprint': 'Distancia Sprint',
        'velocidad_max': 'Velocidad Máxima',
        'aceleraciones': 'Aceleraciones',
        'desaceleraciones': 'Desaceleraciones',
        'sprints': 'Sprints',
        'distancia_alta_intensidad': 'Distancia Alta Intensidad',
        'carga_cardio': 'Carga Cardio',
        'impactos': 'Impactos',
        'distancia_por_min': 'Distancia por Minuto',
        'distancia': 'Distancia',
        'distancia_explosiva': 'Distancia Explosiva',
        'hibd': 'HIBD',
        'max_acc': 'Máxima Aceleración',
        'max_dec': 'Máxima Desaceleración',
        'avg_acc': 'Aceleración Promedio',
        'avg_dec': 'Desaceleración Promedio',
        'high_acc_cnt': 'Cantidad Altas Aceleraciones',
        'high_dec_cnt': 'Cantidad Altas Desaceleraciones',
        'high_acc_m': 'Metros Alta Aceleración',
        'high_dec_m': 'Metros Alta Desaceleración',
        'max_hr': 'FC Máxima',
        'avg_hr': 'FC Promedio',
        'high_hr_m': 'Metros Alta FC',
        'high_hr_cnt': 'Cantidad Alta FC',
        'sprints_abs_cnt': 'Cantidad Sprints Absolutos',
        'sprints_rel_cnt': 'Cantidad Sprints Relativos',
        'hsr_abs_cnt': 'Cantidad HSR Absoluto',
        'hsr_rel_cnt': 'Cantidad HSR Relativo',
        'sprints_rel_m': 'Metros Sprint Relativo',
        'sprints_abs_m': 'Metros Sprint Absoluto',
        'hsr_abs_m': 'Metros HSR Absoluto',
        'hsr_rel_m': 'Metros HSR Relativo',
        'max_speed': 'Velocidad Máxima',
        'avg_speed': 'Velocidad Promedio',
        'total_impacts': 'Impactos Totales',
        'steps_count': 'Cantidad de Pasos',
        'step_balance': 'Balance de Pasos',
        'jumps_count': 'Cantidad de Saltos',
        'hmld': 'HMLD',
        'hml_cnt': 'Cantidad HML'
    }
    
    if df_players.empty:
        st.error("No se pudieron cargar los datos de jugadores. Verifica el archivo CSV.")
        st.stop()
    
    if df_physical.empty:
        st.error("No se pudieron cargar los datos físicos. Verifica la base de datos.")
        st.stop()

    # Subtítulo
    st.subheader("Selecciona un jugador")    

    # Crear lista de jugadores para el selectbox
    player_options = df_players['nombre_completo'].tolist()

    # Obtener el índice de Julián Álvarez si existe
    julian_index = next((i for i, name in enumerate(player_options) if "Julián Álvarez" in name), 0)

    if check_player_access():
        # Si es jugador, no puede elegir, se le muestra directamente su nombre
        selected_player = get_user_name()
        st.markdown(f"👤 Estás viendo tus propios datos: **{selected_player}**")
    else:
        # Admins o profes pueden seleccionar cualquier jugador
        selected_player = st.selectbox(
            "Jugador:",
            player_options,
            index=player_options.index("Julián Álvarez") if "Julián Álvarez" in player_options else 0
        )    
    
    # Filtrar datos del jugador seleccionado
    player_info = df_players[df_players['nombre_completo'] == selected_player].iloc[0]
    
    # Obtener datos físicos del jugador seleccionado
    player_physical = df_physical[df_physical['nombre'] == selected_player]

    # Mapeamos directo por posibles conflictos futuros con nombres:
    if selected_player == "Jorge Resurreccion" or selected_player == "Jorge Resurrección":
        player_physical = df_physical[df_physical['nombre'] == "Koke"]
    elif selected_player == "Jose Maria Gimenez":
        player_physical = df_physical[df_physical['nombre'] == "José María Giménez"]
    elif selected_player == "Alexander Sorloth":
        player_physical = df_physical[df_physical['nombre'] == "Alexander Sørloth"]
    elif selected_player == "Clement Lenglet":
        player_physical = df_physical[df_physical['nombre'] == "Clément Lenglet"]
    else:
        # Intenta encontrar el jugador directamente
        player_physical = df_physical[df_physical['nombre'] == selected_player]
        
    # Si no encuentra nada, intenta usando la función de mapeo
    if player_physical.empty:
        db_name = map_name_to_database(selected_player)
        player_physical = df_physical[df_physical['nombre'] == db_name]
    
    # Si no hay datos físicos, mostrar mensaje
    if player_physical.empty:
        st.warning(f"No hay datos físicos disponibles para {selected_player}")
    
    # Crear tabs para las diferentes visualizaciones
    tab1, tab2, tab3 = st.tabs(["Perfil Jugador", "Análisis Físico", "Evolución"])
    
    # TAB 1 - Perfil Jugador
    with tab1:
        # Crear layout para la información del jugador
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Crear subcolumnas para centrar el contenido
            col1_a, col1_b, col1_c = st.columns([1, 3, 1])
            
            with col1_b:
                
                try:
                    # Intentar cargar la imagen desde la ruta en el CSV
                    if pd.notna(player_info['ruta_foto']):
                        player_img_path = Path(player_info['ruta_foto'])
                        if player_img_path.exists():
                            player_img = Image.open(player_img_path)
                            st.image(player_img, width=200)
                        else:
                            # Intentar con el número de dorsal
                            dorsal_path = Path(f"assets/images/players/{int(player_info['dorsal'])}.png")
                            if dorsal_path.exists():
                                player_img = Image.open(dorsal_path)
                                st.image(player_img, width=200)
                            else:
                                st.warning("No se encontró la imagen del jugador")
                    else:
                        # Intentar con el número de dorsal
                        dorsal_path = Path(f"assets/images/players/{int(player_info['dorsal'])}.png")
                        if dorsal_path.exists():
                            player_img = Image.open(dorsal_path)
                            st.image(player_img, width=200)
                        else:
                            st.warning("No se encontró la imagen del jugador")
                except Exception as e:
                    st.warning(f"No se pudo cargar la imagen del jugador: {e}")
            
            # Calcular la edad para incluirla en la tarjeta
            edad_texto = "N/A"
            if pd.notna(player_info['fecha_nacimiento']):
                try:
                    # Intentar con formato DD/MM/YYYY
                    birth_date = datetime.strptime(str(player_info['fecha_nacimiento']), '%d/%m/%Y')
                    today = datetime.now()
                    age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
                    edad_texto = f"{age} años"
                except:
                    try:
                        # Si falla, intentar con formato YYYY-MM-DD
                        birth_date = datetime.strptime(str(player_info['fecha_nacimiento']), '%Y-%m-%d')
                        today = datetime.now()
                        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
                        edad_texto = f"{age} años"
                    except:
                        edad_texto = "N/A"
            
            # Generar toda la tarjeta en un solo bloque HTML
            st.markdown(f"""
            <div class="player-card" style="text-align: center; padding: 15px; margin-top: 10px; border-radius: 10px; background-color: white; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                <h3>{player_info['nombre_completo']}</h3>
                <p><strong>Dorsal:</strong> {int(player_info['dorsal']) if pd.notna(player_info['dorsal']) else 'N/A'}</p>
                <p><strong>Posición:</strong> {player_info['posicion']}</p>
                <p><strong>Edad:</strong> {edad_texto}</p>
                <p><strong>Nacionalidad:</strong> {player_info['pais']}</p>
            </div>
            """, unsafe_allow_html=True)              
        
        with col2:
            # Mostrar heatmap desde una imagen guardada
            st.subheader("Heatmap de Posiciones")
            
            # Obtener el dorsal para buscar el heatmap
            dorsal = int(player_info['dorsal']) if pd.notna(player_info['dorsal']) else 0
            
            # Buscar imagen de heatmap guardada usando el dorsal
            heatmap_path = Path(f"assets/images/heatmaps/{dorsal}heatmap.png")
            
            if heatmap_path.exists():
                # Mostrar imagen guardada - usando use_container_width en lugar de use_column_width
                heatmap_img = Image.open(heatmap_path)
                
                # Reducir el tamaño del heatmap 
                st.image(heatmap_img, use_container_width=True)
            else:
                # Mostrar mensaje de aviso
                st.warning(f"No se encontró el heatmap para este jugador.")

        # Sección de estadísticas generales
        st.subheader("Estadísticas Generales")
                
        if not player_physical.empty:
            # Calculamos algunas estadísticas generales
            avg_distance = player_physical['distancia_total'].mean()
            max_speed = player_physical['velocidad_max'].max()
            avg_sprints = player_physical['sprints'].mean()
                    
            # Mostrar en 3 columnas
            col1, col2, col3 = st.columns(3)
                    
            with col1:
                st.metric("Distancia Prom. (km)", f"{avg_distance:.2f}")
                    
            with col2:
                st.metric("Velocidad Máx. (km/h)", f"{max_speed:.2f}")
                    
            with col3:
                st.metric("Sprints Prom.", f"{avg_sprints:.1f}")
        else:
            st.warning("No hay estadísticas disponibles para este jugador")       
    
    # TAB 2 - Análisis Físico
    with tab2:
        if not player_physical.empty:
            # Seleccionar métricas para visualizar
            st.subheader("Análisis Comparativo")

            # Define métricas predeterminadas que funcionan bien juntas visualmente
            default_metrics2 = ['distancia_total', 'distancia_alta_intensidad', 'distancia_por_min']       
                        
            # Selector de métricas (máximo 3)
            selected_metrics = st.multiselect(
                "Selecciona hasta 3 métricas para comparar:",
                available_metrics,
                default=default_metrics2,
                max_selections=5,
                format_func=lambda x: metric_display_names.get(x, x.replace('_', ' ').title())
            )
            
            if selected_metrics:
                # Selector de jornada para comparar
                jornadas = sorted(player_physical['jornada'].unique())
                selected_jornada = st.selectbox(
                    "Jornada para comparación:",
                    jornadas,
                    index=len(jornadas)-1 if jornadas else 0
                )
                
                # Crear gráfico comparativo
                fig_comp = create_comparison_chart(player_physical, selected_metrics, selected_jornada)
                
                if fig_comp:
                    st.plotly_chart(fig_comp, use_container_width=True)
                else:
                    st.warning("No se pueden crear visualizaciones con los datos seleccionados")
                
                # Mostrar tendencias en métricas seleccionadas
                st.subheader("Tendencias")
                
                # Crear columnas para mostrar métricas con tendencias
                cols = st.columns(len(selected_metrics))
                
                for i, metric in enumerate(selected_metrics):
                    with cols[i]:
                        # Calcular tendencia
                        trend, change_pct = calculate_trend(player_physical, metric)
                        
                        # Obtener último valor
                        last_value = player_physical.sort_values('jornada')[metric].iloc[-1]
                        
                        # Mostrar con formato
                        metric_name = metric.replace('_', ' ').title()
                        display_metric_with_trend(metric_name, f"{last_value:.2f}", trend, change_pct)
            else:
                st.warning("Selecciona al menos una métrica para visualizar")
        else:
            st.warning("No hay datos físicos disponibles para este jugador")
    
    # TAB 3 - Evolución
    with tab3:
        if not player_physical.empty:
            st.subheader("Evolución a lo largo de la temporada")
            
            # Selector de métricas para la evolución
            selected_metrics_evol = st.multiselect(
                "Selecciona métricas para visualizar evolución:",
                available_metrics,
                default=['distancia_total', 'distancia_alta_intensidad', 'distancia_por_min'],
                max_selections=5,
                format_func=lambda x: metric_display_names.get(x, x.replace('_', ' ').title())
            )

            if selected_metrics_evol:
                # Crear gráfico de evolución
                fig_evol = create_evolution_chart(player_physical, selected_metrics_evol)
                
                if fig_evol:
                    st.plotly_chart(fig_evol, use_container_width=True)
                else:
                    st.warning("No se pueden crear visualizaciones de evolución con los datos seleccionados")
                
                # Tabla de datos completa
                with st.expander("Ver todos los datos"):
                    # Seleccionar columnas relevantes
                    cols_to_show = ['jornada', 'fecha'] + selected_metrics_evol
                    
                    # Mostrar tabla con formato
                    st.dataframe(
                        player_physical[cols_to_show].sort_values('jornada', ascending=False)
                    )
            else:
                st.warning("Selecciona al menos una métrica para visualizar evolución")
        else:
            st.warning("No hay datos físicos disponibles para este jugador")

    # Obtener el nombre de la página actual
    current_page = __file__.split('\\')[-1]

    # Inicializar y actualizar el historial
    if "page_history" not in st.session_state:
        st.session_state.page_history = []

    # Actualizar historial solo si es una página nueva
    if not st.session_state.page_history or st.session_state.page_history[-1] != current_page:
        st.session_state.page_history.append(current_page)

    # Crear contenido para PDF con las columnas físicas más importantes
    display_cols = ['jornada', 'fecha', 'distancia_total', 'distancia_alta_intensidad', 'distancia_por_min']
    df_export = player_physical[display_cols] if not player_physical.empty else pd.DataFrame()

    # Define el nombre del archivo PDF 
    pdf_filename = f"informe_fisico_{selected_player.replace(' ', '_')}.pdf"

    # --- BARRA INFERIOR CON BACK, NOMBRE, PDF, EXIT ---
    st.markdown("""<hr style='margin-top: 2rem; margin-bottom: 1rem;'>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([3, 2, 1])

    with col1:
        st.markdown(f"""
            <div style="text-align: center; font-size: 18px; font-weight: bold; color: #232E61;">
                Ramón González<br>Mod11 MPAD
            </div>
        """, unsafe_allow_html=True)

    with col2:
        generar_pdf = st.button("📄 Generar PDF", key="pdf_button_fisico", use_container_width=True)

        if generar_pdf:
            columnas_pdf = ['jornada', 'fecha', 'distancia_total', 'distancia_alta_intensidad', 'distancia_por_min']
            columnas_disponibles = [col for col in columnas_pdf if col in player_physical.columns]
            
            if columnas_disponibles:
                contenido_pdf = dataframe_a_pdf_contenido(player_physical[columnas_disponibles], columnas_disponibles)
                export_to_pdf(f"Informe físico de {selected_player}", contenido_pdf, autor="Ramón González", output_path=pdf_filename)

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
                st.warning("No hay columnas disponibles para exportar.")

    with col3:
        if st.button("⏻ Exit", use_container_width=True):
            logout()

# Punto de entrada para ejecución directa
if __name__ == "__main__":
    app()