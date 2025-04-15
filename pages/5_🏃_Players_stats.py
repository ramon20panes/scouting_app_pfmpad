import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from pathlib import Path
import os
import sys
import statsmodels 

# Importación utilidades
from utils.auth import check_auth
from utils.styles import load_all_styles
from utils.players_stats import load_player_data, calculate_improvement, plot_player_metric, generate_rankings

from utils.export_pdf import export_to_pdf, dataframe_a_pdf_contenido

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Jugadores",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos
load_all_styles()

# Autenticación
if not check_auth():
    st.warning("No estás autenticado. Por favor, inicia sesión.")
    st.stop()

# Ruta raíz al path para importar módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
        st.markdown('<h2 class="main-title">Análisis Avanzado de Jugadores</h2>', unsafe_allow_html=True)
        st.markdown("<h4>Visualización interactiva del rendimiento de los jugadores</h4>", unsafe_allow_html=True)

    with col2:

        ESCUDO_PATH = Path("assets/images/logos/atm.png")
        try:
            st.image(ESCUDO_PATH, width=100)
        except FileNotFoundError:
            st.warning("Logo no encontrado. Verifica la ruta del escudo.")

    # Datos de los jugadores
    @st.cache_data
    def load_data():
        csv_file = "data/raw/players_atm_x_jorna_24_25_pag5.csv"
        try:
            df = load_player_data(csv_file)
            
            if df is not None:
                df.columns = df.columns.str.strip()
            return df
        except Exception as e:
            st.error(f"Error al cargar datos: {e}")
            return None

    df = load_data()

    # Definir todas las métricas disponibles una sola vez
    todas_las_metricas = [
        'Min', 'Pases', 'Pases_int', '%_pases', 'Pases_progr', 'Pases_compl', 'Acc_creacion_tiro', 'Acc_creacion_gol',
        'Gls', 'Asist', 'xG', 'xA', 'xAG', 'Penaltis', 'Disp.', 'Disp_a_puerta', 'Tiros', 'Num_toques',
        'Entradas', 'Intercep.', 'Bloqueos', '%_duelos_aereos', 'Recuperac', 'Duelos_aereos_gan', '%_acosos',
        'Pases_clave', 'Pases_ult_tercio', 'Pases_area', 'Dist_tt_pases', 'Dist_progr_pases', 'Centros', 'Pases_prof',
        'Conducc_progr', 'Acosos_int', 'Exito_acosos', 'Pases_cort_cmp', 'Pases_cort_int', '%_cortos',
        'Pases_med_cmp', 'Pases_med_int', '%_medios', 'Pases_lrg_cmp', 'Pases_lrg_int', '%_largos',
        'Pases_en_vivo', 'Pases_balon_parado', 'Saques_falta', 'Pases_lrg', 'Cambio_orient', 'Pases_cruzad',
        'Pases_lanz.', 'Saques_córner', 'Entradas_gan', 'Ent_prim_terc', 'Ent_seg_terc', 'Ent_ult_terc',
        'DF_acosos', 'Exit_acosos', 'Acosos_perd', 'Bloq_tiro', 'Bloq_pase', 'Tkls+intercp', 'Despejes',
        'Error_+_tiro', 'Toq_prop_area', 'Toq_prim_terc', 'Toq_seg_terc', 'Toq_ult_terc', 'Toq_area_riv',
        'Dist_tt_traslado', 'Dist_tt_prog', 'Conducc_ult_terc', 'Conducc_area_riv', 'Perd_control', 'Perdida',
        'Pases_recib', 'Pases_recib_prof', 'Faltas_com', 'Faltas_recib', 'Penal_recib', 'Penal_comet', 'Gol_recib',
        'Duelos_aereos_perd', 'Tarj_am', 'Tarj_rj', 'Fueras_de_juego'
    ]

    # Solo nos quedamos con las que realmente estén en el dataframe
    metricas_disponibles = [m for m in todas_las_metricas if m in df.columns]

    if df is None:
        st.error("No se pudieron cargar los datos. Verifica la ruta del archivo.")
        st.stop()

    # CSS personalizado para mejorar la apariencia
    st.markdown("""
    <style>
        .metric-card {
            background-color: white;
            border-radius: 5px;
            padding: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
            margin-bottom: 10px;
        }
        .metric-title {
            font-size: 14px;
            color: #555;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
        }
        .trend-up {
            color: #4CAF50;
        }
        .trend-down {
            color: #F44336;
        }
        .player-selector {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 15px;
        }
        .visualization-container {
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 1px;
        }
        .stTabs [data-baseweb="tab"] {
            padding-top: 10px;
            padding-bottom: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Tabs para diferentes visualizaciones
    tab1, tab2, tab3, tab4 = st.tabs(["Evolución por Jornada", "Comparativa de Jugadores", "Análisis de Métricas", "Ranking de Rendimiento"])

    with tab1:
        st.markdown("### Evolución del rendimiento por jornada")
        col1, col2 = st.columns([1, 3])         
        with col1:
            st.markdown('<div class="player-selector">', unsafe_allow_html=True)

            # Jugadores disponibles
            jugadores_filtrados = df['Nombre'].unique()

            # Definir una clave única para mantenerlo persistente entre tabs
            if 'jugador_seleccionado' not in st.session_state:
                st.session_state.jugador_seleccionado = "Julián Álvarez" if "Julián Álvarez" in jugadores_filtrados else sorted(jugadores_filtrados)[0]

            st.session_state.jugador_seleccionado = st.selectbox(
                "Seleccionar jugador:",
                sorted(jugadores_filtrados),
                index=sorted(jugadores_filtrados).index(st.session_state.jugador_seleccionado)
            )

            jugador_seleccionado = st.session_state.jugador_seleccionado 
            
            # Definir métricas por defecto al cargar
            metricas_por_defecto = ['Pases_int', 'Duelos_aereos_gan', 'Tiros']
            metricas_default = [m for m in metricas_por_defecto if m in metricas_disponibles]

            if "metricas_seleccionadas" not in st.session_state:
                st.session_state.metricas_seleccionadas = metricas_default

            st.session_state.metricas_seleccionadas = st.multiselect(
                "Seleccionar métricas:",
                metricas_disponibles,
                default=st.session_state.metricas_seleccionadas,
                key="metricas_jugador"
            )
            metricas_seleccionadas = st.session_state.metricas_seleccionadas
            
            # Rango de jornadas
            jornadas = sorted(df['Jornada'].unique())
            jornada_inicio, jornada_fin = st.select_slider(
                "Rango de jornadas:",
                options=jornadas,
                value=(min(jornadas), max(jornadas))
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Métricas resumen
            if jugador_seleccionado and metricas_seleccionadas:
                st.markdown("### Métricas clave")
                
                # Filtrar datos del jugador
                datos_jugador = df[(df['Nombre'] == jugador_seleccionado) & 
                                   (df['Jornada'] >= jornada_inicio) & 
                                   (df['Jornada'] <= jornada_fin)]
                
                # Crear tarjetas de métricas
                for metrica in metricas_seleccionadas:
                    if metrica in datos_jugador.columns:
                        valor_medio = datos_jugador[metrica].mean()
                        
                        # Calcular tendencia
                        if len(datos_jugador) >= 2:
                            primer_valor = datos_jugador.sort_values('Jornada').iloc[0][metrica]
                            ultimo_valor = datos_jugador.sort_values('Jornada').iloc[-1][metrica]
                            tendencia = ultimo_valor - primer_valor
                            
                            # Icono de tendencia
                            if tendencia > 0:
                                icono = "↗️"
                                clase_tendencia = "trend-up"
                            elif tendencia < 0:
                                icono = "↘️"
                                clase_tendencia = "trend-down"
                            else:
                                icono = "➡️"
                                clase_tendencia = ""
                        else:
                            icono = ""
                            clase_tendencia = ""
                            
                        # Mostrar tarjeta
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">{metrica}</div>
                            <div class="metric-value {clase_tendencia}">{valor_medio:.2f} {icono}</div>
                        </div>
                        """, unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="visualization-container">', unsafe_allow_html=True)          
            if jugador_seleccionado and metricas_seleccionadas:
                # Filtrar datos del jugador para el rango de jornadas seleccionado
                datos_jugador = df[(df['Nombre'] == jugador_seleccionado) & 
                                   (df['Jornada'] >= jornada_inicio) & 
                                   (df['Jornada'] <= jornada_fin)]
                
                if not datos_jugador.empty:
                    # Crear gráfico de evolución con Plotly
                    fig = go.Figure()
                    
                    # Colores para las diferentes métricas
                    colores = px.colors.qualitative.Plotly
                    
                    for i, metrica in enumerate(metricas_seleccionadas):
                        if metrica in datos_jugador.columns:
                            fig.add_trace(go.Scatter(
                                x=datos_jugador['Jornada'],
                                y=datos_jugador[metrica],
                                mode='lines+markers',
                                name=metrica,
                                line=dict(color=colores[i % len(colores)], width=3),
                                marker=dict(size=8)
                            ))
                    
                    # Personalizar el diseño
                    fig.update_layout(
                        title=f"Evolución de {jugador_seleccionado} por jornada",
                        xaxis_title="Jornada",
                        yaxis_title="Valor",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        height=500,
                        hovermode="x unified"
                    )
                    
                    # Añadir línea de tendencia para la primera métrica si hay suficientes datos
                    if len(metricas_seleccionadas) > 0 and len(datos_jugador) > 2:
                        metrica_principal = metricas_seleccionadas[0]
                        x = datos_jugador['Jornada']
                        y = datos_jugador[metrica_principal]
                        
                        # Calcular línea de tendencia
                        z = np.polyfit(x, y, 1)
                        p = np.poly1d(z)
                        
                        fig.add_trace(go.Scatter(
                            x=x,
                            y=p(x),
                            mode='lines',
                            name=f'Tendencia {metrica_principal}',
                            line=dict(color='rgba(0,0,0,0.3)', width=2, dash='dash')
                        ))
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tabla de datos
                    st.markdown("### Datos por jornada")
                    st.dataframe(
                        datos_jugador[['Jornada'] + metricas_seleccionadas].sort_values('Jornada'),
                        hide_index=True
                    )
                else:
                    st.warning(f"No hay datos disponibles para {jugador_seleccionado} en el rango seleccionado.")
            else:
                st.info("Selecciona un jugador y al menos una métrica para visualizar su evolución.")
            
            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown("### Comparativa de jugadores")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown('<div class="player-selector">', unsafe_allow_html=True)

            jugadores_disponibles = sorted(df['Nombre'].unique())
            jugadores_por_defecto = ["Julián Álvarez", "Antoine Griezmann", "Pablo Barrios"]
            jugadores_default = [j for j in jugadores_por_defecto if j in jugadores_disponibles]

            jugadores_seleccionados = st.multiselect(
                "Seleccionar jugadores para comparar:",
                jugadores_disponibles,
                default=jugadores_default[:3],
                max_selections=3
            )
            
            # Selección de métrica para comparar
            metrica_comparacion = st.selectbox(
                "Métrica para comparar:",
                metricas_disponibles
            )
                        
            # Tipo de visualización
            tipo_viz = st.radio(
                "Tipo de visualización:",
                ["Barras", "Radar", "Heatmap"],
                horizontal=True
            )
            
            # Mostrar datos agregados o por jornada
            if tipo_viz != "Heatmap":
                agregar_datos = st.checkbox("Mostrar promedio por jugador", value=True)
            else:
                agregar_datos = False
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="visualization-container">', unsafe_allow_html=True)
            
            if jugadores_seleccionados and metrica_comparacion:
                # Crear dataframe filtrado
                if agregar_datos:
                    # Calcular promedio por jugador
                    datos_comp = df[df['Nombre'].isin(jugadores_seleccionados)].groupby('Nombre')[metrica_comparacion].mean().reset_index()
                    datos_comp = datos_comp.sort_values(metrica_comparacion, ascending=False)
                else:
                    # Usar datos por jornada
                    datos_comp = df[df['Nombre'].isin(jugadores_seleccionados)][['Nombre', 'Jornada', metrica_comparacion]]
                
                # Crear visualización según tipo seleccionado
                if tipo_viz == "Barras":
                    if agregar_datos:
                        # Gráfico de barras para promedios
                        fig = px.bar(
                            datos_comp,
                            x='Nombre',
                            y=metrica_comparacion,
                            title=f"Comparativa de {metrica_comparacion} por jugador",
                            color='Nombre',
                            color_discrete_sequence=px.colors.qualitative.Plotly
                        )
                        
                        # Personalizar diseño
                        fig.update_layout(
                            xaxis_title="Jugador",
                            yaxis_title=metrica_comparacion,
                            height=500
                        )
                    else:
                        # Gráfico de barras agrupadas por jornada
                        fig = px.bar(
                            datos_comp,
                            x='Jornada',
                            y=metrica_comparacion,
                            color='Nombre',
                            barmode='group',
                            title=f"Comparativa de {metrica_comparacion} por jornada",
                            color_discrete_sequence=px.colors.qualitative.Plotly
                        )
                        
                        # Personalizar diseño
                        fig.update_layout(
                            xaxis_title="Jornada",
                            yaxis_title=metrica_comparacion,
                            height=500,
                            xaxis=dict(tickmode='linear')
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif tipo_viz == "Radar":
                    # Seleccionar múltiples métricas para radar chart
                    metricas_radar = [m for m in ['Min', 'Pases', 'Pases_int', '%_pases', 'Pases_progr', 'Pases_compl','Acc_creacion_tiro', 'Acc_creacion_gol',
                            'Gls', 'Asist', 'xG', 'xAG','xA','Penaltis', 'Disp.', 'Disp_a_puerta','Tiros','Num_toques',
                            'Entradas', 'Intercep.', 'Bloqueos','%_duelos_aereos','Recuperac', 'Duelos_aereos_gan','%_acosos',
                            'Pases_clave', 'Pases_ult_tercio', 'Pases_area','Dist_tt_pases', 'Dist_progr_pases','Centros', 'Pases_prof'
                            ] if m in df.columns]
                    # Eliminar métricas duplicadas
                    metricas_radar = list(dict.fromkeys(metricas_radar))

                    if len(metricas_radar) > 2:
                        # Crear datos para radar chart
                        if agregar_datos:
                            # Calcular promedio por jugador para todas las métricas
                            radar_data = df[df['Nombre'].isin(jugadores_seleccionados)].groupby('Nombre')[metricas_radar].mean().reset_index()
                        else:
                            # Usar última jornada disponible para cada jugador
                            ultima_jornada = df.groupby('Nombre')['Jornada'].max().reset_index()
                            radar_data = pd.DataFrame()
                            
                            for jugador in jugadores_seleccionados:
                                try:
                                    jornada_max = ultima_jornada[ultima_jornada['Nombre'] == jugador]['Jornada'].values[0]
                                    datos_jugador = df[(df['Nombre'] == jugador) & (df['Jornada'] == jornada_max)][['Nombre'] + metricas_radar]
                                    radar_data = pd.concat([radar_data, datos_jugador])
                                except:
                                    pass
                        
                        # Normalizar los datos para visualización en radar
                        for metrica in metricas_radar:
                            max_val = radar_data[metrica].max()
                            if max_val.max() > 0:  
                                radar_data[f"{metrica}_norm"] = radar_data[metrica] / max_val
                            else:
                                radar_data[f"{metrica}_norm"] = radar_data[metrica]
                        
                        # Crear figura
                        fig = go.Figure()
                        
                        # Añadir un trazo por jugador
                        for i, jugador in enumerate(radar_data['Nombre'].unique()):
                            datos_jug = radar_data[radar_data['Nombre'] == jugador]
                            
                            fig.add_trace(go.Scatterpolar(
                                r=[datos_jug[f"{m}_norm"].values[0] for m in metricas_radar],
                                theta=metricas_radar,
                                fill='toself',
                                name=jugador,
                                line_color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                            ))
                        
                        # Actualizar layout
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1]
                                )
                            ),
                            title="Perfil de jugadores (valores normalizados)",
                            height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Mostrar datos originales en tabla
                        st.markdown("### Valores originales")
                        st.dataframe(radar_data[['Nombre'] + metricas_radar], hide_index=True)
                    else:
                        st.warning("Se necesitan al menos 3 métricas disponibles para crear un gráfico de radar.")
                
                elif tipo_viz == "Heatmap":
                    # Crear pivot table para heatmap
                    pivot_data = df[df['Nombre'].isin(jugadores_seleccionados)].pivot_table(
                        index='Nombre',
                        columns='Jornada',
                        values=metrica_comparacion,
                        aggfunc='mean'
                    )
                    
                    # Crear heatmap con Plotly
                    fig = px.imshow(
                        pivot_data,
                        labels=dict(x="Jornada", y="Jugador", color=metrica_comparacion),
                        x=pivot_data.columns,
                        y=pivot_data.index,
                        color_continuous_scale="Blues",
                        aspect="auto",
                        title=f"Heatmap de {metrica_comparacion} por jornada"
                    )
                    
                    # Añadir anotaciones con valores
                    annotations = []
                    for i, jugador in enumerate(pivot_data.index):
                        for j, jornada in enumerate(pivot_data.columns):
                            value = pivot_data.iloc[i, j]
                            if not np.isnan(value):  # Solo añadir anotación si el valor no es NaN
                                annotations.append(dict(
                                    x=jornada,
                                    y=jugador,
                                    text=str(round(value, 1)),
                                    showarrow=False,
                                    font=dict(color="white" if value > pivot_data.max().max()/2 else "black")
                                ))
                    
                    fig.update_layout(annotations=annotations, height=400)
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Selecciona al menos un jugador y una métrica para visualizar la comparación.")
            
            st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown("### Análisis detallado de métricas")

        col1, col2 = st.columns([1, 3])

        with col1:
            st.markdown('<div class="player-selector">', unsafe_allow_html=True)

            metricas_analisis = [m for m in ['Min', 'Pases', 'Pases_int', '%_pases', 'Pases_progr', 'Pases_compl','Acc_creacion_tiro', 'Acc_creacion_gol',
                            'Gls', 'Asist', 'xG', 'xA', 'xAG','xA','Penaltis', 'Disp.', 'Disp_a_puerta','Tiros','Num_toques',
                            'Entradas', 'Intercep.', 'Bloqueos','%_duelos_aereos','Recuperac', 'Duelos_aereos_gan','%_acosos',
                            'Pases_clave', 'Pases_ult_tercio', 'Pases_area','Dist_tt_pases', 'Dist_progr_pases','Centros', 'Pases_prof',
                            'Conducc_progr', 'Acosos_int', 'Exito_acosos', 'Pases_cort_cmp', 'Pases_cort_int', '%_cortos', 'Pases_med_cmp',
                            'Pases_med_int', '%_medios', 'Pases_lrg_cmp', 'Pases_lrg_int', '%_largos', 'Pases_en_vivo', 'Pases_balon_parado',
                            'Saques_falta', 'Pases_lrg', 'Cambio_orient','Pases_cruzad', 'Pases_lanz.', 'Saques_córner', 'Entradas_gan', 
                            'Ent_prim_terc',  'Ent_seg_terc', 'Ent_ult_terc', 'DF_acosos', 'Exit_acosos',  'Acosos_perd', 'Bloq_tiro', 
                            'Bloq_pase', 'Tkls+intercp', 'Despejes', 'Error_+_tiro', 'Toq_prop_area', 'Toq_prim_terc', 'Toq_seg_terc', 
                            'Toq_ult_terc', 'Toq_area_riv', 'Dist_tt_traslado',  'Dist_tt_prog', 'Conducc_ult_terc', 'Conducc_area_riv', 
                            'Perd_control', 'Perdida', 'Pases_recib', 'Pases_recib_prof', 'Faltas_com', 'Faltas_recib','Penal_recib', 
                            'Penal_comet', 'Gol_recib', 'Duelos_aereos_perd', 'Tarj_am', 'Tarj_rj', 'Fueras_de_juego'] 
                            if m in df.columns]
            
            metrica_analisis = st.selectbox("Seleccionar métrica para analizar:", metricas_analisis)

            tipo_analisis = st.radio(
                "Tipo de análisis:",
                ["Distribución", "Correlación", "Tendencia temporal"],
                horizontal=True
            )

            if tipo_analisis == "Correlación":
                segunda_metrica = st.selectbox(
                    "Correlacionar con:",
                    [m for m in metricas_analisis if m != metrica_analisis]
                )

            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="visualization-container">', unsafe_allow_html=True)

            if metrica_analisis:
                if tipo_analisis == "Distribución":
                    # Histograma general
                    fig = px.histogram(
                        df,
                        x=metrica_analisis,
                        title=f"Distribución de {metrica_analisis}",
                        opacity=0.8,
                        marginal="box"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Estadísticas descriptivas
                    st.markdown("### Estadísticas descriptivas")
                    stats_desc = df[metrica_analisis].describe().reset_index()
                    stats_desc.columns = ['Estadística', 'Valor']
                    st.dataframe(stats_desc, hide_index=True)

                    # Top jugadores
                    st.markdown("### Top 5 jugadores")
                    top_jugadores = df.groupby('Nombre')[metrica_analisis].mean().sort_values(ascending=False).head(5).reset_index()

                    fig_top = px.bar(
                        top_jugadores,
                        x='Nombre',
                        y=metrica_analisis,
                        title=f"Mejores jugadores en {metrica_analisis}",
                        color=metrica_analisis,
                        color_continuous_scale="Blues"
                    )
                    st.plotly_chart(fig_top, use_container_width=True)

                elif tipo_analisis == "Correlación" and 'segunda_metrica' in locals():
                    try:
                        fig = px.scatter(
                            df,
                            x=metrica_analisis,
                            y=segunda_metrica,
                            title=f"Correlación entre {metrica_analisis} y {segunda_metrica}",
                            color='Nombre',
                            hover_data=['Nombre', 'Jornada'],
                            opacity=0.7,
                            trendline="ols"
                        )
                    except Exception as e:
                        st.warning(f"No se pudo calcular la línea de tendencia. Error: {e}")
                        fig = px.scatter(
                            df,
                            x=metrica_analisis,
                            y=segunda_metrica,
                            title=f"Correlación entre {metrica_analisis} y {segunda_metrica}",
                            color='Nombre',
                            hover_data=['Nombre', 'Jornada'],
                            opacity=0.7
                        )

                    st.plotly_chart(fig, use_container_width=True)

                    # Calcular correlación
                    corr_value = df[[metrica_analisis, segunda_metrica]].corr().iloc[0, 1]
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Coeficiente de correlación</div>
                        <div class="metric-value" style="color: {'blue' if corr_value > 0 else 'red'}">{corr_value:.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    if corr_value > 0.7:
                        interpretacion = "Correlación fuerte positiva"
                    elif corr_value > 0.3:
                        interpretacion = "Correlación moderada positiva"
                    elif corr_value > -0.3:
                        interpretacion = "Correlación débil o nula"
                    elif corr_value > -0.7:
                        interpretacion = "Correlación moderada negativa"
                    else:
                        interpretacion = "Correlación fuerte negativa"

                    st.markdown(f"**Interpretación:** {interpretacion}")

                    if st.checkbox("Mostrar datos completos"):
                        st.dataframe(df[['Nombre', 'Jornada', metrica_analisis, segunda_metrica]].sort_values(metrica_analisis, ascending=False), hide_index=True)

                elif tipo_analisis == "Tendencia temporal":
                    tendencia_df = df.groupby('Jornada')[metrica_analisis].mean().reset_index()
                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                        x=tendencia_df['Jornada'],
                        y=tendencia_df[metrica_analisis],
                        mode='lines+markers',
                        name='Media general',
                        line=dict(color='black', width=3),
                        marker=dict(size=8)
                    ))

                    fig.update_layout(
                        title=f"Evolución de {metrica_analisis} a lo largo de la temporada",
                        xaxis_title="Jornada",
                        yaxis_title=metrica_analisis,
                        height=500,
                        xaxis=dict(tickmode='linear')
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    if len(tendencia_df) > 3:
                        x = tendencia_df['Jornada']
                        y = tendencia_df[metrica_analisis]
                        z = np.polyfit(x, y, 1)
                        pendiente = z[0]

                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">Tendencia general</div>
                            <div class="metric-value" style="color: {'green' if pendiente > 0 else 'red'}">{pendiente:.4f} por jornada</div>
                        </div>
                        """, unsafe_allow_html=True)

                        if pendiente > 0:
                            st.markdown("**Interpretación:** La métrica muestra una tendencia creciente a lo largo de la temporada.")
                        elif pendiente < 0:
                            st.markdown("**Interpretación:** La métrica muestra una tendencia decreciente a lo largo de la temporada.")
                        else:
                            st.markdown("**Interpretación:** La métrica se mantiene estable a lo largo de la temporada.")
            else:
                st.info("Selecciona una métrica para comenzar el análisis.")

            st.markdown('</div>', unsafe_allow_html=True)


    with tab4:
        st.markdown("### Ranking de rendimiento")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown('<div class="player-selector">', unsafe_allow_html=True)

            metrica_ranking = st.selectbox(
                "Métrica para ranking:",
                [m for m in ['Min', 'Pases', 'Pases_int', '%_pases', 'Pases_progr', 'Pases_compl','Acc_creacion_tiro', 'Acc_creacion_gol',
                            'Gls', 'Asist', 'xG', 'xA', 'xAG','xA','Penaltis', 'Disp.', 'Disp_a_puerta','Tiros','Num_toques',
                            'Entradas', 'Intercep.', 'Bloqueos','%_duelos_aereos','Recuperac', 'Duelos_aereos_gan','%_acosos',
                            'Pases_clave', 'Pases_ult_tercio', 'Pases_area','Dist_tt_pases', 'Dist_progr_pases','Centros', 'Pases_prof',
                            'Conducc_progr', 'Acosos_int', 'Exito_acosos', 'Pases_cort_cmp', 'Pases_cort_int', '%_cortos', 'Pases_med_cmp',
                            'Pases_med_int', '%_medios', 'Pases_lrg_cmp', 'Pases_lrg_int', '%_largos', 'Pases_en_vivo', 'Pases_balon_parado',
                            'Saques_falta', 'Pases_lrg', 'Cambio_orient','Pases_cruzad', 'Pases_lanz.', 'Saques_córner', 'Entradas_gan', 
                            'Ent_prim_terc',  'Ent_seg_terc', 'Ent_ult_terc', 'DF_acosos', 'Exit_acosos',  'Acosos_perd', 'Bloq_tiro', 
                            'Bloq_pase', 'Tkls+intercp', 'Despejes', 'Error_+_tiro', 'Toq_prop_area', 'Toq_prim_terc', 'Toq_seg_terc', 
                            'Toq_ult_terc', 'Toq_area_riv', 'Dist_tt_traslado',  'Dist_tt_prog', 'Conducc_ult_terc', 'Conducc_area_riv', 
                            'Perd_control', 'Perdida', 'Pases_recib', 'Pases_recib_prof', 'Faltas_com', 'Faltas_recib','Penal_recib', 
                            'Penal_comet', 'Gol_recib', 'Duelos_aereos_perd', 'Tarj_am', 'Tarj_rj', 'Fueras_de_juego'] 
                            if m in df.columns],

                key="ranking_metric"
            )

            top_n = st.slider("Mostrar top jugadores:", min_value=5, max_value=20, value=10)

            tipo_ranking = st.radio(
                "Tipo de visualización:",
                ["Tabla", "Gráfico de barras", "Gráfico circular"],
                horizontal=True
            )

            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="visualization-container">', unsafe_allow_html=True)

            if metrica_ranking:
                # Ya no se filtra por posición
                df_filtrado = df

                # Calcular ranking
                ranking = df_filtrado.groupby('Nombre')[metrica_ranking].mean().sort_values(ascending=False).reset_index()
                ranking = ranking.head(top_n)
                ranking['Posición en ranking'] = range(1, len(ranking) + 1)

                if tipo_ranking == "Tabla":
                    st.markdown(f"### Top {len(ranking)} jugadores por {metrica_ranking}")
                    st.dataframe(ranking[['Posición en ranking', 'Nombre', metrica_ranking]], hide_index=True)

                    if st.checkbox("Mostrar estadísticas detalladas"):
                        st.markdown("### Estadísticas detalladas")
                        stats_detalladas = df_filtrado[df_filtrado['Nombre'].isin(ranking['Nombre'])].groupby('Nombre')[
                            [m for m in ['Min', 'Pases', 'Pases_int', '%_pases', 'Pases_progr', 'Pases_compl','Acc_creacion_tiro', 'Acc_creacion_gol',
                            'Gls', 'Asist', 'xG', 'xA', 'xAG','xA','Penaltis', 'Disp.', 'Disp_a_puerta','Tiros','Num_toques',
                            'Entradas', 'Intercep.', 'Bloqueos','%_duelos_aereos','Recuperac', 'Duelos_aereos_gan','%_acosos',
                            'Pases_clave', 'Pases_ult_tercio', 'Pases_area','Dist_tt_pases', 'Dist_progr_pases','Centros', 'Pases_prof',
                            'Conducc_progr', 'Acosos_int', 'Exito_acosos', 'Pases_cort_cmp', 'Pases_cort_int', '%_cortos', 'Pases_med_cmp',
                            'Pases_med_int', '%_medios', 'Pases_lrg_cmp', 'Pases_lrg_int', '%_largos', 'Pases_en_vivo', 'Pases_balon_parado',
                            'Saques_falta', 'Pases_lrg', 'Cambio_orient','Pases_cruzad', 'Pases_lanz.', 'Saques_córner', 'Entradas_gan', 
                            'Ent_prim_terc',  'Ent_seg_terc', 'Ent_ult_terc', 'DF_acosos', 'Exit_acosos',  'Acosos_perd', 'Bloq_tiro', 
                            'Bloq_pase', 'Tkls+intercp', 'Despejes', 'Error_+_tiro', 'Toq_prop_area', 'Toq_prim_terc', 'Toq_seg_terc', 
                            'Toq_ult_terc', 'Toq_area_riv', 'Dist_tt_traslado',  'Dist_tt_prog', 'Conducc_ult_terc', 'Conducc_area_riv', 
                            'Perd_control', 'Perdida', 'Pases_recib', 'Pases_recib_prof', 'Faltas_com', 'Faltas_recib','Penal_recib', 
                            'Penal_comet', 'Gol_recib', 'Duelos_aereos_perd', 'Tarj_am', 'Tarj_rj', 'Fueras_de_juego'] 
                            if m in df.columns]

                        ].agg(['mean', 'sum', 'max']).reset_index()

                        # Añade estas líneas para arreglar los nombres de columnas duplicados
                        # Obtenemos la lista de nombres de columnas actuales
                        columnas_actuales = stats_detalladas.columns.tolist()

                        # Creamos un nuevo conjunto de nombres de columnas
                        nuevas_columnas = []
                        for col in columnas_actuales:
                            if col == 'Nombre':  # Mantener la columna 'Nombre' sin cambios
                                nuevas_columnas.append(col)
                            else:
                                # Para columnas multiíndice, crear un nombre combinado
                                metrica, agregacion = col
                                nuevas_columnas.append(f"{metrica}_{agregacion}")

                        # Asignar los nuevos nombres de columnas
                        stats_detalladas.columns = nuevas_columnas
                        # Mostrar información sobre las columnas
                        st.write("Nombres de columnas:", stats_detalladas.columns.tolist())
                        st.write("Estructura de columnas:", stats_detalladas.columns)
                        st.write("Primeras filas:", stats_detalladas.head())
                        st.dataframe(stats_detalladas)

                elif tipo_ranking == "Gráfico de barras":
                    fig = px.bar(
                        ranking,
                        y='Nombre',
                        x=metrica_ranking,
                        title=f"Top {len(ranking)} jugadores por {metrica_ranking}",
                        color=metrica_ranking,
                        orientation='h',
                        color_continuous_scale="Blues",
                        text='Posición en ranking'
                    )
                    fig.update_layout(
                        yaxis=dict(categoryorder='total ascending'),
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)

                elif tipo_ranking == "Gráfico circular":
                    pie_data = ranking.head(8).copy()
                    if len(ranking) > 8:
                        otros_valor = ranking[ranking['Posición en ranking'] > 8][metrica_ranking].sum()
                        pie_data = pd.concat([pie_data, pd.DataFrame({'Nombre': ['Otros'], metrica_ranking: [otros_valor]})], ignore_index=True)

                    if len(pie_data) > 0:
                        fig = px.pie(
                            pie_data,
                            values=metrica_ranking,
                            names='Nombre',
                            title=f"Distribución de {metrica_ranking} entre top jugadores"
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No hay datos suficientes para crear un gráfico circular.")
            else:
                st.info("Selecciona una métrica para generar el ranking.")

            st.markdown('</div>', unsafe_allow_html=True)

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
            display_columns = ["Nombre", "Jornada"] + [col for col in ['Min', 'Pases', 'Tiros', 'Gls', 'Asist'] if col in df.columns]
            contenido_pdf = dataframe_a_pdf_contenido(df, display_columns)
            pdf_filename = "informe_jugadores.pdf"

            export_to_pdf("Informe Avanzado de Jugadores", contenido_pdf, autor="Ramón González", output_path=pdf_filename)

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

# Punto de entrada para ejecución directa
if __name__ == "__main__":
    app()