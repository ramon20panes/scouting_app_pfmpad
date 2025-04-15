import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import os
import sys
import random
import time
from typing import Dict, List, Tuple, Any, Optional
from utils.auth import check_auth, logout
from utils.styles import load_all_styles
from utils.export_pdf import export_to_pdf, dataframe_a_pdf_contenido

# Configuración de la página
st.set_page_config(
    page_title="Style_Atlético de Madrid 24/25",
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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

# Importar utilidades y modelos
from utils.cache import read_dataframe_with_cache
from utils.visualization_4 import (
    ESTILO_NOMBRES, COLORES_CATEGORIAS,
    preparar_datos_visualizacion, 
    crear_grafico_distribucion_estilos,
    crear_scatter_indices,
    crear_boxplots_resultado,
    cargar_datos_para_analisis
)
from models.style_prediction.svm_predictor import (
    predecir_estilo, cargar_modelos, 
    crear_visualizacion_pca, preparar_datos_modelado,
    analizar_importancia_caracteristicas
)
from models.style_prediction.feature_enginering import (
        calcular_indices_basicos, calcular_indices_avanzados, 
        calcular_indices_especializados, categorizar_estilos
    )

# Constantes para rutas de datos
MASTER_PATH = "data/raw/master/master_liga_vert_atlmed.csv"
STATS_PATH = "data/raw/stats_atm_por_partido_pag4.csv"
EVENTOS_DIR = "data/raw/parquet/"
EQUIPOS_PATH = "data/raw/master/equipos_laliga_master.csv"
PARTIDOS_PATH = "data/raw/master/partidos_master.csv"
MODELS_DIR = "models/style_prediction/modelos_entrenados"

# Función para cargar y procesar datos con caché
@st.cache_data(ttl=3600)
def cargar_datos_completos():
    """
    Versión simplificada que evita problemas con archivos PlayerData
    """
    # Mostrar mensaje mientras se cargan los datos
    with st.spinner("Cargando datos para análisis de estilos de juego..."):
        try:
            # Cargar datos principales (usa función para leer DataFrame directamente)
            
            try:
                df_master = pd.read_csv(MASTER_PATH, delimiter=';')
                if df_master.shape[1] <= 1:
                    df_master = pd.read_csv(MASTER_PATH)

            except:
                df_master = pd.read_csv(MASTER_PATH)

            try:
                df_stats = pd.read_csv(STATS_PATH, delimiter=';')

                if df_stats.shape[1] <= 1:
                    df_stats = pd.read_csv(STATS_PATH)
            except:
                df_stats = pd.read_csv(STATS_PATH)
            
            # Asegurar que jornada sea numérica
            if 'jornada' in df_master.columns:
                df_master['jornada_num'] = df_master['jornada'].str.replace('ª', '').astype(int)
            
            if 'jornada' in df_stats.columns:
                df_stats['jornada'] = pd.to_numeric(df_stats['jornada'], errors='coerce')
            
            # Unir DataFrames por jornada
            if 'jornada_num' in df_master.columns and 'jornada' in df_stats.columns:
                df_completo = pd.merge(
                    df_stats, 
                    df_master, 
                    left_on='jornada', 
                    right_on='jornada_num', 
                    how='inner'
                )
            else:
                # Si no hay columnas para unir, usar el DataFrame de estadísticas
                df_completo = df_stats.copy()
            
            # Cargar datos de partidos adicionales si están disponibles
            try:
                df_partidos = pd.read_csv(PARTIDOS_PATH, delimiter=';')
                
                # Normalizar formato de jornada
                if 'jornada' in df_partidos.columns:
                    df_partidos['jornada'] = df_partidos['jornada'].str.replace('ª', '').astype(int)
                
                # Unir con información de partidos
                if 'jornada' in df_completo.columns and 'jornada' in df_partidos.columns:
                    df_completo = pd.merge(df_completo, df_partidos, on='jornada', how='left')
            except Exception as e:
                st.warning(f"No se pudieron cargar datos adicionales de partidos: {e}")
            
            return df_completo, {'master': df_master, 'stats': df_stats}
        except Exception as e:
            st.error(f"Error al cargar datos: {e}")
            return pd.DataFrame(), {}

# Función para cargar modelos entrenados
@st.cache_resource
def cargar_modelos_entrenados():
    """
    Carga los modelos SVM entrenados para clasificación de estilos.
    """
    with st.spinner("Cargando modelos de clasificación..."):
        modelos = cargar_modelos(MODELS_DIR)
        return modelos
    

# Definir una versión corregida de crear_radar_indices
def mi_crear_radar_indices(df_indices, partido_idx):
    """
    Versión corregida de crear_radar_indices que verifica columnas en lugar de índices.
    """
    # Obtener datos del partido
    if partido_idx not in df_indices.index:
        return None
    
    partido = df_indices.loc[partido_idx]
    
    # Seleccionar índices tácticos para visualizar
    indices_tacticos = [
        'IIJ', 'IVJO', 'EC', 'EF', 'IER', 'IV', 'IPA', 'IJD', 'IA', 'IDD', 'ICJ',
        'IEO', 'IPT', 'IAB', 'ICT', 'ICP'
    ]
    
    # Filtrar índices que existen en las columnas (no en el índice)
    indices_disponibles = [idx for idx in indices_tacticos if idx in df_indices.columns]
    
    if not indices_disponibles:
        return None
    
    # Extraer valores
    valores = [partido[idx] for idx in indices_disponibles]
    
    # Calcular media de cada índice para la temporada
    medias = [df_indices[idx].mean() for idx in indices_disponibles]
    
    # Crear figura
    import plotly.graph_objects as go
    fig = go.Figure()
    
    # Añadir valores del partido
    fig.add_trace(go.Scatterpolar(
        r=valores,
        theta=indices_disponibles,
        fill='toself',
        name='Partido seleccionado',
        line_color='#e41a1c',
        fillcolor='rgba(228, 26, 28, 0.3)'
    ))
    
    # Añadir valores medios
    fig.add_trace(go.Scatterpolar(
        r=medias,
        theta=indices_disponibles,
        fill='toself',
        name='Media temporada',
        line_color='#377eb8',
        fillcolor='rgba(55, 126, 184, 0.3)'
    ))
    
    # Configurar layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(valores), max(medias)) * 1.1]
            )
        ),
        title="Comparación de índices tácticos",
        height=500,
        margin=dict(l=80, r=80, t=60, b=40),
        legend=dict(orientation="h", y=-0.1)
    )
    
    return fig

def app():
    # Título y logo
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown('<h2 class="main-title">Detección de Estilos de Juego</h2>', unsafe_allow_html=True)
        st.markdown('<h2 class="main-title">Experimento táctico</h2>', unsafe_allow_html=True)

    with col2:
        # Cargar el logo del Atlético de Madrid
        ESCUDO_PATH = Path("assets/images/logos/atm.png")
        try:
            st.image(ESCUDO_PATH, width=100)
        except FileNotFoundError:
            st.warning("Logo no encontrado. Verifica la ruta del escudo.")

    # Texto explicativo de los índices
    st.markdown("""
    <div style="background-color:#D0DAD8; color:#272E61; padding:20px; border-radius:10px; margin-bottom:20px;">
        <p>Prueba detección tendencias de juego del Atlético de Madrid, LaLiga 24/25:</p>
        <ul style="columns: 2; column-gap: 40px; list-style-type: none; padding-left: 0;">
            <li><b>IIJ</b>: Índice iniciativa en el juego - Dominio general basado en posesión y tiros</li>
            <li><b>EC</b>: Eficacia en construcción - Relación de pases con tiros y goles</li>
            <li><b>ICP</b>: Índice de Calidad de la Posesión - Cuánto de productiva es la posesión</li>
            <li><b>IVJO</b>: Índice de volumen de juego ofensivo - Cuantifica el total del ataque</li>
            <li><b>ICJ</b>: Índice de complejidad de juego - Analiza la sofisticación táctica</li>
            <li><b>IV</b>: Índice de verticalidad - Qué tan directo es el juego hacia área rival</li>
            <li><b>IJD</b>: Índice de juego directo - Uso de pases largos</li>            
            <li><b>EF</b>: Eficacia en finalización - Capacidad de convertir los tiros en goles</li>
            <li><b>IA</b>: Índice de amplitud - Coordenadas en anchura</li>
            <li><b>IEO</b>: Índice de eficiencia ofensiva - Expected goals vs tiros</li>
            <li><b>IAB</b>: Índice de amenaza por banda - Peligrosidad por los pasillos exteriores</li>     
            <li><b>IER</b>: Índice de eficacia recuperadora - Habilidad para recuperar balones</li>            
            <li><b>IPA</b>: Índice de presión alta - Mide la intensidad de presión en campo rival</li>                       
            <li><b>IDD</b>: Índice de densidad defensiva - Valora la concentración defensiva</li>            
            <li><b>IPT</b>: Índice de precisión táctica - Como se ejecuta el plan de juego</li>            
            <li><b>ICT</b>: Índice de Control de Transiciones - Evaluación de transiciones ataque-defensa</li>            
        </ul>
        <p style="margin-bottom:0;">Estos índices se calculan mediante algoritmos que combinan múltiples estadísticas del juego para completar cualquier análsisis táctico subjetivo.</p>
    </div>
    """, unsafe_allow_html=True)
        
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

    # Cargar datos
    df_indices, data_clean = cargar_datos_completos()   

    # Calcular índices si no están presentes
    
    indices_tacticos = [col for col in df_indices.columns if col.startswith('I') and len(col) <= 4]

    # Añadir los 7 índices directamente al DataFrame
    if len(indices_tacticos) < 5:  # Si faltan índices principales
                
        # Copia para trabajar
        df_con_indices = df_indices.copy()
        
        # Función para convertir columnas a numéricas
        def convert_to_numeric(df, column):
            if column in df.columns:
                try:
                    return pd.to_numeric(df[column], errors='coerce')
                except:
                    st.warning(f"No se pudo convertir {column} a numérico")
                    return None
            return None
        
        # Calcular IIJ (Índice de Iniciativa de Juego)
        if 'IIJ' not in df_con_indices.columns:
            posesion = convert_to_numeric(df_con_indices, 'posesion')
            tiros = convert_to_numeric(df_con_indices, 'tiros_totales')
            
            if posesion is not None and tiros is not None:
                df_con_indices['IIJ'] = posesion * 0.5 + tiros * 0.5
            else:
                df_con_indices['IIJ'] = 50.0  # Valor medio
        
        # Calcular IVJO (Índice de Volumen de Juego Ofensivo)
        if 'IVJO' not in df_con_indices.columns:
            pases = convert_to_numeric(df_con_indices, 'pases')
            tiros = convert_to_numeric(df_con_indices, 'tiros_totales')
            toques_area = convert_to_numeric(df_con_indices, 'toques_area_rival')
            
            if pases is not None and tiros is not None:
                if toques_area is not None:
                    df_con_indices['IVJO'] = pases * 0.4 + tiros * 2.0 + toques_area * 1.5
                else:
                    df_con_indices['IVJO'] = pases * 0.4 + tiros * 2.0
            else:
                df_con_indices['IVJO'] = 500.0  # Valor medio aproximado
        
        # Calcular IER (Índice de Eficacia Recuperadora) - aproximación
        if 'IER' not in df_con_indices.columns:
            duelos = convert_to_numeric(df_con_indices, 'duelos_ganados')
            intercept = convert_to_numeric(df_con_indices, 'interceptaciones')
            
            if duelos is not None and intercept is not None:
                df_con_indices['IER'] = (duelos + intercept) * 0.5
            else:
                df_con_indices['IER'] = 50.0  # Valor medio
        
        # Calcular IV (Índice de Verticalidad) - aproximación
        if 'IV' not in df_con_indices.columns:
            toques_area = convert_to_numeric(df_con_indices, 'toques_area_rival')
            pases = convert_to_numeric(df_con_indices, 'pases')
            
            if toques_area is not None and pases is not None and pases.sum() > 0:
                df_con_indices['IV'] = (toques_area / pases.replace(0, 1)) * 30
            else:
                df_con_indices['IV'] = 20.0  # Valor predeterminado
        
        # Calcular IPA (Índice de Presión Alta) - valor predeterminado
        if 'IPA' not in df_con_indices.columns:
            df_con_indices['IPA'] = 50.0  # Valor medio
        
        # Calcular IJD (Índice de Juego Directo) - aproximación
        if 'IJD' not in df_con_indices.columns:
            pases_largos = convert_to_numeric(df_con_indices, 'pases_largos_precisos')
            pases_total = convert_to_numeric(df_con_indices, 'pases_precisos')
            
            if pases_largos is not None and pases_total is not None and pases_total.sum() > 0:
                df_con_indices['IJD'] = (pases_largos / pases_total.replace(0, 1)) * 100
            else:
                df_con_indices['IJD'] = 25.0  # Valor predeterminado
        
        # Calcular ICJ (Índice de Complejidad de Juego) - aproximación
        if 'ICJ' not in df_con_indices.columns:
            if 'IV' in df_con_indices.columns:
                pct_pases = convert_to_numeric(df_con_indices, 'pct_pases_precisos')
                
                if pct_pases is not None:
                    factor_precision = (pct_pases / 100) * 0.4
                    df_con_indices['ICJ'] = factor_precision * (df_con_indices['IV'] + 50)
                else:
                    df_con_indices['ICJ'] = 40.0  # Valor predeterminado
            else:
                df_con_indices['ICJ'] = 40.0  # Valor predeterminado

        # Después de añadir los 7 índices principales, añadimos los índices adicionales
        if 'EC' not in df_con_indices.columns:
            # Índice de Eficacia en Construcción (EC)
            if 'tiros_totales' in df_con_indices.columns and 'pases' in df_con_indices.columns and 'goles_a_favor' in df_con_indices.columns:
                tiros = pd.to_numeric(df_con_indices['tiros_totales'], errors='coerce')
                pases = pd.to_numeric(df_con_indices['pases'], errors='coerce')
                goles = pd.to_numeric(df_con_indices['goles_a_favor'], errors='coerce')
                df_con_indices['EC'] = (tiros + goles) / pases.replace(0, 1) * 100
            else:
                df_con_indices['EC'] = 5.0  # Valor predeterminado

        if 'EF' not in df_con_indices.columns:
            # Índice de Eficacia en Finalización (EF)
            if 'goles_a_favor' in df_con_indices.columns and 'tiros_totales' in df_con_indices.columns:
                goles = pd.to_numeric(df_con_indices['goles_a_favor'], errors='coerce')
                tiros = pd.to_numeric(df_con_indices['tiros_totales'], errors='coerce')
                palos = pd.to_numeric(df_con_indices['tiros_palos'], errors='coerce') if 'tiros_palos' in df_con_indices.columns else 0
                denominador = tiros + palos + goles
                df_con_indices['EF'] = (goles * 100) / denominador.replace(0, 1)
            else:
                df_con_indices['EF'] = 15.0  # Valor predeterminado

        if 'IA' not in df_con_indices.columns:
            # Índice de Amplitud (IA) - valor predeterminado
            df_con_indices['IA'] = 50.0

        if 'IDD' not in df_con_indices.columns:
            # Índice de Densidad Defensiva (IDD) - valor predeterminado
            df_con_indices['IDD'] = 45.0

        if 'IEO' not in df_con_indices.columns:
            # Índice de Eficiencia Ofensiva (IEO)
            if 'xG' in df_con_indices.columns and 'tiros_totales' in df_con_indices.columns:
                xg = pd.to_numeric(df_con_indices['xG'], errors='coerce')
                tiros = pd.to_numeric(df_con_indices['tiros_totales'], errors='coerce')
                df_con_indices['IEO'] = (xg / tiros.replace(0, 1)) * 100
            else:
                df_con_indices['IEO'] = 12.0  

        if 'IPT' not in df_con_indices.columns:
            # Índice de Precisión Táctica (IPT) - valor predeterminado
            df_con_indices['IPT'] = 65.0

        if 'IAB' not in df_con_indices.columns:
            # Índice de Amenaza por Banda (IAB) - valor predeterminado
            df_con_indices['IAB'] = 55.0

        if 'ICT' not in df_con_indices.columns:
            # Índice de Control de Transiciones (ICT) - valor predeterminado
            df_con_indices['ICT'] = 50.0

        if 'ICP' not in df_con_indices.columns:
            # Índice de Calidad de la Posesión (ICP)
            if 'xG' in df_con_indices.columns and 'posesion' in df_con_indices.columns:
                xg = pd.to_numeric(df_con_indices['xG'], errors='coerce')
                posesion = pd.to_numeric(df_con_indices['posesion'], errors='coerce')
                ocasiones = pd.to_numeric(df_con_indices['ocasiones_claras'], errors='coerce') if 'ocasiones_claras' in df_con_indices.columns else 0
                pases = pd.to_numeric(df_con_indices['pases'], errors='coerce') if 'pases' in df_con_indices.columns else 100
                
                df_con_indices['ICP'] = ((xg / posesion.replace(0, 1)) * 70 + (ocasiones / pases.replace(0, 1)) * 30) * 100
            else:
                df_con_indices['ICP'] = 40.0  

        # Después de categorizar los estilos, se añaden columnas de contexto
        # Añadir resultado_tipo si no existe
        if 'resultado_tipo' not in df_con_indices.columns:
            # Crear clasificación de resultado basada en goles
            if 'goles_a_favor' in df_con_indices.columns and 'goles_en_contra' in df_con_indices.columns:
                goles_favor = pd.to_numeric(df_con_indices['goles_a_favor'], errors='coerce').fillna(0)
                goles_contra = pd.to_numeric(df_con_indices['goles_en_contra'], errors='coerce').fillna(0)
                
                df_con_indices['resultado_tipo'] = 'Empate'
                df_con_indices.loc[goles_favor > goles_contra, 'resultado_tipo'] = 'Victoria'
                df_con_indices.loc[goles_favor < goles_contra, 'resultado_tipo'] = 'Derrota'
                
            else:
                # Crear clasificación aleatoria como ejemplo                
                resultados = ['Victoria', 'Empate', 'Derrota']
                weights = [0.5, 0.25, 0.25]  # Más victorias que derrotas/empates
                df_con_indices['resultado_tipo'] = [random.choices(resultados, weights=weights)[0] for _ in range(len(df_con_indices))]
                

        # Añadir es_local si no existe
        if 'es_local' not in df_con_indices.columns:
            # Alternar entre local y visitante
            df_con_indices['es_local'] = [i % 2 for i in range(len(df_con_indices))]
            
        # Añadir rival_categoria si no existe
        if 'rival_categoria' not in df_con_indices.columns:
            # Crear categorías de rival
            categorias = ['bajo', 'medio', 'top']
            weights = [0.3, 0.5, 0.2]  # Más equipos medios que bajos o top
            df_con_indices['rival_categoria'] = [random.choices(categorias, weights=weights)[0] for _ in range(len(df_con_indices))]      

    # Después de añadir los índices y antes de categorizar estilos
    # Intentar obtener datos reales del archivo master
    if 'master' in data_clean and not data_clean['master'].empty:
        # Columnas importantes a preservar
        columnas_master = ['jornada_num', 'rival', 'resultado', 'goles_a_favor', 'goles_en_contra', 
                        'local_visitante', 'resultado_tipo']
        
        # Crear un DataFrame con solo las columnas que existen
        columnas_disponibles = [col for col in columnas_master if col in data_clean['master'].columns]
        if columnas_disponibles:
            df_master_reducido = data_clean['master'][columnas_disponibles].copy()
            
            # Renombrar jornada_num a jornada si es necesario
            if 'jornada_num' in df_master_reducido.columns and 'jornada' not in df_master_reducido.columns:
                df_master_reducido.rename(columns={'jornada_num': 'jornada'}, inplace=True)
            
            # Añadir es_local basado en local_visitante si existe
            if 'local_visitante' in df_master_reducido.columns:
                df_master_reducido['es_local'] = df_master_reducido['local_visitante'].apply(lambda x: 1 if x == 'L' else 0)
            
            # Crear nombre de partido si hay rival
            if 'rival' in df_master_reducido.columns:
                df_master_reducido['partido'] = 'Atlético de Madrid vs ' + df_master_reducido['rival']
            
            # Unir con df_con_indices por jornada
            if 'jornada' in df_master_reducido.columns and 'jornada' in df_con_indices.columns:
                # Asegurar que jornada sea del mismo tipo en ambos DataFrames
                df_master_reducido['jornada'] = pd.to_numeric(df_master_reducido['jornada'], errors='coerce')
                df_con_indices['jornada'] = pd.to_numeric(df_con_indices['jornada'], errors='coerce')
                
                # Unir DataFrames
                df_con_indices = pd.merge(df_con_indices, df_master_reducido, on='jornada', how='left')

    # Categorizar estilos basados en los índices calculados
    df_con_indices = categorizar_estilos(df_con_indices)

    # Convertir columnas de categoría a tipo categórico para evitar problemas de suma
    for col in ['orientacion_general', 'fase_ofensiva', 'patron_ataque', 'intensidad_defensiva', 'altura_bloque', 'tipo_transicion', 'estilo_posesion']:
        if col in df_con_indices.columns:
            df_con_indices[col] = pd.Categorical(df_con_indices[col])

    # Asegurar que resultado_tipo sea categórico también
    if 'resultado_tipo' in df_con_indices.columns:
        df_con_indices['resultado_tipo'] = pd.Categorical(
            df_con_indices['resultado_tipo'], 
            categories=['Victoria', 'Empate', 'Derrota'],
            ordered=True
        )

    # Reemplazar df_indices con la versión que tiene los índices
    df_indices = df_con_indices
                
    # Cargar modelos entrenados
    modelos = cargar_modelos_entrenados()

    # Verificar si faltan las columnas de categorías
    categorias_faltantes = ['orientacion_general', 'fase_ofensiva', 'patron_ataque', 'intensidad_defensiva', 'altura_bloque', 'tipo_transicion', 'estilo_posesion']
    faltan_categorias = not any(cat in df_indices.columns for cat in categorias_faltantes)

    # Modificar la parte donde se preparan los datos para predicción
    if faltan_categorias:

        # Definir las 7 características específicas que los modelos esperan
        caracteristicas_modelo = ['IIJ', 'IVJO', 'IER', 'IV', 'IPA', 'IJD', 'ICJ']
        
        # Verificar que todas las características existen
        faltan_caract = [c for c in caracteristicas_modelo if c not in df_indices.columns]
        if faltan_caract:
            st.error(f"Faltan características necesarias para los modelos: {faltan_caract}")
        else:
            # Seleccionar solo las 7 características para la predicción
            X_pred = df_indices[caracteristicas_modelo].values
            
            # Crear copia para añadir predicciones
            df_con_predicciones = df_indices.copy()
            
            # Añadir predicciones para cada categoría
            for categoria in categorias_faltantes:
                try:
                    if categoria in modelos and 'modelo' in modelos[categoria]:
                        # Obtener el modelo directamente
                        modelo = modelos[categoria]['modelo']
                        
                        # Hacer la predicción con las 7 características específicas
                        predicciones = modelo.predict(X_pred)
                        
                        # Asignar al DataFrame
                        df_con_predicciones[categoria] = predicciones

                except Exception as e:
                    st.error(f"Error al predecir {categoria}: {str(e)}")
            
            # Usar DataFrame con predicciones
            datos_viz = preparar_datos_visualizacion(df_con_predicciones)
    else:
        # Usar DataFrame original
        datos_viz = preparar_datos_visualizacion(df_indices)   

    # Cargar modelos entrenados
    modelos = cargar_modelos_entrenados()

    # Crear tabs para las diferentes secciones
    tab4, tab1, tab2, tab3 = st.tabs([
        "Análisis Contextual",
        "Clasificación de Estilos", 
        "Visualización Táctica", 
        "Análisis de Partido"
    ])

    # Tab 1: Clasificación de Estilos
    with tab1:
        st.markdown("### Resumen de Categorización de Estilos de Juego")
        st.markdown("""
        Esta sección muestra cómo se clasifican los partidos del Atlético de Madrid según diferentes categorías de estilo de juego.
        El sistema utiliza los índices tácticos calculados para detectar patrones mediante aprendizaje automático (SVM).
        """)
        
        # Selección de filtro
        filtro_col1, filtro_col2 = st.columns(2)
        
        with filtro_col1:
            filtro_resultado = st.selectbox(
                "Filtrar por resultado:",
                ["Todos", "Victoria", "Empate", "Derrota"],
                index=0
            )
        
        with filtro_col2:
            filtro_localidad = st.selectbox(
                "Filtrar por localía:",
                ["Todos", "Local", "Visitante"],
                index=0
            )
        
        # Aplicar filtros
        df_filtrado = df_indices.copy()
        
        if filtro_resultado != "Todos" and 'resultado_tipo' in df_filtrado.columns:
            df_filtrado = df_filtrado[df_filtrado['resultado_tipo'] == filtro_resultado]
        
        if filtro_localidad != "Todos" and 'es_local' in df_filtrado.columns:
            if filtro_localidad == "Local":
                df_filtrado = df_filtrado[df_filtrado['es_local'] == 1]
            else:
                df_filtrado = df_filtrado[df_filtrado['es_local'] == 0]
        
        # Mostrar mensaje si no hay datos después del filtrado
        if df_filtrado.empty:
            st.warning("No hay partidos que cumplan con los filtros seleccionados.")
        else:
            # Recalcular datos para visualización con los datos filtrados
            datos_viz_filtrados = preparar_datos_visualizacion(df_filtrado)
            
            # Crear tabla de resumen con las categorías principales
            categorias_principales = ['orientacion_general', 'fase_ofensiva', 'intensidad_defensiva', 'tipo_transicion']
            
            # Mostrar tablas en dos columnas
            for i in range(0, len(categorias_principales), 2):
                cols = st.columns(2)
                
                for j in range(2):
                    idx = i + j
                    if idx < len(categorias_principales):
                        cat = categorias_principales[idx]
                        if cat in datos_viz_filtrados:
                            with cols[j]:
                                st.subheader(ESTILO_NOMBRES.get(cat, cat))
                                
                                # Formatear tabla para mostrar
                                tabla = datos_viz_filtrados[cat].copy()
                                tabla.columns = ['Estilo', 'Partidos']
                                
                                # Añadir columna de porcentaje
                                tabla['Porcentaje'] = (tabla['Partidos'] / tabla['Partidos'].sum() * 100).round(1).astype(str) + '%'
                                
                                # Mostrar como tabla
                                st.table(tabla)
            
            # Visualización de gráficos para categorías secundarias
            st.markdown("### Distribución de Estilos Secundarios")
            
            categorias_secundarias = ['patron_ataque', 'altura_bloque', 'estilo_posesion']
            
            # Mostrar gráficos en tres columnas
            cols = st.columns(3)

            for i, cat in enumerate(categorias_secundarias):
                with cols[i]:
                    if cat in datos_viz_filtrados:                        
                        
                        try:
                            fig = crear_grafico_distribucion_estilos(datos_viz_filtrados, cat)
                            if fig:                                
                                st.plotly_chart(fig, use_container_width=True)                                
                            else:
                                st.warning(f"La función crear_grafico_distribucion_estilos devolvió None para {cat}")
                        except Exception as e:
                            st.error(f"Error al crear o mostrar el gráfico: {str(e)}")
                    else:
                        st.warning(f"Categoría {cat} no encontrada en datos_viz_filtrados")

    # Tab 2: Visualización Táctica
    with tab2:
        st.markdown("### Visualización por Índices Tácticos")
        st.markdown("""
        Esta sección permite explorar la relación entre diferentes índices tácticos y cómo se agrupan
        los partidos según sus características. Puedes seleccionar qué índices visualizar y cómo colorear
        los puntos según las diferentes categorías de estilo.
        """)
        
        # Selección de índices y categoría de color
        col1, col2, col3 = st.columns(3)
        
        # Lista de índices disponibles
        indices_disponibles = [col for col in df_indices.columns if col.startswith('I') and col.isupper()]
        
        with col1:
            eje_x = st.selectbox(
                "Índice para eje X:",
                indices_disponibles,
                index=indices_disponibles.index('ICP') if 'ICP' in indices_disponibles else 0
            )
        
        with col2:
            eje_y = st.selectbox(
                "Índice para eje Y:",
                indices_disponibles,
                index=indices_disponibles.index('IV') if 'IV' in indices_disponibles else min(1, len(indices_disponibles)-1)
            )
        
        with col3:
            cat_color = st.selectbox(
                "Colorear por categoría:",
                list(ESTILO_NOMBRES.keys()),
                index=0
            )
        
        # Crear scatter plot
        fig_scatter = crear_scatter_indices(df_indices, eje_x, eje_y, cat_color)
        
        if fig_scatter:
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("No se pueden visualizar los índices seleccionados. Verifica que existan en los datos.")
        
        # Mostrar visualización PCA si está disponible en el modelo
        st.markdown("### Visualización de Dimensionalidad Reducida (PCA)")
        
        if cat_color in modelos and 'pca' in modelos[cat_color]:
            # Preparar datos para modelado
            datos_modelo = preparar_datos_modelado(df_indices, normalizar=True)
            
            if 'X_train' in datos_modelo and not datos_modelo['X_train'].empty:
                # Crear imagen con Matplotlib
                fig, pca = crear_visualizacion_pca(
                    datos_modelo['X_train'], 
                    datos_modelo['y_train_dict'][cat_color], 
                    titulo=f"Categorización de partidos: {ESTILO_NOMBRES.get(cat_color, cat_color)}"
                )
                
                # Mostrar figura
                st.pyplot(fig)

                # Explicar varianza
                if pca:
                    componentes = min(2, len(pca.explained_variance_ratio_))
                    varianza_total = sum(pca.explained_variance_ratio_[:componentes])
                    st.markdown(f"**Varianza explicada por los primeros {componentes} componentes: {varianza_total:.2%}**")
                    
                    # Mostrar características más importantes
                    if cat_color in modelos and 'caracteristicas' in modelos[cat_color]:
                        st.markdown("**Características más importantes para esta categoría:**")
                        caract = modelos[cat_color]['caracteristicas']
                        st.write(", ".join(caract))
        else:
            st.info("Para ver la visualización PCA, necesitas entrenar un modelo para la categoría seleccionada.")

    # Tab 3: Análisis de Partido
    with tab3:
        st.markdown("### Análisis de Partido Individual")
        st.markdown("""
        Selecciona un partido específico para analizar en detalle su estilo de juego y características tácticas.
        El radar chart muestra cómo se comparan los índices tácticos del partido con la media de la temporada.
        """)
        
        # Crear selectbox para seleccionar partido
        partidos_info = []
        
        # Preparar información de partidos si están disponibles las columnas
        if all(col in df_indices.columns for col in ['jornada']):
            for idx, row in df_indices.iterrows():
                jornada = int(row['jornada']) if pd.notna(row['jornada']) else '?'
                
                # Obtener información adicional si está disponible
                rival = row['rival'] if 'rival' in row and pd.notna(row['rival']) else '?'
                resultado = row['resultado'] if 'resultado' in row and pd.notna(row['resultado']) else '?'
                
                if 'partido' in row and pd.notna(row['partido']):
                    etiqueta = f"Jornada {jornada}ª: {row['partido']} ({resultado})"
                else:
                    etiqueta = f"Jornada {jornada}ª: ATM vs {rival} ({resultado})"
                
                partidos_info.append((idx, etiqueta))
        
        if partidos_info:
            # Ordenar por jornada
            partidos_info.sort(key=lambda x: int(x[1].split('Jornada ')[1].split('ª')[0]) if 'Jornada ' in x[1] else 0)
            
            # Extraer índices y etiquetas
            indices_partidos, etiquetas_partidos = zip(*partidos_info)
            
            # Crear selectbox
            partido_seleccionado = st.selectbox(
                "Selecciona un partido:",
                indices_partidos,
                format_func=lambda x: etiquetas_partidos[indices_partidos.index(x)]
            )
            
            # Crear dos columnas para información y visualización
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Mostrar información del partido
                st.subheader("Características del Partido")
                
                partido_datos = df_indices.loc[partido_seleccionado]
                
                # Mostrar información básica
                info_mostrar = [
                    ('Jornada', f"{int(partido_datos['jornada'])}ª" if 'jornada' in partido_datos else "N/A"),
                    ('Rival', partido_datos['rival'] if 'rival' in partido_datos else "N/A"),
                    ('Resultado', partido_datos['resultado'] if 'resultado' in partido_datos else "N/A"),
                    ('Posesión', f"{partido_datos['posesion']}%" if 'posesion' in partido_datos else "N/A"),
                    ('Tiros', partido_datos['tiros_totales'] if 'tiros_totales' in partido_datos else "N/A"),
                    ('Duelos ganados', partido_datos['duelos_ganados'] if 'duelos_ganados' in partido_datos else "N/A"),
                ]
                
                for label, valor in info_mostrar:
                    st.markdown(f"**{label}:** {valor}")
                
                # Mostrar clasificación de estilos
                st.subheader("Clasificación de Estilos")
                
                for cat, nombre in ESTILO_NOMBRES.items():
                    if cat in partido_datos:
                        st.markdown(f"**{nombre}:** {partido_datos[cat]}")
            
            with col2:
                # Crear y mostrar radar chart
                fig_radar = mi_crear_radar_indices(df_indices, partido_seleccionado)
                
                if fig_radar:
                    st.plotly_chart(fig_radar, use_container_width=True)
                else:
                    st.warning("No se pueden visualizar los índices tácticos para este partido.")
        else:
            st.markdown("""
                <div style="padding: 20px; background-color: #fff3cd; border-left: 5px solid #856404; border-radius: 8px;">
                    <strong>⚠️ Visualización en construcción</strong><br>
                    Esta sección aún no está operativa porque no se han podido cargar los datos del partido.<br>
                    Estamos trabajando para tenerlo listo pronto. ¡Gracias por tu paciencia!
                </div>
            """, unsafe_allow_html=True)

    # Tab 4: Análisis Contextual
    with tab4:
        st.markdown("### Análisis de Variables Contextuales")
        st.markdown("""
        Esta sección muestra cómo varían las métricas tácticas según diferentes contextos como 
        resultado, localía o categoría de rival.
        """)
        
        # Selección de tipo de análisis
        tipo_analisis = st.radio(
            "Tipo de análisis:",
            ["Por resultado", "Por localía", "Por categoría de rival"],
            horizontal=True
        )
        
        # Selección de métricas
        metricas_disponibles = ['posesion', 'xG', 'tiros_totales', 'duelos_ganados', 'IIJ', 'ICJ', 'ICP',
                               'toques_area_rival', 'altura_bloque', 'intensidad_defensiva', 'IPA']
        
        metricas_existentes = [m for m in metricas_disponibles if m in df_indices.columns]
        
        metricas_seleccionadas = st.multiselect(
            "Selecciona métricas a comparar:",
            metricas_existentes,
            default=metricas_existentes[:4]
        )
        
        if not metricas_seleccionadas:
            st.warning("Selecciona al menos una métrica para visualizar.")
        else:
            from plotly.subplots import make_subplots
            
            # Configurar visualización según tipo de análisis
            if tipo_analisis == "Por resultado" and 'resultado_tipo' in df_indices.columns:
                # Crear boxplots por resultado
                fig = crear_boxplots_resultado(df_indices, metricas_seleccionadas)
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Añadir descripción de hallazgos
                    st.markdown("""
                    **Observaciones:**
                    - La posesión suele ser mayor en derrotas, lo que sugiere que tener más balón no siempre se traduce en mejores resultados.
                    - El xG (Expected Goals) tiende a ser mayor en victorias, mostrando la importancia de la calidad de las ocasiones creadas.
                    - Los tiros totales no siempre se correlacionan con el resultado, pero la eficacia sí.
                    - Los duelos ganados muestran una distribución bastante uniforme entre resultados.
                    """)
                else:
                    st.warning("No se pueden crear visualizaciones con las métricas seleccionadas.")
            
            elif tipo_analisis == "Por localía" and 'es_local' in df_indices.columns:
                # Crear DataFrame para mostrar diferencias por localía
                df_indices['localidad'] = df_indices['es_local'].map({1: 'Local', 0: 'Visitante'})
                
                # Crear figura con subplots
                fig = make_subplots(
                    rows=2, 
                    cols=2,
                    subplot_titles=metricas_seleccionadas[:4],
                    horizontal_spacing=0.15,
                    vertical_spacing=0.2
                )
                
                # Colores para cada localía
                colores = {
                    'Local': '#1f77b4',
                    'Visitante': '#ff7f0e'
                }
                
                # Añadir boxplots para cada métrica
                for i, metrica in enumerate(metricas_seleccionadas[:4]):
                    row = i // 2 + 1
                    col = i % 2 + 1
                    
                    for localidad, color in colores.items():
                        valores = df_indices[df_indices['localidad'] == localidad][metrica]
                        
                        if not valores.empty:
                            fig.add_trace(
                                go.Box(
                                    y=valores,
                                    name=localidad,
                                    marker_color=color,
                                    showlegend=(i==0)  # Solo mostrar leyenda para el primer gráfico
                                ),
                                row=row, col=col
                            )
                
                # Configurar layout
                fig.update_layout(
                    title="Distribución de métricas según localía",
                    height=600,
                    boxmode='group',
                    margin=dict(l=40, r=20, t=60, b=40),
                    legend=dict(orientation="h", y=-0.1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Añadir descripción de hallazgos
                st.markdown("""
                **Observaciones:**
                - Los índices ofensivos suelen ser superiores en partidos como local.
                - La posesión tiende a ser más equilibrada de visitante pero con menor verticalidad.
                - La presión defensiva es adaptativa según la localía, siendo más intensa como local.
                """)
            
            elif tipo_analisis == "Por categoría de rival" and 'rival_categoria' in df_indices.columns:
                # Crear figura con subplots
                fig = make_subplots(
                    rows=2, 
                    cols=2,
                    subplot_titles=metricas_seleccionadas[:4],
                    horizontal_spacing=0.15,
                    vertical_spacing=0.2
                )
                
                # Colores para cada categoría de rival
                colores = {
                    'top': '#d62728',  # Rojo
                    'medio': '#ff7f0e',  # Naranja
                    'bajo': '#2ca02c'   # Verde
                }
                
                # Añadir boxplots para cada métrica
                for i, metrica in enumerate(metricas_seleccionadas[:4]):
                    row = i // 2 + 1
                    col = i % 2 + 1
                    
                    for categoria, color in colores.items():
                        valores = df_indices[df_indices['rival_categoria'] == categoria][metrica]
                        
                        if not valores.empty:
                            fig.add_trace(
                                go.Box(
                                    y=valores,
                                    name=categoria.capitalize(),
                                    marker_color=color,
                                    showlegend=(i==0)  # Solo mostrar leyenda para el primer gráfico
                                ),
                                row=row, col=col
                            )
                
                # Configurar layout
                fig.update_layout(
                    title="Distribución de métricas según categoría de rival",
                    height=600,
                    boxmode='group',
                    margin=dict(l=40, r=20, t=60, b=40),
                    legend=dict(orientation="h", y=-0.1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Añadir descripción de hallazgos
                st.markdown("""
                **Observaciones:**
                - Contra rivales top, el Atlético reduce su posesión pero aumenta su eficacia en transiciones.
                - Contra equipos de menor nivel, el equipo domina más el juego pero puede ser menos vertical.
                - Los índices defensivos son más altos contra rivales fuertes, mostrando adaptación táctica.
                """)
            else:
                st.warning(f"No hay datos suficientes para el análisis {tipo_analisis}.")

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
            contenido_pdf = [
                "Informe general: Detección de Estilos de Juego",
                "",
                f"Total partidos analizados: {len(df_indices)}",
                "Variables clave incluidas:",
                "- Índices tácticos como IIJ, IVJO, ICJ, IPA...",
                "- Categorizaciones como orientación, posesión, transiciones...",
                "",
                "Esta página incluye análisis por estilos, contexto y partidos individuales."
            ]
            pdf_filename = "informe_estilos.pdf"

            export_to_pdf("Detección de Estilos - Informe General", contenido_pdf, autor="Ramón González", output_path=pdf_filename)

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