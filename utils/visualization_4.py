import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Any, Optional

# Importamos las funciones del módulo feature_enginering
from models.style_prediction.feature_enginering import create_features, load_data, clean_data

# Funciones para la carga y procesamiento de datos para análisis de estilos
def cargar_datos_para_analisis(master_path, stats_path, eventos_dir, equipos_path):
    """
    Carga y procesa los datos necesarios para el análisis de estilos de juego.
    Utiliza las funciones del módulo feature_engineering.
    
    Args:
        master_path (str): Ruta al archivo master_liga_vert_atlmed.csv
        stats_path (str): Ruta al archivo stats_atm_por_partido_pag4.csv
        eventos_dir (str): Ruta al directorio con archivos de eventos (.parquet)
        equipos_path (str): Ruta al archivo de equipos
        
    Returns:
        tuple: (df_indices, data_clean) - DataFrame con índices tácticos y datos limpios
    """
    try:
        
        
        # Cargar datos
        data_raw = load_data(master_path, stats_path, eventos_dir, equipos_path)
        
        # Limpiar datos
        data_clean = clean_data(data_raw)
        
        # Crear features para análisis
        df_indices = create_features(data_clean)
        
        return df_indices, data_clean
    except Exception as e:
        print(f"Error al cargar datos para análisis: {e}")
        return pd.DataFrame(), {}

# Diccionario para mapeo de nombres de estilos
ESTILO_NOMBRES = {
    'orientacion_general': 'Orientación General',
    'fase_ofensiva': 'Fase Ofensiva',
    'patron_ataque': 'Patrón de Ataque',
    'intensidad_defensiva': 'Intensidad Defensiva',
    'altura_bloque': 'Altura del Bloque',
    'tipo_transicion': 'Tipo de Transición',
    'estilo_posesion': 'Estilo de Posesión'
}

# Colores por categoría
COLORES_CATEGORIAS = {
    'orientacion_general': {
        'Ofensivo de calidad': '#1f77b4',
        'Ofensivo en cantidad': '#2ca02c',
        'Equilibrado': '#9467bd',
        'Defensivo activo': '#d62728',
        'Defensivo pasivo': '#ff7f0e',
        'Ofensivo': '#1f77b4',  # Alternativo simple
        'Defensivo': '#d62728'  # Alternativo simple
    },
    'fase_ofensiva': {
        'Posicional': '#1f77b4',
        'Vertical-Preciso': '#2ca02c',
        'Vertical': '#9467bd',
        'Directo-Efectivo': '#d62728',
        'Directo': '#ff7f0e'
    },
    'patron_ataque': {
        'Equilibrio pasillos de ataque': '#1f77b4',
        'Enfoque pasillos exteriores': '#2ca02c',
        'Enfoque pasillo central': '#d62728',
        'Equilibrado': '#9467bd'  # Alternativo
    },
    'intensidad_defensiva': {
        'Cierre de trayectorias en altura': '#1f77b4',
        'Presión alta agresiva': '#2ca02c',
        'Moderada': '#9467bd',
        'Defensa pasiva organizada': '#d62728',
        'Defensa pasiva reactiva': '#ff7f0e',
        'Presión alta': '#2ca02c',  # Alternativo simple
        'Defensa pasiva': '#ff7f0e'  # Alternativo simple
    },
    'altura_bloque': {
        'Alto': '#1f77b4',
        'Medio': '#9467bd',
        'Bajo': '#d62728'
    },
    'tipo_transicion': {
        'Elaborada efectiva': '#1f77b4',
        'Elaborada ineficiente': '#2ca02c',
        'Convencional': '#9467bd',
        'Directa efectiva': '#d62728',
        'Directa ineficiente': '#ff7f0e',
        'Contragolpe adaptativo': '#8c564b',
        'Elaborada': '#1f77b4',  # Alternativo simple
        'Directa': '#d62728'     # Alternativo simple
    },
    'estilo_posesion': {
        'Posesión dominante': '#1f77b4',
        'Posesión funcional': '#2ca02c',
        'Posesión estéril': '#d62728',
        'Posesión eficiente': '#9467bd'
    }
}

def preparar_datos_visualizacion(df_indices: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Prepara los datos para la visualización en la aplicación.
    
    Args:
        df_indices (pd.DataFrame): DataFrame con índices tácticos y categorías
    
    Returns:
        dict: Diccionario con datos preparados para visualización
    """
    # Verificar si hay datos
    if df_indices.empty:
        return {}
    
    # Crear diccionario para almacenar resultados
    datos_viz = {}
    
    # Para cada categoría, obtener conteo
    categorias = ['orientacion_general', 'fase_ofensiva', 'patron_ataque',
                 'intensidad_defensiva', 'altura_bloque', 'tipo_transicion',
                 'estilo_posesion']
    
    for cat in categorias:
        if cat in df_indices.columns:
            # Conteo de valores
            conteo = df_indices[cat].value_counts().reset_index()
            conteo.columns = ['estilo', 'partidos']
            
            # Ordenar por número de partidos (descendente)
            conteo = conteo.sort_values('partidos', ascending=False)
            
            # Guardar en diccionario
            datos_viz[cat] = conteo
    
    # Añadir datos para análisis por resultado
    if 'resultado_tipo' in df_indices.columns:
        # Convertir resultado_tipo a categórica para ordenar
        df_indices['resultado_tipo'] = pd.Categorical(
            df_indices['resultado_tipo'], 
            categories=['Victoria', 'Empate', 'Derrota'],
            ordered=True
        )
        
        # Para cada categoría, crear tabla por resultado
        for cat in categorias:
            if cat in df_indices.columns:
                tabla_resultado = pd.crosstab(
                    df_indices[cat], 
                    df_indices['resultado_tipo']
                ).reset_index()
                
                # Calcular porcentajes - MODIFICADO: Solo sumar columnas numéricas
                columnas_numericas = [col for col in tabla_resultado.columns if col != cat]
                tabla_resultado['Total'] = tabla_resultado[columnas_numericas].sum(axis=1)
                
                for resultado in ['Victoria', 'Empate', 'Derrota']:
                    if resultado in tabla_resultado.columns:
                        tabla_resultado[f'Pct_{resultado}'] = (tabla_resultado[resultado] / tabla_resultado['Total'] * 100).round(1)
                
                # Guardar en diccionario
                datos_viz[f"{cat}_resultado"] = tabla_resultado
    
    # Datos para análisis por localía
    if 'es_local' in df_indices.columns:
        # Convertir a categórica para mejor visualización
        df_indices['localidad'] = df_indices['es_local'].map({1: 'Local', 0: 'Visitante'})
        
        # Para cada categoría, crear tabla por localía
        for cat in categorias:
            if cat in df_indices.columns:
                tabla_localidad = pd.crosstab(
                    df_indices[cat], 
                    df_indices['localidad']
                ).reset_index()
                
                # Calcular porcentajes - MODIFICADO: Solo sumar columnas numéricas
                columnas_numericas = [col for col in tabla_localidad.columns if col != cat]
                tabla_localidad['Total'] = tabla_localidad[columnas_numericas].sum(axis=1)
                
                for localidad in ['Local', 'Visitante']:
                    if localidad in tabla_localidad.columns:
                        tabla_localidad[f'Pct_{localidad}'] = (tabla_localidad[localidad] / tabla_localidad['Total'] * 100).round(1)
                
                # Guardar en diccionario
                datos_viz[f"{cat}_localidad"] = tabla_localidad
    
    return datos_viz

def crear_grafico_distribucion_estilos(datos: Dict[str, pd.DataFrame], categoria: str) -> Optional[go.Figure]:
    """
    Crea un gráfico de barras para mostrar la distribución de estilos.
    
    Args:
        datos (dict): Diccionario con datos de visualización
        categoria (str): Categoría de estilo a visualizar
    
    Returns:
        go.Figure: Figura de Plotly
    """
    if categoria not in datos:
        return None
    
    df = datos[categoria]
    
    # Colores para el gráfico
    colores = list(COLORES_CATEGORIAS.get(categoria, {}).values())
    if not colores or len(colores) < len(df):
        colores = px.colors.qualitative.Plotly[:len(df)]
    
    # Crear figura
    fig = go.Figure()
    
    # Añadir barras
    fig.add_trace(go.Bar(
        x=df['estilo'],
        y=df['partidos'],
        text=df['partidos'],
        textposition='auto',
        marker_color=colores[:len(df)],
        name=''
    ))
    
    # Configurar layout
    fig.update_layout(
        title=f"Distribución de {ESTILO_NOMBRES.get(categoria, categoria)}",
        xaxis_title="Estilo",
        yaxis_title="Número de partidos",
        template="plotly_white",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    return fig

def crear_scatter_indices(df_indices: pd.DataFrame, x_col: str, y_col: str, color_col: str) -> Optional[go.Figure]:
    """
    Crea un scatter plot donde los ejes son índices tácticos y el color representa
    una categoría de estilo.
    
    Args:
        df_indices (pd.DataFrame): DataFrame con índices tácticos
        x_col (str): Columna para eje X
        y_col (str): Columna para eje Y
        color_col (str): Columna para color de puntos
    
    Returns:
        go.Figure: Figura de Plotly
    """
    # Verificar que existan las columnas
    if not all(col in df_indices.columns for col in [x_col, y_col, color_col]):
        return None
    
    # Crear etiquetas para los puntos
    df_scatter = df_indices.copy()
    
    # Añadir etiqueta de partido si es posible
    if all(col in df_scatter.columns for col in ['jornada', 'partido']):
        df_scatter['etiqueta'] = df_scatter.apply(
            lambda row: f"{int(row['jornada'])}ª_{row['partido']}" if pd.notna(row['partido']) else f"{int(row['jornada'])}ª",
            axis=1
        )
    elif 'jornada' in df_scatter.columns:
        df_scatter['etiqueta'] = df_scatter['jornada'].apply(lambda x: f"{int(x)}ª")
    else:
        df_scatter['etiqueta'] = range(len(df_scatter))
    
    # Colores por categoría
    colores = COLORES_CATEGORIAS.get(color_col, {})
    
    # Crear figura con Plotly Express
    fig = px.scatter(
        df_scatter,
        x=x_col,
        y=y_col,
        color=color_col,
        color_discrete_map=colores,
        hover_name="etiqueta",
        hover_data=[color_col, 'resultado_tipo'] if 'resultado_tipo' in df_scatter.columns else [color_col],
        size_max=10,
        opacity=0.8,
        title=f"Categorización de partidos: {color_col}"
    )
    
    # Ajustar layout
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        legend_title=ESTILO_NOMBRES.get(color_col, color_col),
        height=600,
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    # Actualizar trazos
    fig.update_traces(
        marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')),
    )
    
    return fig

def crear_radar_indices(df_indices: pd.DataFrame, partido_idx: Any) -> Optional[go.Figure]:
    """
    Crea un gráfico radar para visualizar los índices tácticos de un partido específico.
    
    Args:
        df_indices (pd.DataFrame): DataFrame con índices tácticos
        partido_idx: Índice o identificador del partido a visualizar
    
    Returns:
        go.Figure: Figura de Plotly
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
    
    # Filtrar índices que existen en los datos
    indices_disponibles = [idx for idx in indices_tacticos if idx in partido.index]
    
    if not indices_disponibles:
        return None
    
    # Extraer valores
    valores = [partido[idx] for idx in indices_disponibles]
    
    # Calcular media de cada índice para la temporada
    medias = [df_indices[idx].mean() for idx in indices_disponibles]
    
    # Crear figura
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

def crear_boxplots_resultado(df_indices: pd.DataFrame, metricas: List[str]) -> Optional[go.Figure]:
    """
    Crea gráficos de boxplot para mostrar la distribución de métricas según resultado.
    
    Args:
        df_indices (pd.DataFrame): DataFrame con índices y resultados
        metricas (list): Lista de métricas a visualizar
    
    Returns:
        go.Figure: Figura de Plotly
    """
    # Verificar que existan las columnas
    if 'resultado_tipo' not in df_indices.columns:
        return None
    
    metricas_disponibles = [m for m in metricas if m in df_indices.columns]
    if not metricas_disponibles:
        return None
    
    # Convertir resultado_tipo a categórica para ordenar
    df_plot = df_indices.copy()
    df_plot['resultado_tipo'] = pd.Categorical(
        df_plot['resultado_tipo'], 
        categories=['Victoria', 'Empate', 'Derrota'],
        ordered=True
    )
    
    # Crear figura con subplots
    fig = make_subplots(
        rows=2, 
        cols=2,
        subplot_titles=metricas_disponibles[:4],
        horizontal_spacing=0.15,
        vertical_spacing=0.2
    )
    
    # Colores para cada resultado
    colores = {
        'Victoria': '#2ca02c',
        'Empate': '#ff7f0e',
        'Derrota': '#d62728'
    }
    
    # Añadir boxplots para cada métrica
    for i, metrica in enumerate(metricas_disponibles[:4]):
        row = i // 2 + 1
        col = i % 2 + 1
        
        for resultado, color in colores.items():
            valores = df_plot[df_plot['resultado_tipo'] == resultado][metrica]
            
            if not valores.empty:
                fig.add_trace(
                    go.Box(
                        y=valores,
                        name=resultado,
                        marker_color=color,
                        showlegend=(i==0)  # Solo mostrar leyenda para el primer gráfico
                    ),
                    row=row, col=col
                )
    
    # Configurar layout
    fig.update_layout(
        title="Distribución de métricas según resultado",
        height=600,
        boxmode='group',
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", y=-0.1)
    )
    
    return fig