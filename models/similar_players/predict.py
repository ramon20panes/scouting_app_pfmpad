import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys
import unicodedata
from sklearn.metrics.pairwise import euclidean_distances
from utils.export_pdf import dataframe_a_pdf_contenido
from . import train_model

# Directorio donde se encuentran los modelos
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# Cargar recursos necesarios
def cargar_recursos():
    """
    Carga los modelos y datos necesarios para las predicciones
    
    Returns:
        dict: Diccionario con los recursos cargados
    """
    recursos = {}
    
    try:
        # Cargar datos de jugadores del Atlético
        recursos['df_atletico'] = joblib.load(os.path.join(MODEL_DIR, 'df_atletico.joblib'))
        
        # Cargar tabla de alias
        recursos['alias_mapping'] = joblib.load(os.path.join(MODEL_DIR, 'alias_mapping.joblib'))

        # Añadir alias manuales si no están en el mapping original
        correcciones_manual = {
            "Jorge Resurrección": "Koke",
            "Koke": "Jorge Resurrección",  # Usa uno como nombre oficial
            "Alexander Sorloth": "Alexander Sørloth",
            "Aleksander Sorloth": "Alexander Sørloth",
            "Sorloth": "Alexander Sørloth"
        }

        # Actualizar el mapping cargado
        recursos['alias_mapping'].update(correcciones_manual)
        
        # Cargar clusters
        recursos['resultado_clusters'] = joblib.load(os.path.join(MODEL_DIR, 'resultado_clusters.joblib'))
        
        # Cargar datos procesados
        recursos['datos_procesados'] = joblib.load(os.path.join(MODEL_DIR, 'datos_procesados.joblib'))
        
        # Cargar modelos KNN para cada posición
        recursos['modelos_knn'] = {}
        recursos['scalers'] = {}
        
        for pos in ['GK', 'DF', 'MF', 'FW']:
            try:
                recursos['modelos_knn'][pos] = joblib.load(os.path.join(MODEL_DIR, f'knn_{pos}.joblib'))
                recursos['scalers'][pos] = joblib.load(os.path.join(MODEL_DIR, f'scaler_{pos}.joblib'))
            except:
                print(f"No se encontró modelo para posición {pos}")
        
        return recursos
    
    except Exception as e:
        print(f"Error al cargar recursos: {e}")
        
        # Si hay un error, intentar entrenar los modelos automáticamente
        try:
            print("Intentando entrenar modelos automáticamente...")
                      
            # Entrenar modelos
            resultado_clusters, modelos_knn, datos_procesados = train_model.main()
            
            # Intentar cargar recursos nuevamente
            return cargar_recursos()
        except Exception as e2:
            print(f"Error al entrenar modelos automáticamente: {e2}")
            return {}

def obtener_jugadores_atletico():
    """
    Obtiene la lista de jugadores del Atlético de Madrid
    
    Returns:
        DataFrame: DataFrame con información de los jugadores
    """
    try:
        df_atletico = joblib.load(os.path.join(MODEL_DIR, 'df_atletico.joblib'))
        return df_atletico
    except:
        print("No se encontraron datos de jugadores del Atlético")
        return pd.DataFrame()

def aplicar_alias(nombre, alias_mapping):
    """
    Aplica la tabla de alias para normalizar nombres de jugadores
    
    Args:
        nombre: Nombre del jugador
        alias_mapping: Diccionario con mapeo de alias
        
    Returns:
        str: Nombre normalizado
    """
    return alias_mapping.get(nombre, nombre)

def normalizar_texto(texto):
    """
    Normaliza texto (quita acentos, pasa a minúsculas)
    
    Args:
        texto: Texto a normalizar
        
    Returns:
        str: Texto normalizado
    """
    if isinstance(texto, str):
        texto = texto.strip().lower()
        texto = unicodedata.normalize('NFKD', texto)
        return ''.join(c for c in texto if not unicodedata.combining(c))
    return texto

def obtener_jugadores_similares(jugador, recursos, top_n=8, include_attrs=None):
    """
    Encuentra jugadores similares al jugador especificado
    
    Args:
        jugador: Nombre del jugador
        recursos: Diccionario con los recursos cargados
        top_n: Número de jugadores similares a devolver
        include_attrs: Lista opcional de atributos adicionales a incluir
        
    Returns:
        dict: Diccionario con resultados e información para visualización
    """
    # Verificar que se han cargado los recursos
    if not recursos:
        print("No se han cargado los recursos necesarios")
        return None
    
    # Acceso directo a recursos
    df_atletico = recursos['df_atletico']
    alias_mapping = recursos['alias_mapping']
    resultado_clusters = recursos['resultado_clusters']
    datos_procesados = recursos['datos_procesados']
    modelos_knn = recursos['modelos_knn']
    scalers = recursos['scalers']
    
    # Normalizar nombre del jugador
    jugador_real = aplicar_alias(jugador, alias_mapping)
    
    # Verificar que el jugador existe en la plantilla
    if jugador_real not in df_atletico['Nombre'].values:
        print(f"El jugador '{jugador}' no está en la plantilla del Atlético")
        return None
    
    # Obtener posición del jugador
    posicion = df_atletico.loc[df_atletico['Nombre'] == jugador_real, 'Posicion'].values[0]
    
    # Verificar posición
    if posicion not in ['GK', 'DF', 'MF', 'FW']:
        print(f"Posición '{posicion}' no reconocida")
        return None
    
    # Verificar que existen datos para esta posición
    if posicion not in datos_procesados:
        print(f"No hay datos procesados para la posición {posicion}")
        return None
    
    # Obtener datos del modelo
    df_pos = datos_procesados[posicion]['df'].copy()
    feature_cols = datos_procesados[posicion]['feature_cols']
    
    # Normalizar nombres
    jugador_norm = normalizar_texto(jugador_real)
    
    # Verificar que el jugador está en los datos
    if jugador_norm not in df_pos['Nombre'].values:
        print(f"El jugador '{jugador_real}' no está en los datos de la posición {posicion}")
        return None
    
    # Obtener el cluster del jugador
    cluster_id = df_pos.loc[df_pos['Nombre'] == jugador_norm, 'cluster'].values[0]
    
    # Filtrar jugadores del mismo cluster y excluir al Atlético
    df_cluster = df_pos[df_pos['cluster'] == cluster_id].copy()
    df_cluster = df_cluster[~df_cluster['Equipo'].str.contains('atletico', case=False, na=False)]
    
    if df_cluster.empty:
        print("No hay jugadores en el mismo cluster fuera del Atlético")
        return None
    
    # Obtener el índice del jugador
    idx_jugador = df_pos[df_pos['Nombre'] == jugador_norm].index[0]
    
    # Obtener características del jugador
    X_jugador = df_pos.loc[idx_jugador, feature_cols].values.reshape(1, -1)
    
    # Escalar características
    X_scaled = scalers[posicion].transform(X_jugador)
    
    # Aplicar KNN
    if posicion in modelos_knn:
        # Usar el modelo KNN pre-entrenado
        distancias, indices = modelos_knn[posicion].kneighbors(X_scaled)
        
        # El primer índice es el propio jugador, tomamos los siguientes top_n
        indices = indices[0][1:top_n+1]
        distancias = distancias[0][1:top_n+1]
        
        # Obtener los jugadores similares
        similares = df_pos.iloc[indices].copy()
        similares['distancia'] = distancias
    else:
        # Si no hay modelo KNN, usar distancia euclidiana directamente
        cluster_values = df_cluster[feature_cols].values
        distancias = euclidean_distances(X_jugador, cluster_values)[0]
        df_cluster['distancia'] = distancias
        similares = df_cluster.sort_values('distancia').head(top_n).copy()
    
    # Excluir cualquier jugador del Atlético en la lista de similares
    similares = similares[~similares['Equipo'].str.contains('atletico', case=False, na=False)]
    
    # Ordenar por distancia
    similares = similares.sort_values('distancia').head(top_n)
    
    # Definir columnas de resultado según posición
    columnas_resultado = ['Nombre', 'Equipo', 'Contrato', 'Nacimiento', 'distancia']
    
    # Añadir atributos adicionales si se solicitaron
    if include_attrs:
        for attr in include_attrs:
            if attr in similares.columns:
                columnas_resultado.append(attr)
    
    # Añadir métricas específicas por posición
    if posicion == 'GK':
        metricas = ['Eficiencia_Paradas_original', 'Porcentaje_Porterias_Cero_original']
        ejes = {'x': 'Eficiencia_Paradas_original', 'y': 'Porcentaje_Porterias_Cero_original'}
    elif posicion == 'DF':
        metricas = ['Contribucion_Defensiva_original', 'Eficiencia_Pases_original']
        ejes = {'x': 'Contribucion_Defensiva_original', 'y': 'Eficiencia_Pases_original'}
    elif posicion == 'MF':
        metricas = ['Creacion_Juego_90_original', 'Eficiencia_Pases_original']
        ejes = {'x': 'Creacion_Juego_90_original', 'y': 'Eficiencia_Pases_original'}
    elif posicion == 'FW':
        metricas = ['Efectividad_Ataque_original', 'Goles_Por_90_original']
        ejes = {'x': 'Efectividad_Ataque_original', 'y': 'Goles_Por_90_original'}
    
    # Añadir las métricas a las columnas de resultado
    for metrica in metricas:
        if metrica in similares.columns:
            columnas_resultado.append(metrica)
    
    # Formatear nombres (capitalizar)
    similares['Nombre'] = similares['Nombre'].str.title()
    similares['Equipo'] = similares['Equipo'].str.title()
    
    # Seleccionar columnas finales
    columnas_disponibles = [col for col in columnas_resultado if col in similares.columns]
    df_result = similares[columnas_disponibles].copy()
    
    # Preparar resultados para visualización
    result = {
        'jugador': jugador_real,
        'posicion': posicion,
        'similares': df_result,
        'ejes': ejes,
        'cluster_id': cluster_id,
        'metricas': metricas
    }
    
    return result

def generar_grafico_similares(result, recursos):
    """
    Genera un gráfico de dispersión con los jugadores similares
    
    Args:
        result: Diccionario con resultados de jugadores similares
        recursos: Diccionario con los recursos cargados
        
    Returns:
        fig: Figura de matplotlib
    """
    if not result:
        return None
    
    jugador = result['jugador']
    posicion = result['posicion']
    df_similares = result['similares']
    ejes = result['ejes']
    cluster_id = result['cluster_id']
    
    # Obtener datos del cluster
    df_pos = recursos['datos_procesados'][posicion]['df'].copy()
    
    # Normalizar nombre del jugador
    jugador_norm = normalizar_texto(jugador)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(10, 7), facecolor="lightgrey")
    
    # Jugadores del mismo cluster (fondo)
    cluster_df = df_pos[df_pos['cluster'] == cluster_id]
    sns.scatterplot(
        data=cluster_df, 
        x=ejes['x'], 
        y=ejes['y'], 
        color='lightgrey',
        alpha=0.5,
        s=50,
        label=f'Cluster {cluster_id}',
        ax=ax
    )
    
    # Jugadores similares
    sns.scatterplot(
        data=df_similares, 
        x=ejes['x'], 
        y=ejes['y'], 
        color='green',
        s=100,
        label='Recomendados',
        ax=ax
    )
    
    # Jugador evaluado
    jugador_data = df_pos[df_pos['Nombre'] == jugador_norm]
    ax.scatter(
        jugador_data[ejes['x']], 
        jugador_data[ejes['y']],
        color='darkblue', 
        s=200, 
        edgecolor='white',
        linewidth=2,
        label=jugador
    )
    
    # Añadir etiquetas a los similares
    for i, row in df_similares.iterrows():
        ax.text(
            row[ejes['x']], 
            row[ejes['y']], 
            row['Nombre'],
            fontsize=8, 
            color='darkgreen',
            ha='center',
            va='bottom'
        )
    
    # Estilo del gráfico
    ax.set_title(f'Jugadores similares a {jugador} ({posicion})', 
              fontsize=15, color='darkblue', weight='bold')
    ax.set_xlabel(ejes['x'].replace('_original', '').replace('_', ' ').title(), color='darkblue')
    ax.set_ylabel(ejes['y'].replace('_original', '').replace('_', ' ').title(), color='darkblue')
    
    # Personalización
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_color('darkblue')
        spine.set_linewidth(1)
    ax.grid(True, color='lightgray', linestyle='-', alpha=0.7)
    
    ax.legend(loc='best')
    plt.tight_layout()
    
    return fig

def verificar_modelos():
    """
    Verifica si los modelos están entrenados
    
    Returns:
        bool: True si los modelos están entrenados, False en caso contrario
    """
    archivos_necesarios = [
        'df_atletico.joblib',
        'alias_mapping.joblib',
        'resultado_clusters.joblib',
        'datos_procesados.joblib'
    ]
    
    for archivo in archivos_necesarios:
        if not os.path.exists(os.path.join(MODEL_DIR, archivo)):
            return False
    
    # Verificar que existe al menos un modelo KNN
    existe_modelo = False
    for pos in ['GK', 'DF', 'MF', 'FW']:
        if os.path.exists(os.path.join(MODEL_DIR, f'knn_{pos}.joblib')):
            existe_modelo = True
            break
    
    return existe_modelo

def comparar_jugadores(jugador1, jugador2, recursos):
    """
    Compara dos jugadores y calcula su similitud
    
    Args:
        jugador1: Nombre del primer jugador
        jugador2: Nombre del segundo jugador
        recursos: Diccionario con los recursos cargados
        
    Returns:
        dict: Diccionario con resultados de la comparación
    """
    # TODO: Implementar comparación de jugadores
    pass

# Función para entrenar modelos si no existen
def entrenar_modelos_si_necesario():
    """
    Verifica si los modelos existen y los entrena si es necesario
    
    Returns:
        bool: True si los modelos están disponibles, False en caso contrario
    """
    if verificar_modelos():
        return True
    
    try:
        print("Entrenando modelos...")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)        
        train_model.main()
        return verificar_modelos()
    except Exception as e:
        print(f"Error al entrenar modelos: {e}")
        return False

if not verificar_modelos():
    print("ADVERTENCIA: Los modelos no están entrenados. Utiliza entrenar_modelos_si_necesario() para entrenarlos.")