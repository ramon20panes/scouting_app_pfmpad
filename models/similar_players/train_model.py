import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib
import os
import unicodedata
import time
import re

# Crear directorio para guardar modelos si no existe
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Alias de nombres para normalizar inconsistencias
alias_mapping = {
    "Musso": "Juan Musso", "Gimenez": "Jose Maria Gimenez", "Azpilicueta": "Cesar Azpilicueta",
    "Koke": "Jorge Resurreccion", "Gallagher": "Conor Gallagher", "De Paul": "Rodrigo De Paul",
    "Griezmann": "Antoine Griezmann", "Barrios": "Pablo Barrios", "Sorloth": "Alexander Sorloth",
    "Aleksander Sorloth": "Alexander Sorloth", "Sorloth": "Alexander Sørloth", "Alexander Sorloth": "Alexander Sørloth",
    "Correa": "Angel Correa", "Lemar": "Thomas Lemar", "Samu Lino": "Samuel Lino", "Lino": "Samu Lino",
    "Oblak": "Jan Oblak", "Llorente": "Marcos Llorente", "Lenglet": "Clement Lenglet",
    "Molina": "Nahuel Molina", "Riquelme": "Rodrigo Riquelme", "J. Alvarez": "Julian Alvarez", "Julian": "Julian Alvarez",
    "Witsel": "Axel Witsel", "J. Galan": "Javi Galan", "Galan": "Javi Galan", "Giuliano": "Giuliano Simeone",
    "Reinildo": "Reinildo Mandava", "Le Normand": "Robin Le Normand",
}

def normalizar_texto(texto):
    """
    Normaliza texto (quita acentos, pasa a minúsculas)
    """
    if isinstance(texto, str):
        texto = texto.strip().lower()
        texto = unicodedata.normalize('NFKD', texto)
        return ''.join(c for c in texto if not unicodedata.combining(c))
    return texto

def cargar_datos():
    """
    Carga los datasets necesarios para el análisis
    """
    try:
        # Carga los datos según tu función original
        df_general = pd.read_parquet('data/raw/similar/stats_big5_24_25.parquet')
        df_porteros = pd.read_csv('data/raw/similar/stats_big5_gk_24_25.csv')
        df_mercado = pd.read_csv('data/raw/similar/knn_players.csv')
        df_atletico = pd.read_csv('data/raw/master/jugadores_master.csv', sep=';')
        df_equipos = pd.read_csv('data/raw/similar/equipos_big5_stats_24_25.csv', sep=',')
        df_master_equipos = pd.read_csv('data/raw/master/equipos_big5_master.csv', sep=';')
        
        return df_general, df_porteros, df_mercado, df_atletico, df_equipos, df_master_equipos
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        # Intentar con ruta alternativa
        try:
            df_general = pd.read_csv('../data/raw/similar/stats_big5_24_25.csv', sep=',')
            df_porteros = pd.read_csv('../data/raw/similar/stats_big5_gk_24_25.csv', sep=',')
            df_mercado = pd.read_csv('../data/raw/similar/knn_players.csv', sep=',')
            df_atletico = pd.read_csv('../data/raw/master/jugadores_master.csv', sep=';')
            df_equipos = pd.read_csv('../data/raw/similar/equipos_big5_stats_24_25.csv')
            df_master_equipos = pd.read_csv('../data/raw/master/equipos_big5_master.csv', sep=';')
            
            return df_general, df_porteros, df_mercado, df_atletico, df_equipos, df_master_equipos
        except Exception as e2:
            print(f"Error al cargar los datos con ruta alternativa: {e2}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def limpiar_datos(df_general, df_porteros, df_mercado, df_atletico, df_equipos, df_master_equipos):
    """
    Realiza la limpieza básica de los datos
    """
    # Función para normalizar texto (quitar acentos y caracteres especiales)
    def normalizar_texto(texto):
        if isinstance(texto, str):
            texto_normalizado = unicodedata.normalize('NFKD', texto)
            texto_sin_acentos = ''.join([c for c in texto_normalizado if not unicodedata.combining(c)])
            return texto_sin_acentos
        return texto
    
    # Limpieza básica para df_atletico
    if not df_atletico.empty:
        # Duplicados
        df_atletico.drop_duplicates(subset=['nombre_completo', 'dorsal'], inplace=True, keep='first')
        
        # Normalizar nombres
        if 'nombre_completo' in df_atletico.columns:
            df_atletico['nombre_completo'] = df_atletico['nombre_completo'].apply(normalizar_texto)
        if 'short_name' in df_atletico.columns:
            df_atletico['short_name'] = df_atletico['short_name'].apply(normalizar_texto)
        
        # Convertir fecha de nacimiento a datetime
        if 'fecha_nacimiento' in df_atletico.columns:
            df_atletico['fecha_nacimiento'] = pd.to_datetime(
                df_atletico['fecha_nacimiento'], 
                format='%d/%m/%Y', 
                errors='coerce'
            )
            # Extraer solo el año
            df_atletico['fecha_nacimiento'] = df_atletico['fecha_nacimiento'].dt.year
        
        # Renombrar columnas con diccionario
        cambios_columnas = {
            'nombre_completo': 'Nombre',
            'short_name': 'Nombre_corto',
            'posicion': 'Posicion',
            'dorsal': 'Numero',
            'pais': 'Nacionalidad',
            'fecha_nacimiento': 'Nacimiento'
        }
        df_atletico.rename(columns=cambios_columnas, inplace=True)
    
    # Limpieza básica para df_general
    if not df_general.empty:
        # Eliminar duplicados
        df_general.drop_duplicates(subset=['Jugador', 'Equipo'], inplace=True, keep='first')

        # Normalizar nombres de jugadores
        if 'Jugador' in df_general.columns:
            df_general['Jugador'] = df_general['Jugador'].apply(normalizar_texto)

        # Renombrar columnas con diccionario
        cambios_columnas = {
            'Jugador': 'Nombre',
            'Entradas_canadas': 'Entradas_ganadas',
            'Bloqueos_cisparo': 'Bloqueos_disparos',
            'pAdj_Entradas+Intercepciones_90min': 'Acc_defensivas_por90_ajus_posesion',
            'pAdjClrPer90': 'Despejes_por90_ajus_posesion'
        }
        df_general.rename(columns=cambios_columnas, inplace=True)
        
        # Convertir Minutos a int
        if 'Minutos' in df_general.columns:
            df_general['Minutos'] = df_general['Minutos'].astype(int)
        
        # Manejo de valores faltantes para datos numéricos
        numeric_cols = df_general.select_dtypes(include=['float64', 'int64']).columns
        df_general[numeric_cols] = df_general.groupby('Posicion')[numeric_cols].transform(
            lambda x: x.fillna(x.median())
        )

        # Eliminar columna de Posicion Principal si existe
        if 'Posicion_princ' in df_general.columns:
            df_general.drop('Posicion_princ', axis=1, inplace=True)
    
    # Limpieza básica para df_porteros
    if not df_porteros.empty:
        # Eliminar duplicados
        df_porteros.drop_duplicates(subset=['Jugador', 'Equipo'], inplace=True, keep='first')

        # Normalizar nombres de porteros
        if 'Jugador' in df_porteros.columns:
            df_porteros['Jugador'] = df_porteros['Jugador'].apply(normalizar_texto)
        
        # Convertir Minutos a int
        if 'Minutos' in df_porteros.columns:
            df_porteros['Minutos'] = df_porteros['Minutos'].astype(int)
        
        # Manejo de valores faltantes para datos numéricos
        numeric_cols = df_porteros.select_dtypes(include=['float64', 'int64']).columns
        df_porteros[numeric_cols] = df_porteros[numeric_cols].fillna(df_porteros[numeric_cols].median())
        
        # Renombrar columnas con diccionario
        cambios_columnas = {
            'Jugador': 'Nombre',
            'CS%': 'Porcentajes_porterias_a_cero',
            'PKsFaced': 'Penaltis_recibidos',
            'PKsv': 'Penaltis_parados',
            'PKA':  'Penaltis_encajados',
            'PKm': 'Penaltis_errados_por_rival',
            'PSxG': 'xG_tras_tiro',
            'PSxG/SoT': 'PSxG_por_disparos_a_puerta',
            'PSxG+/-': 'Dif_PSxG_goles_encajados',
            'PSxG+/- /90': 'Diferencial_por_90',
            'LaunchCmp': 'Pases_largos_cmpl',
            'LaunchAtt': 'Pases_largos_int',
            'LaunchPassCmp%': 'Porcentaje_pases_largos',
            'GoalKicksAtt': 'Saques_de_puerta',
            'GoalKicksLaunch%': 'Saques_de_puerta_lanzados_largos',
            'AvgLen': 'Long_media_desplaz_largos',
            'CrsStp%': '%Centros_atajados',
            '#OPA/90': 'Acc_fuera_area_por_90',
            'AvgDistOPA': 'Distancia_media_porteria',
            'pAdjAerialWinsPer90': 'Duelos_aereos_ganados_por_90',
            'pAdjTouchesPer90': 'Toques_por_90_ajust_posesion'
        }
        df_porteros.rename(columns=cambios_columnas, inplace=True)
    
    # Limpieza básica para df_mercado
    if not df_mercado.empty:
        # Duplicados
        df_mercado.drop_duplicates(subset=['nombre', 'dorsal', 'equipo'], inplace=True, keep='first')

        # Normalizar nombres
        if 'nombre' in df_mercado.columns:
            df_mercado['nombre'] = df_mercado['nombre'].apply(normalizar_texto)
        
        # Renombrar columnas
        cambios_columnas = {
            'dorsal': 'Numero',
            'nombre': 'Nombre',
            'fin_contrato': 'Contrato',
            'valor_mercado': 'Valor',
            'equipo': 'Equipo'
        }
        df_mercado.rename(columns=cambios_columnas, inplace=True)
    
    return df_general, df_porteros, df_mercado, df_atletico, df_equipos, df_master_equipos

def crear_features(df_general, df_porteros):
    """
    Crear características específicas por posición para análisis de similitud de jugadores.
    """
    # Inicializar diccionario para almacenar DFs por posición
    dfs_posicion = {
        'GK': pd.DataFrame(),
        'DF': pd.DataFrame(),
        'MF': pd.DataFrame(),
        'FW': pd.DataFrame()
    }
    
    # Copiar los DataFrames originales
    df_general = df_general.copy() if not df_general.empty else pd.DataFrame()
    df_porteros = df_porteros.copy() if not df_porteros.empty else pd.DataFrame()
    
    # Preparar características comunes para jugadores de campo
    if not df_general.empty:
        # 1. Efectividad en ataque (goles + asistencias por 90 minutos)
        if all(col in df_general.columns for col in ['Goles', 'Asistencias', 'Minutos']):
            df_general['Efectividad_Ataque'] = (df_general['Goles'] + df_general['Asistencias']) / (df_general['Minutos'] / 90)
            df_general['Efectividad_Ataque'] = df_general['Efectividad_Ataque'].fillna(0)
        
        # 2. Eficiencia de pases
        if all(col in df_general.columns for col in ['Pases_completados', 'Pases_intentados']):
            df_general['Eficiencia_Pases'] = df_general['Pases_completados'] / df_general['Pases_intentados']
            df_general['Eficiencia_Pases'] = df_general['Eficiencia_Pases'].fillna(0)
        
        # 3. Contribución defensiva
        if all(col in df_general.columns for col in ['Entradas', 'Intercepciones', 'Minutos']):
            df_general['Contribucion_Defensiva'] = (df_general['Entradas'] + df_general['Intercepciones']) / (df_general['Minutos'] / 90)
            df_general['Contribucion_Defensiva'] = df_general['Contribucion_Defensiva'].fillna(0)
        
        # 4. Creación de ocasiones
        if all(col in df_general.columns for col in ['Pases_clave', 'Minutos']):
            df_general['Creacion_Ocasiones'] = df_general['Pases_clave'] / (df_general['Minutos'] / 90)
            df_general['Creacion_Ocasiones'] = df_general['Creacion_Ocasiones'].fillna(0)
            
        # 5. Precisión de disparo
        if all(col in df_general.columns for col in ['Disparos_porteria', 'Disparos']):
            df_general['Precision_Disparo'] = df_general['Disparos_porteria'] / df_general['Disparos']
            df_general['Precision_Disparo'] = df_general['Precision_Disparo'].fillna(0)
        
        # 6. Eficiencia ofensiva
        if all(col in df_general.columns for col in ['Goles', 'xG']):
            df_general['Eficiencia_Ofensiva'] = df_general['Goles'] / df_general['xG']
            df_general['Eficiencia_Ofensiva'] = df_general['Eficiencia_Ofensiva'].fillna(1)
        
        # 7. Progresión de juego
        if all(col in df_general.columns for col in ['Pases_progresivos', 'Conducciones_progresivas', 'Minutos']):
            df_general['Indice_Progresion'] = (df_general['Pases_progresivos'] + df_general['Conducciones_progresivas']) / (df_general['Minutos'] / 90)
            df_general['Indice_Progresion'] = df_general['Indice_Progresion'].fillna(0)
        
        # 8. Eficiencia en duelos
        if all(col in df_general.columns for col in ['Duelos_aereos_ganados', 'Duelos_aereos_perdidos']):
            df_general['Eficiencia_Duelos_Aereos'] = df_general['Duelos_aereos_ganados'] / (df_general['Duelos_aereos_ganados'] + df_general['Duelos_aereos_perdidos'])
            df_general['Eficiencia_Duelos_Aereos'] = df_general['Eficiencia_Duelos_Aereos'].fillna(0)
            
        if all(col in df_general.columns for col in ['Regates_exitosos', 'Regates_intentados']):
            df_general['Eficiencia_Regates'] = df_general['Regates_exitosos'] / df_general['Regates_intentados']
            df_general['Eficiencia_Regates'] = df_general['Eficiencia_Regates'].fillna(0)
        
        # 9. Influencia en el juego
        if all(col in df_general.columns for col in ['Centralidad_Toques', 'Minutos']):
            df_general['Influencia_Juego'] = df_general['Centralidad_Toques'] * (df_general['Minutos'] / 90)
            df_general['Influencia_Juego'] = df_general['Influencia_Juego'].fillna(0)
        
    # Preparar características para porteros
    if not df_porteros.empty:
        # 1. Eficiencia de paradas
        if all(col in df_porteros.columns for col in ['Paradas', 'Disparos_recibidos_porteria']):
            df_porteros['Eficiencia_Paradas'] = df_porteros['Paradas'] / df_porteros['Disparos_recibidos_porteria']
            df_porteros['Eficiencia_Paradas'] = df_porteros['Eficiencia_Paradas'].fillna(0)
        
        # 2. Porterías a cero por partido
        if all(col in df_porteros.columns for col in ['Porterías_cero', 'Partidos']):
            df_porteros['Porcentaje_Porterias_Cero'] = df_porteros['Porterías_cero'] / df_porteros['Partidos']
            df_porteros['Porcentaje_Porterias_Cero'] = df_porteros['Porcentaje_Porterias_Cero'].fillna(0)
        
        # 3. Eficiencia de pases para porteros
        if all(col in df_porteros.columns for col in ['Pases_largos_cmpl', 'Pases_largos_int']):
            df_porteros['Eficiencia_Pases_Largos'] = df_porteros['Pases_largos_cmpl'] / df_porteros['Pases_largos_int']
            df_porteros['Eficiencia_Pases_Largos'] = df_porteros['Eficiencia_Pases_Largos'].fillna(0)
        
        # 4. Rendimiento vs esperado
        if 'Dif_PSxG_goles_encajados' in df_porteros.columns:
            df_porteros['Rendimiento_vs_Esperado'] = df_porteros['Dif_PSxG_goles_encajados']
            df_porteros['Rendimiento_vs_Esperado'] = df_porteros['Rendimiento_vs_Esperado'].fillna(0)
        
        # 5. Actividad fuera del área
        if all(col in df_porteros.columns for col in ['Acciones_fuera_area', 'Minutos']):
            df_porteros['Actividad_Fuera_Area'] = df_porteros['Acciones_fuera_area'] / (df_porteros['Minutos'] / 90)
            df_porteros['Actividad_Fuera_Area'] = df_porteros['Actividad_Fuera_Area'].fillna(0)
        
    # PORTEROS (GK)
    if not df_porteros.empty:
        gk_df = df_porteros.copy()
        
        gk_features = [
            'Nombre', 'Equipo', 'Edad', 'Minutos',
            'Eficiencia_Paradas', 'Porcentaje_Porterias_Cero', 
            'Eficiencia_Pases_Largos', 'Rendimiento_vs_Esperado',
            'Actividad_Fuera_Area'
        ]
        
        # Filtrar columnas disponibles
        gk_features = [col for col in gk_features if col in gk_df.columns]
        
        # Asignar al diccionario
        if len(gk_features) > 3:
            dfs_posicion['GK'] = gk_df[gk_features]
    
    # JUGADORES DE CAMPO (DF, MF, FW)
    if not df_general.empty:
        # DEFENSAS (DF)
        df_df = df_general[df_general['Posicion'] == 'DF'].copy()
        if not df_df.empty:
            # Características específicas para defensas
            if all(col in df_df.columns for col in ['Entradas', 'Entradas_ganadas']):
                df_df['Eficiencia_Entradas'] = df_df['Entradas_ganadas'] / df_df['Entradas']
                df_df['Eficiencia_Entradas'] = df_df['Eficiencia_Entradas'].fillna(0)
            
            if all(col in df_df.columns for col in ['Bloqueos_disparos', 'Despejes', 'Minutos']):
                df_df['Acciones_Defensivas_90'] = (df_df['Bloqueos_disparos'] + df_df['Despejes']) / (df_df['Minutos'] / 90)
                df_df['Acciones_Defensivas_90'] = df_df['Acciones_Defensivas_90'].fillna(0)
            
            # Seleccionar características para defensas
            df_features = [
                'Nombre', 'Equipo', 'Edad', 'Minutos', 'Posicion',
                'Contribucion_Defensiva', 'Eficiencia_Pases', 'Eficiencia_Duelos_Aereos',
                'Eficiencia_Entradas', 'Acciones_Defensivas_90', 
                'Indice_Progresion', 'Influencia_Juego'
            ]
            
            # Filtrar columnas disponibles
            df_features = [col for col in df_features if col in df_df.columns]
            
            # Asignar al diccionario
            if len(df_features) > 3:
                dfs_posicion['DF'] = df_df[df_features]
        
        # CENTROCAMPISTAS (MF)
        df_mf = df_general[df_general['Posicion'] == 'MF'].copy()
        if not df_mf.empty:
            # Características específicas para centrocampistas
            if all(col in df_mf.columns for col in ['Pases_clave', 'Pases_progresivos', 'Minutos']):
                df_mf['Creacion_Juego_90'] = (df_mf['Pases_clave'] + df_mf['Pases_progresivos']) / (df_mf['Minutos'] / 90)
                df_mf['Creacion_Juego_90'] = df_mf['Creacion_Juego_90'].fillna(0)
            
            if all(col in df_mf.columns for col in ['Pases_filtrados', 'Minutos']):
                df_mf['Pases_Filtrados_90'] = df_mf['Pases_filtrados'] / (df_mf['Minutos'] / 90)
                df_mf['Pases_Filtrados_90'] = df_mf['Pases_Filtrados_90'].fillna(0)
            
            # Seleccionar características para centrocampistas
            mf_features = [
                'Nombre', 'Equipo', 'Edad', 'Minutos', 'Posicion',
                'Eficiencia_Pases', 'Contribucion_Defensiva', 'Creacion_Ocasiones',
                'Creacion_Juego_90', 'Pases_Filtrados_90', 'Indice_Progresion',
                'Eficiencia_Regates', 'Influencia_Juego', 'Efectividad_Ataque'
            ]
            
            # Filtrar columnas disponibles
            mf_features = [col for col in mf_features if col in df_mf.columns]
            
            # Asignar al diccionario
            if len(mf_features) > 3:
                dfs_posicion['MF'] = df_mf[mf_features]
        
        # DELANTEROS (FW)
        df_fw = df_general[df_general['Posicion'] == 'FW'].copy()
        if not df_fw.empty:
            # Características específicas para delanteros
            if all(col in df_fw.columns for col in ['Goles', 'xG', 'Minutos']):
                df_fw['Goles_Sobre_xG'] = df_fw['Goles'] - df_fw['xG']
                df_fw['Goles_Por_90'] = df_fw['Goles'] / (df_fw['Minutos'] / 90)
            
            if all(col in df_fw.columns for col in ['Disparos', 'Minutos']):
                df_fw['Disparos_Por_90'] = df_fw['Disparos'] / (df_fw['Minutos'] / 90)
            
            # Seleccionar características para delanteros
            fw_features = [
                'Nombre', 'Equipo', 'Edad', 'Minutos', 'Posicion',
                'Efectividad_Ataque', 'Precision_Disparo', 'Eficiencia_Ofensiva',
                'Goles_Sobre_xG', 'Goles_Por_90', 'Disparos_Por_90',
                'Eficiencia_Regates', 'Creacion_Ocasiones', 'Eficiencia_Duelos_Aereos'
            ]
            
            # Filtrar columnas disponibles
            fw_features = [col for col in fw_features if col in df_fw.columns]
            
            # Asignar al diccionario
            if len(fw_features) > 3:
                dfs_posicion['FW'] = df_fw[fw_features]
    
    # Normalizar características numéricas para cada posición
    for pos in dfs_posicion:
        if not dfs_posicion[pos].empty:
            # Identificar columnas numéricas exceptuando identificadores
            non_numeric_cols = ['Nombre', 'Equipo', 'Posicion', 'Posicion_2', 'Nacionalidad']
            numeric_cols = [col for col in dfs_posicion[pos].columns 
                           if col not in non_numeric_cols]
            
            # Crear copia de columnas originales
            for col in numeric_cols:
                dfs_posicion[pos][f'{col}_original'] = dfs_posicion[pos][col]
            
            # Aplicar normalización
            if len(numeric_cols) > 0:
                scaler = StandardScaler()
                dfs_posicion[pos][numeric_cols] = scaler.fit_transform(dfs_posicion[pos][numeric_cols])
    
    return dfs_posicion

def preparar_datos_modelado(dfs_posicion, n_clusters=None):
    """
    Aplica algoritmo K-means a cada conjunto de datos por posición.
    """
    resultado_clusters = {}
    
    # Determinar número de clusters automáticamente
    if n_clusters is None:
        n_clusters = {
            'GK': 5,
            'DF': 6,
            'MF': 7,
            'FW': 6
        }
    
    # Para cada posición
    for pos, df in dfs_posicion.items():
        if df.empty:
            resultado_clusters[pos] = df
            continue
        
        # Guardar columnas no numéricas para después
        non_numeric_cols = ['Nombre', 'Equipo', 'Posicion', 'Posicion_2', 'Nacionalidad']
        id_cols = [col for col in non_numeric_cols if col in df.columns]

        # Obtener columnas para clustering (numéricas que no sean "_original")
        feature_cols = [col for col in df.columns 
                    if col not in id_cols 
                    and not col.endswith('_original')
                    and pd.api.types.is_numeric_dtype(df[col])]
        
        if len(feature_cols) < 2:
            resultado_clusters[pos] = df
            continue
        
        # Determinar el número de clusters
        k = n_clusters[pos]
        if isinstance(n_clusters, dict) and pos in n_clusters:
            k = n_clusters[pos]
        
        # Aplicar K-means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        df_copy = df.copy()
        
        # Añadir cluster al DataFrame
        df_copy['cluster'] = kmeans.fit_predict(df[feature_cols])
        
        # Calcular PCA para visualización
        if len(feature_cols) >= 2:
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(df[feature_cols])
            df_copy['pca_x'] = pca_result[:, 0]
            df_copy['pca_y'] = pca_result[:, 1]
        
        # Guardar resultado
        resultado_clusters[pos] = df_copy
    
    return resultado_clusters

def entrenar_modelo_knn(resultado_clusters, df_atletico, df_general, df_porteros, df_mercado=None, df_equipos=None):
    """
    Prepara los modelos KNN para cada posición basados en los clusters
    """
    # Diccionario para almacenar modelos KNN por posición
    modelos_knn = {}
    
    # Diccionario para almacenar datos procesados
    datos_procesados = {}
    
    # Para cada posición
    for pos in resultado_clusters.keys():
        if resultado_clusters[pos].empty:
            continue
                
        # Obtener datos del cluster
        df_pos = resultado_clusters[pos].copy()
        
        # Añadir datos adicionales según posición
        if pos != 'GK':
            # Para jugadores de campo, añadir datos de df_general
            columnas_extra = ['Nombre', 'Edad', 'Nacimiento', 'Partidos', 'Posicion_2']
            columnas_disponibles = [col for col in columnas_extra if col in df_general.columns]
            if columnas_disponibles:
                # Normalizar nombres para merge
                df_general_temp = df_general[columnas_disponibles].copy()
                df_general_temp['Nombre'] = df_general_temp['Nombre'].apply(lambda x: x.strip().lower())
                df_pos['Nombre'] = df_pos['Nombre'].apply(lambda x: x.strip().lower())
                
                # Merge con datos generales
                df_pos = df_pos.merge(df_general_temp, on='Nombre', how='left')
        else:
            # Para porteros, añadir datos específicos de df_porteros
            columnas_extra = ['Nombre', 'Edad', 'Partidos']
            columnas_disponibles = [col for col in columnas_extra if col in df_porteros.columns]
            if columnas_disponibles:
                # Normalizar nombres para merge
                df_porteros_temp = df_porteros[columnas_disponibles].copy()
                df_porteros_temp['Nombre'] = df_porteros_temp['Nombre'].apply(lambda x: x.strip().lower())
                df_pos['Nombre'] = df_pos['Nombre'].apply(lambda x: x.strip().lower())
                
                # Merge con datos de porteros
                df_pos = df_pos.merge(df_porteros_temp, on='Nombre', how='left')
        
        # Añadir datos de mercado (Valor, Contrato, Altura, Pie)
        if df_mercado is not None:
            cols_mercado = ['Nombre', 'Valor', 'Contrato', 'Altura', 'Pie']
            cols_disponibles = [col for col in cols_mercado if col in df_mercado.columns]
            if cols_disponibles:
                df_mercado_temp = df_mercado[cols_disponibles].copy()
                df_mercado_temp['Nombre'] = df_mercado_temp['Nombre'].apply(lambda x: x.strip().lower())
                
                # Merge con datos de mercado
                df_pos = df_pos.merge(df_mercado_temp, on='Nombre', how='left')
        
        # Añadir datos de equipos (Media_edad, Posesion)
        if df_equipos is not None and 'Equipo' in df_pos.columns:
            cols_equipos = ['Equipo', 'Media_edad', 'Posesion']
            cols_disponibles = [col for col in cols_equipos if col in df_equipos.columns]
            if cols_disponibles:
                df_equipos_temp = df_equipos[cols_disponibles].copy()
                df_equipos_temp['Equipo'] = df_equipos_temp['Equipo'].apply(lambda x: x.strip().lower())
                df_pos['Equipo'] = df_pos['Equipo'].apply(lambda x: x.strip().lower())
                
                # Merge con datos de equipos
                df_pos = df_pos.merge(df_equipos_temp, on='Equipo', how='left')
        
        # Rellenar valores numéricos faltantes
        for col in df_pos.select_dtypes(include=['float64', 'int64']).columns:
            df_pos[col] = df_pos[col].fillna(df_pos[col].median())
        
        # Definir columnas para el modelo KNN
        non_feature_cols = ['Nombre', 'Equipo', 'Posicion', 'Posicion_2', 'Nacionalidad', 
                           'cluster', 'pca_x', 'pca_y', 'Pie']
        feature_cols = [col for col in df_pos.columns 
                       if col not in non_feature_cols 
                       and not col.endswith('_original')
                       and pd.api.types.is_numeric_dtype(df_pos[col])]
        
        if len(feature_cols) < 2:
            print(f"Insuficientes características numéricas para posición {pos}")
            continue
        
        # Eliminar filas con NaN en características
        df_pos = df_pos.dropna(subset=feature_cols)
        
        if df_pos.empty:
            print(f"No hay datos válidos para entrenar el modelo KNN en posición {pos}")
            continue
        
        # Escalar características
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_pos[feature_cols])
        
        # Guardar el scaler para uso posterior
        joblib.dump(scaler, os.path.join(MODEL_DIR, f'scaler_{pos}.joblib'))
        
        # Entrenar modelo KNN
        n_neighbors = min(11, len(df_pos))  # Garantizar que hay suficientes vecinos
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        knn.fit(X_scaled)
        
        # Guardar modelo KNN
        joblib.dump(knn, os.path.join(MODEL_DIR, f'knn_{pos}.joblib'))
        
        # Normalizar nombres en la columna 'Nombre' antes de guardar
        df_pos['Nombre'] = df_pos['Nombre'].apply(normalizar_texto)

        datos_procesados[pos] = {
            'df': df_pos,
            'feature_cols': feature_cols
        }
    
    # Guardar datos procesados para uso posterior
    joblib.dump(datos_procesados, os.path.join(MODEL_DIR, 'datos_procesados.joblib'))
    
    # Guardar resultado_clusters
    joblib.dump(resultado_clusters, os.path.join(MODEL_DIR, 'resultado_clusters.joblib'))
    
    return modelos_knn, datos_procesados

def main():
    """
    Función principal para entrenar y guardar modelos
    """

    # 1. Cargar datos    
    df_general, df_porteros, df_mercado, df_atletico, df_equipos, df_master_equipos = cargar_datos()
    
    # 2. Limpiar datos    
    df_general, df_porteros, df_mercado, df_atletico, df_equipos, df_master_equipos = limpiar_datos(
        df_general, df_porteros, df_mercado, df_atletico, df_equipos, df_master_equipos
    )
    
    # 3. Crear características por posición
    dfs_posicion = crear_features(df_general, df_porteros)
    
    # 4. Aplicar clustering K-means
    resultado_clusters = preparar_datos_modelado(dfs_posicion)
    
    # 5. Entrenar modelos KNN
    modelos_knn, datos_procesados = entrenar_modelo_knn(
        resultado_clusters, df_atletico, df_general, df_porteros, df_mercado, df_equipos
    )
    
    # 6. Guardar df_atletico para uso en predicciones
    # Normalizar columna 'Nombre' en df_atletico
    df_atletico['Nombre'] = df_atletico['Nombre'].apply(normalizar_texto)

    joblib.dump(df_atletico, os.path.join(MODEL_DIR, 'df_atletico.joblib'))
    
    # 7. Guardar información de alias
    joblib.dump(alias_mapping, os.path.join(MODEL_DIR, 'alias_mapping.joblib'))
    
    # 8. Guardar timestamp de entrenamiento
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    with open(os.path.join(MODEL_DIR, 'last_training.txt'), 'w') as f:
        f.write(f"Último entrenamiento: {timestamp}")
        
    return resultado_clusters, modelos_knn, datos_procesados

if __name__ == "__main__":
    main()