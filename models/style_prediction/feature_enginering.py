import streamlit as st  
import os
import numpy as np
import pandas as pd
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from typing import Dict, List, Tuple, Union, Optional

# Configuración de visualización
plt.style.use('ggplot')
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14
})

def load_data(master_path: str, stats_path: str, eventos_dir: str, equipos_path: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Carga los datos de los diferentes archivos necesarios para el análisis
    de estilos de juego del Atlético de Madrid.
    
    Args:
        master_path (str): Ruta al archivo master_liga_vert_atlmed.csv
        stats_path (str): Ruta al archivo stats_atm_por_partido_pag4.csv
        eventos_dir (str): Ruta al directorio con archivos de eventos (.parquet)
        equipos_path (str, opcional): Ruta al archivo de equipos
    
    Returns:
        dict: Diccionario con DataFrames cargados
    """
    data = {}
    
    # Carga del archivo master
    try:
        print(f"Leyendo archivo master: {master_path}")
        data['master'] = pd.read_csv(master_path, delimiter=';')
    except Exception as e:
        print(f"Error al cargar el archivo master: {e}")
        data['master'] = pd.DataFrame()
    
    # Carga de estadísticas por partido
    try:        
        # Mostrar primeras líneas del archivo para diagnóstico
        
        with open(stats_path, 'r') as f:
            for i in range(3):  # Mostrar 3 primeras líneas
                line = f.readline().strip()
        
        # Probar diferentes delimitadores
        delimitadores = [';', ',', '\t', '|']
        mejor_delimitador = ';'  # Delimitador predeterminado
        max_columnas = 0
        
        for delim in delimitadores:
            try:
                temp_df = pd.read_csv(stats_path, delimiter=delim)
                num_columnas = len(temp_df.columns)
                
                if num_columnas > max_columnas:
                    max_columnas = num_columnas
                    mejor_delimitador = delim
                    
            except Exception as e:
                print(f"Error con delimitador '{delim}': {e}")
        
        # Usar el mejor delimitador encontrado
        data['stats'] = pd.read_csv(stats_path, delimiter=mejor_delimitador)
                
    except Exception as e:
        print(f"Error al cargar el archivo de estadísticas: {e}")
        data['stats'] = pd.DataFrame()
    
    # Carga de archivos de eventos (parquet)
    try:
        print(f"Buscando archivos de eventos en: {eventos_dir}")
        evento_files = glob.glob(os.path.join(eventos_dir, '*EventData*.parquet'))
        
        # Lista para almacenar DataFrames de eventos
        event_dfs = []
        
        # Expresión regular para extraer jornada de nombres de archivo
        jornada_pattern = r'(\d+)ª?_([A-Z]{3})-([A-Z]{3})_EventData'
        
        # Procesar cada archivo
        for file_path in evento_files:
            file_name = os.path.basename(file_path)
            match = re.search(jornada_pattern, file_name)
            
            if match:
                jornada = int(match.group(1))
                equipo_local = match.group(2)
                equipo_visitante = match.group(3)
                
                # Leer archivo parquet
                df_evento = pd.read_parquet(file_path)
                
                # Añadir información de jornada y partido
                df_evento['jornada'] = jornada
                df_evento['partido'] = f"{equipo_local}-{equipo_visitante}"
                
                # Filtrar para eventos del Atlético (asumiendo teamId = 63)
                df_atletico = df_evento[df_evento['teamId'] == 63].copy()
                
                # Filtrar tiros y goles del rival
                df_rival = df_evento[(df_evento['teamId'] != 63) & 
                                    (df_evento['type'].str.contains('Shot|Goal', na=False))].copy()
                
                # Marcar como eventos del Atleti o del rival
                df_atletico['es_rival'] = False
                df_rival['es_rival'] = True
                
                # Combinar
                df_combined = pd.concat([df_atletico, df_rival])
                
                # Añadir al listado
                event_dfs.append(df_combined)
            else:
                print(f"No se pudo extraer información de jornada del archivo: {file_name}")
        
        # Combinar todos los DataFrames de eventos
        if event_dfs:
            data['eventos'] = pd.concat(event_dfs, ignore_index=True)
        else:
            print("No se pudieron procesar archivos de eventos")
            data['eventos'] = pd.DataFrame()
    except Exception as e:
        print(f"Error al procesar archivos de eventos: {e}")
        data['eventos'] = pd.DataFrame()
    
    # Cargar datos de equipos si se proporciona ruta
    if equipos_path:
        try:
            data['equipos'] = pd.read_csv(equipos_path, delimiter=';')
        except Exception as e:
            print(f"Error al cargar el archivo de equipos: {e}")
            data['equipos'] = pd.DataFrame()
    
    return data

def clean_data(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Realiza la limpieza de los DataFrames cargados.
    
    Args:
        data (dict): Diccionario con DataFrames originales
    
    Returns:
        dict: Diccionario con DataFrames limpios
    """
    clean_data = {}
    
    # Limpiar datos master
    if 'master' in data and not data['master'].empty:
        df_master = data['master'].copy()
        
        # Jornada a numérico
        if 'jornada' in df_master.columns:
            df_master['jornada_num'] = df_master['jornada'].str.replace('ª', '').astype(int)
        
        # Verticalidad a numérico (si existe)
        if 'verticalidad' in df_master.columns:
            df_master['verticalidad_num'] = df_master['verticalidad'].str.rstrip('%').astype(float)
        
        # Extraer componentes del resultado
        if 'resultado' in df_master.columns:
            df_master[['goles_favor_res', 'goles_contra_res']] = df_master['resultado'].str.split('-', expand=True).astype(float)
        
        # Verificar coherencia entre resultado y columnas de goles
        if all(col in df_master.columns for col in ['goles_favor_res', 'goles_a_favor', 'goles_contra_res', 'goles_en_contra']):
            mask_inconsistencia = (df_master['goles_favor_res'] != df_master['goles_a_favor']) | (df_master['goles_contra_res'] != df_master['goles_en_contra'])
            if mask_inconsistencia.any():
                print(f"ADVERTENCIA: Se encontraron {mask_inconsistencia.sum()} inconsistencias entre resultado y goles registrados")
        
        # Crear columna de resultado tipo
        df_master['resultado_tipo'] = 'Sin disputar'
        mask_jugados = ~df_master['resultado'].isna()
        
        if all(col in df_master.columns for col in ['goles_a_favor', 'goles_en_contra']):
            df_master.loc[mask_jugados & (df_master['goles_a_favor'] > df_master['goles_en_contra']), 'resultado_tipo'] = 'Victoria'
            df_master.loc[mask_jugados & (df_master['goles_a_favor'] == df_master['goles_en_contra']), 'resultado_tipo'] = 'Empate'
            df_master.loc[mask_jugados & (df_master['goles_a_favor'] < df_master['goles_en_contra']), 'resultado_tipo'] = 'Derrota'
        
        # Clasificar impacto de expulsiones
        if 'min_expulsion_propia' in df_master.columns:
            df_master['impacto_expulsion_propia'] = 0
            df_master.loc[df_master['min_expulsion_propia'] <= 30, 'impacto_expulsion_propia'] = 3  # Alta
            df_master.loc[(df_master['min_expulsion_propia'] > 30) & (df_master['min_expulsion_propia'] <= 60), 'impacto_expulsion_propia'] = 2  # Media
            df_master.loc[df_master['min_expulsion_propia'] > 60, 'impacto_expulsion_propia'] = 1  # Baja
        
        if 'min_expulsion_rival' in df_master.columns:
            df_master['impacto_expulsion_rival'] = 0
            df_master.loc[df_master['min_expulsion_rival'] <= 30, 'impacto_expulsion_rival'] = 3  # Alta
            df_master.loc[(df_master['min_expulsion_rival'] > 30) & (df_master['min_expulsion_rival'] <= 60), 'impacto_expulsion_rival'] = 2  # Media
            df_master.loc[df_master['min_expulsion_rival'] > 60, 'impacto_expulsion_rival'] = 1  # Baja
        
        # Crear variables para goles tempranos
        if all(col in df_master.columns for col in ['min_primer_gol_a_favor', 'min_primer_gol_en_contra']):
            df_master['gol_temprano_favor'] = (df_master['min_primer_gol_a_favor'] <= 15) & (~df_master['min_primer_gol_a_favor'].isna())
            df_master['gol_temprano_contra'] = (df_master['min_primer_gol_en_contra'] <= 15) & (~df_master['min_primer_gol_en_contra'].isna())
        
        clean_data['master'] = df_master
    
    # Limpiar datos de estadísticas
    if 'stats' in data and not data['stats'].empty:
        df_stats = data['stats'].copy()
        
        # Extracción de valores numéricos de columnas con porcentajes
        columnas_con_porcentaje = [
            'pases_precisos', 'pases_largos_precisos', 'centros_precisos', 
            'regates_exitosos', 'duelos_suelo_ganados', 'duelos_aereos_ganados', 
            'entradas_exitosas'
        ]
        
        # Función para extraer valor y porcentaje
        def extraer_valor_porcentaje(valor):
            if isinstance(valor, str) and '(' in valor:
                match = re.search(r'(\d+)\s*\((\d+(?:\.\d+)?)%\)', valor)
                if match:
                    return int(match.group(1)), float(match.group(2))
            # Si no se puede extraer, devolver valores predeterminados
            return valor, None
        
        # Procesar columnas con porcentajes
        for col in columnas_con_porcentaje:
            if col in df_stats.columns:
                # Crear columna para porcentaje
                nuevo_col = f"pct_{col}"
                
                # Extraer valor y porcentaje
                valores_porcentajes = df_stats[col].apply(extraer_valor_porcentaje)
                
                # Asignar valores
                df_stats[col] = valores_porcentajes.apply(lambda x: x[0])
                df_stats[nuevo_col] = valores_porcentajes.apply(lambda x: x[1])
        
        # Eliminar columnas duplicadas de tiros_bloqueados
        columnas_duplicadas = [col for col in df_stats.columns if re.match(r'tiros_bloqueados\.\d+', col)]
        if columnas_duplicadas:
            
            df_stats = df_stats.drop(columns=columnas_duplicadas)
        
        # Convertir columnas numéricas
        for col in df_stats.columns:
            if col != 'jornada' and not col.startswith('Unnamed'):
                try:
                    df_stats[col] = pd.to_numeric(df_stats[col], errors='coerce')
                except:
                    print(f"No se pudo convertir la columna {col} a numérico")
        
        # Asegurar que jornada sea numérica
        if 'jornada' in df_stats.columns:
            df_stats['jornada'] = pd.to_numeric(df_stats['jornada'], errors='coerce')
        
        # Ordenar por jornada
        if 'jornada' in df_stats.columns:
            df_stats = df_stats.sort_values('jornada')
        
        clean_data['stats'] = df_stats
    
    # Limpiar datos de eventos
    if 'eventos' in data and not data['eventos'].empty:
        df_eventos = data['eventos'].copy()
        
        # Procesar columnas JSON si es necesario
        for col in ['type', 'outcomeType', 'period', 'qualifiers']:
            if col in df_eventos.columns and df_eventos[col].dtype == 'object':
                # Verificar si es necesario parsear JSON
                try:
                    sample = df_eventos[col].iloc[0]
                    if isinstance(sample, str) and ('{' in sample or '[' in sample):
                       
                        # Crear columnas para nombre y valor
                        if col != 'qualifiers':  # qualifiers suele ser una lista
                            df_eventos[f'{col}_nombre'] = df_eventos[col].apply(
                                lambda x: json.loads(x.replace("'", '"'))['displayName'] 
                                if isinstance(x, str) else None)
                            
                            df_eventos[f'{col}_valor'] = df_eventos[col].apply(
                                lambda x: json.loads(x.replace("'", '"'))['value'] 
                                if isinstance(x, str) else None)
                        else:
                            # Para qualifiers, guardar longitud
                            df_eventos[f'{col}_longitud'] = df_eventos[col].apply(
                                lambda x: len(json.loads(x.replace("'", '"')))
                                if isinstance(x, str) else 0)
                except Exception as e:
                    print(f"Error al procesar columna JSON {col}: {e}")        
            
        # Función para clasificar zonas
        def clasificar_zona_vertical(x):
            if pd.isna(x):
                return 'Desconocida'
            if x <= 33.3:
                return 'Zona1'
            elif x <= 66.6:
                return 'Zona2'
            else:
                return 'Zona3'
        
        def clasificar_zona_horizontal(y):
            if pd.isna(y):
                return 'Desconocida'
            if y <= 25:
                return 'Derecho'
            elif y <= 75:
                return 'Central'
            else:
                return 'Izquierdo'
        
        # Aplicar clasificación si existen coordenadas
        if 'x' in df_eventos.columns:
            df_eventos['zona_vertical'] = df_eventos['x'].apply(clasificar_zona_vertical)
        
        if 'y' in df_eventos.columns:
            df_eventos['zona_horizontal'] = df_eventos['y'].apply(clasificar_zona_horizontal)
        
        # Combinar zonas
        if 'zona_vertical' in df_eventos.columns and 'zona_horizontal' in df_eventos.columns:
            df_eventos['zona_completa'] = df_eventos['zona_vertical'] + '_' + df_eventos['zona_horizontal']
        
        clean_data['eventos'] = df_eventos
    
    # Limpiar datos de equipos
    if 'equipos' in data and not data['equipos'].empty:
        df_equipos = data['equipos'].copy()
        
        # Limpiar nombres de columnas
        df_equipos.columns = df_equipos.columns.str.strip()
        
        # Limpiar rutas de escudos
        if 'ruta_escudo' in df_equipos.columns:
            df_equipos['ruta_escudo'] = df_equipos['ruta_escudo'].str.replace("'", "")
        
        clean_data['equipos'] = df_equipos
    
    return clean_data

def create_features(clean_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Realiza feature engineering para crear índices tácticos y categorizar estilos de juego.
    
    Args:
        clean_data (dict): Diccionario con DataFrames limpios
    
    Returns:
        pd.DataFrame: DataFrame con índices tácticos y categorías de estilo
    """
    
    # Obtener DataFrames limpios
    df_master = clean_data.get('master', pd.DataFrame())
    df_stats = clean_data.get('stats', pd.DataFrame())
    df_eventos = clean_data.get('eventos', pd.DataFrame())
    df_equipos = clean_data.get('equipos', pd.DataFrame())
    
    # Verificar si hay datos suficientes
    if df_master.empty or df_stats.empty:
        print("ERROR: No hay datos suficientes para crear features")
        return pd.DataFrame()
    
    # DataFrame para almacenar índices
    df_indices = pd.DataFrame()
    
    # 1. Procesar información espacial de eventos
    df_zonas = procesar_zonas_campo(df_eventos) if not df_eventos.empty else pd.DataFrame()
    
    # 2. Calcular índices tácticos básicos
    df_indices_basicos = calcular_indices_basicos(df_master, df_stats)
    
    # 3. Calcular índices tácticos avanzados
    df_indices_avanzados = calcular_indices_avanzados(df_eventos, df_zonas, df_stats)
    
    # 4. Calcular variables de contexto
    df_contexto = calcular_variables_contexto(df_master, df_equipos)
    
    # 5. Combinar índices básicos y avanzados
    
    # Asegurar que todos los DataFrame tengan 'jornada' como clave
    if 'jornada' not in df_indices_basicos.columns and 'jornada_num' in df_indices_basicos.columns:
        df_indices_basicos = df_indices_basicos.rename(columns={'jornada_num': 'jornada'})
    
    if not df_indices_avanzados.empty and 'jornada' not in df_indices_avanzados.columns:
        if 'jornada_num' in df_indices_avanzados.columns:
            df_indices_avanzados = df_indices_avanzados.rename(columns={'jornada_num': 'jornada'})
    
    # Combinar índices básicos y avanzados
    if not df_indices_avanzados.empty:
        df_combinado = pd.merge(df_indices_basicos, df_indices_avanzados, on='jornada', how='outer')
    else:
        df_combinado = df_indices_basicos.copy()
    
    # Incorporar estadísticas espaciales si están disponibles
    if not df_zonas.empty:
        df_combinado = pd.merge(df_combinado, df_zonas, on='jornada', how='left')
    
    # Añadir estadísticas generales para cálculos adicionales
    df_combinado = pd.merge(df_combinado, df_stats, on='jornada', how='left')
    
    # 6. Calcular índices especializados
    df_indices_especializados = calcular_indices_especializados(df_combinado)
    
    # 7. Integrar todos los índices y variables
    dfs_to_combine = [df_indices_basicos, df_indices_avanzados, df_indices_especializados, df_contexto]
    dfs_to_combine = [df for df in dfs_to_combine if not df.empty]
    
    # Combinar DataFrames para resultado final
    if dfs_to_combine:
        df_indices = dfs_to_combine[0].copy()
        for df in dfs_to_combine[1:]:
            df_indices = pd.merge(df_indices, df, on='jornada', how='outer')
    else:
        print("ADVERTENCIA: No hay DataFrames para combinar")
        return pd.DataFrame()
    
    # Asegurar que posesión esté en el DataFrame final
    if 'posesion' not in df_indices.columns and 'posesion' in df_combinado.columns:
        df_indices = pd.merge(df_indices, df_combinado[['jornada', 'posesion']], on='jornada', how='left')
    
    # Ordenar por jornada
    df_indices = df_indices.sort_values('jornada')
    
    # 8. Categorizar estilos de juego
    df_indices = categorizar_estilos(df_indices)
    
    return df_indices

def procesar_zonas_campo(df_eventos: pd.DataFrame) -> pd.DataFrame:
    """
    Procesa eventos para clasificarlos por zonas del campo y generar estadísticas espaciales.
    
    Args:
        df_eventos (pd.DataFrame): DataFrame con eventos
    
    Returns:
        pd.DataFrame: DataFrame con estadísticas espaciales por jornada
    """
    
    # Verificar si hay datos suficientes
    if df_eventos.empty:
        print("  No hay eventos para procesar")
        return pd.DataFrame()
    
    # Verificar que existan las columnas necesarias
    required_cols = ['jornada', 'x', 'y', 'type_nombre'] if 'type_nombre' in df_eventos.columns else ['jornada', 'x', 'y', 'type']
    if not all(col in df_eventos.columns for col in required_cols):
        print(f"  Faltan columnas necesarias para procesar zonas: {required_cols}")
        return pd.DataFrame()
    
    # Usar la columna de tipo adecuada
    type_col = 'type_nombre' if 'type_nombre' in df_eventos.columns else 'type'
    
    # Identificar pases progresivos (avance vertical significativo)
    df_eventos['es_pase_progresivo'] = False
    
    # Filtrar pases
    mask_pases = df_eventos[type_col].str.contains('Pass', na=False)
    
    # Determinar si son progresivos
    if 'endX' in df_eventos.columns:
        pases_con_destino = mask_pases & ~df_eventos['endX'].isna()
        if pases_con_destino.any():
            df_eventos.loc[pases_con_destino, 'es_pase_progresivo'] = (
                df_eventos.loc[pases_con_destino, 'endX'] - 
                df_eventos.loc[pases_con_destino, 'x'] > 10
            )
    
    # Identificar pases largos
    df_eventos['es_pase_largo'] = False
    df_eventos['distancia_pase'] = np.nan
    
    if all(col in df_eventos.columns for col in ['endX', 'endY']):
        mask_coord_completas = mask_pases & ~df_eventos['endX'].isna() & ~df_eventos['endY'].isna()
        
        if mask_coord_completas.any():
            df_eventos.loc[mask_coord_completas, 'distancia_pase'] = np.sqrt(
                (df_eventos.loc[mask_coord_completas, 'endX'] - df_eventos.loc[mask_coord_completas, 'x'])**2 +
                (df_eventos.loc[mask_coord_completas, 'endY'] - df_eventos.loc[mask_coord_completas, 'y'])**2
            )
            
            # Marcar pases largos
            df_eventos.loc[mask_coord_completas & (df_eventos['distancia_pase'] > 30), 'es_pase_largo'] = True
    
    # Identificar recuperaciones
    recuperaciones = ['Interception', 'BallRecovery', 'Tackle', 'BlockedPass', 'Challenge']
    df_eventos['es_recuperacion'] = df_eventos[type_col].isin(recuperaciones)
    
    # Identificar acciones defensivas
    acciones_defensivas = recuperaciones + ['Clearance', 'Block', 'Save', 'Claim']
    df_eventos['es_accion_defensiva'] = df_eventos[type_col].isin(acciones_defensivas)
    
    # Identificar acciones en área propia
    df_eventos['en_area_propia'] = df_eventos['x'] <= 20
    
    # Identificar secuencias de posesión
    recuentos_secuencias = identificar_secuencias_posesion(df_eventos)
    
    # Generar estadísticas espaciales por jornada
    df_zonas = generar_stats_espaciales(df_eventos, recuentos_secuencias)
    
    return df_zonas

def identificar_secuencias_posesion(df_eventos: pd.DataFrame) -> Dict[int, Dict[str, int]]:
    """
    Identifica secuencias de posesión en los eventos y devuelve recuentos.
    
    Args:
        df_eventos (pd.DataFrame): DataFrame con eventos
    
    Returns:
        dict: Diccionario con recuentos de secuencias por jornada
    """
    # Diccionario para almacenar recuentos
    recuentos = {}
    
    # Verificar que existan las columnas necesarias
    if df_eventos.empty or 'jornada' not in df_eventos.columns:
        print("  No hay datos suficientes para identificar secuencias")
        return recuentos
    
    # Determinar qué columnas usar
    minute_col = 'minute' if 'minute' in df_eventos.columns else 'expandedMinute' if 'expandedMinute' in df_eventos.columns else None
    second_col = 'second' if 'second' in df_eventos.columns else None
    type_col = 'type_nombre' if 'type_nombre' in df_eventos.columns else 'type'
    
    if not all([minute_col, type_col]):
        print("  Faltan columnas necesarias para identificar secuencias")
        return recuentos
    
    # Proceso por jornada
    for jornada in df_eventos['jornada'].unique():
        print(f"  Procesando secuencias en jornada {jornada}...")
        
        df_jornada = df_eventos[df_eventos['jornada'] == jornada].copy()
        
        # Ordenar por tiempo
        if second_col and second_col in df_jornada.columns:
            df_jornada = df_jornada.sort_values([minute_col, second_col])
        else:
            df_jornada = df_jornada.sort_values(minute_col)
        
        # Variables para seguimiento
        id_secuencia_actual = 1
        secuencias_largas = 0
        total_secuencias = 0
        pases_en_secuencia_actual = 0
        
        # Eventos que llevan a una interrupción
        eventos_quiebre = ['Interception', 'Foul', 'OffsidePass', 'Offside', 'Save', 'Goal', 'Clearance']
        eventos_secuencia = ['Pass']
        
        # Iterar ordenadamente
        prev_idx = None
        
        for idx, row in df_jornada.iterrows():
            es_evento_secuencia = type_col in row and isinstance(row[type_col], str) and any(e in row[type_col] for e in eventos_secuencia)
            
            # Determinar si hay quiebre o gap de tiempo
            hay_quiebre = type_col in row and isinstance(row[type_col], str) and any(e in row[type_col] for e in eventos_quiebre)
            
            gap_tiempo = False
            if prev_idx is not None and minute_col in row and minute_col in df_jornada.loc[prev_idx]:
                mismo_minuto = df_jornada.loc[prev_idx, minute_col] == row[minute_col]
                if mismo_minuto and second_col in row and second_col in df_jornada.loc[prev_idx]:
                    # Si hay más de 10 segundos entre eventos, considerar gap de tiempo
                    gap_tiempo = row[second_col] - df_jornada.loc[prev_idx, second_col] > 10
                elif not mismo_minuto:
                    # Si cambia el minuto, verificar si es consecutivo o hay un gap mayor
                    gap_tiempo = row[minute_col] - df_jornada.loc[prev_idx, minute_col] > 1
            
            # Si hay quiebre o gap de tiempo, finalizar secuencia actual si existía
            if hay_quiebre or gap_tiempo:
                if pases_en_secuencia_actual > 0:
                    total_secuencias += 1
                    if pases_en_secuencia_actual >= 8:  # Secuencia larga: 8+ pases
                        secuencias_largas += 1
                
                # Iniciar nueva secuencia
                id_secuencia_actual += 1
                pases_en_secuencia_actual = 1 if es_evento_secuencia else 0
            else:
                # Continuar secuencia actual
                if es_evento_secuencia:
                    pases_en_secuencia_actual += 1
            
            prev_idx = idx
        
        # Procesar la última secuencia si quedó activa
        if pases_en_secuencia_actual > 0:
            total_secuencias += 1
            if pases_en_secuencia_actual >= 8:
                secuencias_largas += 1
        
        # Guardar recuentos para esta jornada
        recuentos[jornada] = {
            'secuencias_largas': secuencias_largas, 
            'total_secuencias': total_secuencias
        }
    
    return recuentos

def generar_stats_espaciales(df_eventos: pd.DataFrame, recuentos_secuencias: Dict[int, Dict[str, int]]) -> pd.DataFrame:
    """
    Genera estadísticas espaciales agregadas por jornada.
    
    Args:
        df_eventos (pd.DataFrame): DataFrame con eventos clasificados por zonas
        recuentos_secuencias (dict): Recuentos de secuencias por jornada
    
    Returns:
        pd.DataFrame: DataFrame con estadísticas espaciales por jornada
    """
    # DataFrame para almacenar estadísticas
    stats = pd.DataFrame()
    
    # Verificar si hay datos suficientes
    if df_eventos.empty:
        print("  No hay eventos para generar estadísticas espaciales")
        return stats
    
    # Determinar qué columna de tipo usar
    type_col = 'type_nombre' if 'type_nombre' in df_eventos.columns else 'type'
    
    # Procesar por jornada
    for jornada in df_eventos['jornada'].unique():
        df_jornada = df_eventos[df_eventos['jornada'] == jornada].copy()
        
        # Estadísticas básicas
        total_pases = df_jornada[df_jornada[type_col].str.contains('Pass', na=False)].shape[0]
        pases_progresivos = df_jornada['es_pase_progresivo'].sum()
        
        # Conteo de pases por zona
        if 'zona_completa' in df_jornada.columns:
            pases_por_zona = df_jornada[df_jornada[type_col].str.contains('Pass', na=False)].groupby('zona_completa').size()
        else:
            pases_por_zona = pd.Series(dtype=int)
        
        # Identificar saques de puerta para análisis de juego directo
        saques_puerta = df_jornada[df_jornada[type_col].str.contains('GoalKick', na=False)]
        saques_puerta_totales = saques_puerta.shape[0]
        saques_puerta_directos = saques_puerta[saques_puerta['es_pase_largo']].shape[0] if saques_puerta.shape[0] > 0 else 0
        
        # Pases desde el primer tercio (juego directo)
        if 'zona_vertical' in df_jornada.columns:
            pases_primer_tercio = df_jornada[(df_jornada[type_col].str.contains('Pass|GoalKick', na=False)) & 
                                            (df_jornada['zona_vertical'] == 'Zona1')]
            
            pases_largos_primer_tercio = pases_primer_tercio[pases_primer_tercio['es_pase_largo']].shape[0]
            pases_totales_primer_tercio = pases_primer_tercio.shape[0]
        else:
            pases_largos_primer_tercio = 0
            pases_totales_primer_tercio = 0
        
        # Recuperaciones
        total_recuperaciones = df_jornada['es_recuperacion'].sum()
        recuperaciones_campo_contrario = df_jornada[df_jornada['es_recuperacion'] & (df_jornada['x'] > 50)].shape[0]
        
        # Secuencias (desde el diccionario)
        secuencias_largas = recuentos_secuencias.get(jornada, {}).get('secuencias_largas', 0)
        total_secuencias = recuentos_secuencias.get(jornada, {}).get('total_secuencias', 0)
        
        # Acciones defensivas
        acciones_defensivas = df_jornada['es_accion_defensiva'].sum()
        acciones_defensivas_area = df_jornada[df_jornada['es_accion_defensiva'] & df_jornada['en_area_propia']].shape[0]
        
        # Acciones de ataque por pasillo
        if 'zona_vertical' in df_jornada.columns and 'zona_horizontal' in df_jornada.columns:
            ataques_total = df_jornada[(df_jornada['zona_vertical'] == 'Zona3') & 
                                      (df_jornada[type_col].str.contains('Pass|Shot|Goal', na=False))].shape[0]
            
            ataques_banda = df_jornada[(df_jornada['zona_vertical'] == 'Zona3') & 
                                      (df_jornada['zona_horizontal'].isin(['Derecho', 'Izquierdo'])) &
                                      (df_jornada[type_col].str.contains('Pass|Shot|Goal', na=False))].shape[0]
        else:
            ataques_total = 0
            ataques_banda = 0
        
        # Diccionario con estadísticas para esta jornada
        stats_jornada = {
            'jornada': jornada,
            'total_pases_eventos': total_pases,
            'pases_progresivos': pases_progresivos,
            'total_recuperaciones': total_recuperaciones,
            'recuperaciones_campo_contrario': recuperaciones_campo_contrario,
            'secuencias_largas': secuencias_largas,
            'total_secuencias': total_secuencias,
            'acciones_defensivas': acciones_defensivas,
            'acciones_defensivas_area': acciones_defensivas_area,
            'ataques_total': ataques_total,
            'ataques_banda': ataques_banda,
            'saques_puerta_totales': saques_puerta_totales,
            'saques_puerta_directos': saques_puerta_directos,
            'pases_largos_primer_tercio': pases_largos_primer_tercio,
            'pases_totales_primer_tercio': pases_totales_primer_tercio
        }
        
        # Añadir conteos de pases por zona
        zonas_esperadas = ['Zona1_Derecho', 'Zona1_Central', 'Zona1_Izquierdo',
                         'Zona2_Derecho', 'Zona2_Central', 'Zona2_Izquierdo',
                         'Zona3_Derecho', 'Zona3_Central', 'Zona3_Izquierdo']
        
        for zona in zonas_esperadas:
            stats_jornada[f'pases_{zona}'] = pases_por_zona.get(zona, 0)
        
        # Añadir al DataFrame de estadísticas
        stats = pd.concat([stats, pd.DataFrame([stats_jornada])], ignore_index=True)
    
    return stats

def calcular_indices_basicos(df_master: pd.DataFrame, df_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula índices tácticos básicos a partir de estadísticas generales.
    
    Args:
        df_master (pd.DataFrame): DataFrame con datos maestros de partidos
        df_stats (pd.DataFrame): DataFrame con estadísticas por partido
    
    Returns:
        pd.DataFrame: DataFrame con índices básicos calculados
    """
    
    # DataFrame para índices
    df_indices = pd.DataFrame()
    
    # Filtrar registros con datos válidos
    df_master_filtrado = df_master[~df_master['goles_a_favor'].isna()].copy()
    df_stats_filtrado = df_stats.copy()
    
    # Asegurar que las columnas de jornada sean del mismo tipo
    if 'jornada_num' in df_master_filtrado.columns and 'jornada' in df_stats_filtrado.columns:
        df_master_filtrado['jornada_num'] = df_master_filtrado['jornada_num'].astype(int)
        df_stats_filtrado['jornada'] = df_stats_filtrado['jornada'].astype(int)
        
        # Combinar DataFrames
        df_combinado = pd.merge(
            df_stats_filtrado, 
            df_master_filtrado[['jornada_num', 'goles_a_favor', 'goles_en_contra', 'resultado_tipo']], 
            left_on='jornada', 
            right_on='jornada_num', 
            how='inner'
        )
        
        # 1. Índice de Iniciativa de Juego (IIJ)
        if 'tiros_recibidos' in df_combinado.columns:
            df_combinado['IIJ'] = (
                df_combinado['posesion'] * 0.5 + 
                ((df_combinado['tiros_totales'] + df_combinado['goles_a_favor']) - 
                (df_combinado['tiros_recibidos'] + df_combinado['goles_en_contra'])) * 1.0 +
                (df_combinado['toques_area_rival'] / 5) * 0.5
            )
        elif 'tiros_bloqueados' in df_combinado.columns and 'paradas_portero' in df_combinado.columns:
            # Aproximación si no tenemos 'tiros_recibidos'
            df_combinado['tiros_recibidos_aprox'] = df_combinado['tiros_bloqueados'] + df_combinado['paradas_portero']
            df_combinado['IIJ'] = (
                df_combinado['posesion'] * 0.5 + 
                ((df_combinado['tiros_totales'] + df_combinado['goles_a_favor']) - 
                (df_combinado['tiros_recibidos_aprox'] + df_combinado['goles_en_contra'])) * 1.0 +
                (df_combinado['toques_area_rival'] / 5) * 0.5
            )
        else:
            # Versión simplificada si faltan columnas
            df_combinado['IIJ'] = df_combinado['posesion'] * 0.5 + df_combinado['tiros_totales'] * 0.5
        
        # 2. Índice de Volumen de Juego Ofensivo (IVJO)
        if all(col in df_combinado.columns for col in ['pases', 'tiros_totales', 'toques_area_rival']):
            df_combinado['IVJO'] = (
                df_combinado['pases'] * 0.4 + 
                df_combinado['tiros_totales'] * 2.0 + 
                df_combinado['toques_area_rival'] * 1.5
            )
            
            # Añadir componente xG si está disponible
            if 'xG' in df_combinado.columns:
                df_combinado['IVJO'] += df_combinado['xG'] * 10
        
        # 3. Índice de Eficacia en Construcción (EC)
        if all(col in df_combinado.columns for col in ['tiros_totales', 'goles_a_favor', 'pases']):
            df_combinado['EC'] = (df_combinado['tiros_totales'] + df_combinado['goles_a_favor']) / df_combinado['pases'] * 100
        
        # 4. Índice de Eficacia en Finalización (EF)
        if all(col in df_combinado.columns for col in ['goles_a_favor', 'tiros_totales', 'tiros_palos']):
            denominador_ef = df_combinado['tiros_totales'] + df_combinado['tiros_palos'] + df_combinado['goles_a_favor']
            # Evitar división por cero
            df_combinado['EF'] = np.where(
                denominador_ef > 0,
                (df_combinado['goles_a_favor'] * 100) / denominador_ef,
                0
            )
        
        # 8. Índice de Juego Directo (IJD)
        if all(col in df_combinado.columns for col in ['pases_largos_primer_tercio', 'pases_totales_primer_tercio', 'saques_puerta_directos', 'saques_puerta_totales']):
            # Componente de saques de puerta (30%)
            componente_saques = np.where(
                df_combinado['saques_puerta_totales'] > 0,
                (df_combinado['saques_puerta_directos'] / df_combinado['saques_puerta_totales']) * 100,
                0
            ) * 0.3
            
            # Componente de pases largos general (30%)
            if 'pases_largos_precisos' in df_combinado.columns and 'pases_precisos' in df_combinado.columns:
                componente_pases_general = (df_combinado['pases_largos_precisos'] / df_combinado['pases_precisos']) * 100 * 0.3
            else:
                componente_pases_general = 0
            
            # Componente de pases largos desde primer tercio (40%)
            componente_pases_primer_tercio = np.where(
                df_combinado['pases_totales_primer_tercio'] > 0,
                (df_combinado['pases_largos_primer_tercio'] / df_combinado['pases_totales_primer_tercio']) * 100,
                0
            ) * 0.4
            
            # Combinar componentes
            df_combinado['IJD'] = componente_saques + componente_pases_general + componente_pases_primer_tercio
        elif 'pases_largos_precisos' in df_combinado.columns and 'pases_precisos' in df_combinado.columns:
            # Versión básica si faltan datos
            df_combinado['IJD'] = (df_combinado['pases_largos_precisos'] / df_combinado['pases_precisos']) * 100
        
        # Seleccionar columnas para el DataFrame de índices
        indices_basicos = ['jornada', 'IIJ', 'IVJO', 'EC', 'EF', 'IJD']
        indices_disponibles = [col for col in indices_basicos if col in df_combinado.columns]
        
        if indices_disponibles:
            df_indices = df_combinado[indices_disponibles].copy()
    else:
        print("  ADVERTENCIA: No se pueden calcular índices básicos (faltan columnas de jornada)")

    return df_indices

def calcular_indices_avanzados(df_eventos: pd.DataFrame, df_zonas: pd.DataFrame, df_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula índices tácticos avanzados que requieren análisis de eventos.
    
    Args:
        df_eventos (pd.DataFrame): DataFrame con eventos detallados
        df_zonas (pd.DataFrame): DataFrame con estadísticas espaciales
        df_stats (pd.DataFrame): DataFrame con estadísticas generales
    
    Returns:
        pd.DataFrame: DataFrame con índices avanzados calculados
    """
    
    # DataFrame para índices avanzados
    df_indices_avanzados = pd.DataFrame()
    
    # Verificar si hay datos suficientes
    if df_zonas.empty and df_stats.empty:
        print("  No hay datos suficientes para calcular índices avanzados")
        return df_indices_avanzados
    
    # Combinar DataFrames si es posible
    if not df_zonas.empty and 'jornada' in df_zonas.columns and 'jornada' in df_stats.columns:
        df_combinado = pd.merge(df_zonas, df_stats, on='jornada', how='inner')
    elif not df_zonas.empty:
        df_combinado = df_zonas.copy()
    else:
        df_combinado = df_stats.copy()
    
    # 5. Índice de Eficacia Recuperadora (IER)
    if all(col in df_combinado.columns for col in ['total_recuperaciones', 'entradas_exitosas', 'interceptaciones', 'duelos_ganados']):
        # Calcular unidades defensivas
        df_combinado['unidades_defensivas'] = (
            df_combinado['entradas_exitosas'] + 
            df_combinado['interceptaciones'] + 
            df_combinado['duelos_ganados']
        )
        
        # Calcular IER (evitando división por cero)
        df_combinado['IER'] = np.where(
            df_combinado['unidades_defensivas'] > 0,
            (df_combinado['total_recuperaciones'] / df_combinado['unidades_defensivas']) * 100,
            0
        )
    
    # 6. Índice de Verticalidad (IV)
    if all(col in df_combinado.columns for col in ['pases_progresivos', 'pases_precisos', 'toques_area_rival', 'pases']):
        df_combinado['IV'] = (
            (df_combinado['pases_progresivos'] / df_combinado['pases_precisos']) * 70 +
            (df_combinado['toques_area_rival'] / df_combinado['pases']) * 30
        )
    
    # 7. Índice de Presión Alta (IPA)
    if all(col in df_combinado.columns for col in ['total_recuperaciones', 'recuperaciones_campo_contrario', 'duelos_ganados', 'entradas_exitosas', 'acciones_defensivas', 'interceptaciones']):
        df_combinado['IPA'] = (
            np.where(
                df_combinado['total_recuperaciones'] > 0,
                (df_combinado['recuperaciones_campo_contrario'] / df_combinado['total_recuperaciones']) * 70,
                0
            ) +
            np.where(
                df_combinado['duelos_ganados'] > 0,
                (df_combinado['entradas_exitosas'] / df_combinado['duelos_ganados']) * 10,
                0
            ) +
            np.where(
                df_combinado['acciones_defensivas'] > 0,
                (df_combinado['interceptaciones'] / df_combinado['acciones_defensivas']) * 20,
                0
            )
        )
    
    # 9. Índice de Amplitud (IA)
    if all(col in df_combinado.columns for col in ['ataques_banda', 'ataques_total']):
        df_combinado['IA'] = np.where(
            df_combinado['ataques_total'] > 0,
            (df_combinado['ataques_banda'] / df_combinado['ataques_total']) * 100,
            0
        )
    
    # 10. Índice de Densidad Defensiva (IDD)
    if all(col in df_combinado.columns for col in ['acciones_defensivas_area', 'acciones_defensivas']):
        df_combinado['IDD'] = np.where(
            df_combinado['acciones_defensivas'] > 0,
            (df_combinado['acciones_defensivas_area'] / df_combinado['acciones_defensivas']) * 100,
            0
        )
    
    # 11. Índice de Complejidad de Juego (ICJ)
    if all(col in df_combinado.columns for col in ['secuencias_largas', 'total_secuencias', 'IV', 'pct_pases_precisos']):
        # Proporción de secuencias largas (peso 60%)
        proporcion_secuencias = np.where(
            df_combinado['total_secuencias'] > 0,
            (df_combinado['secuencias_largas'] / df_combinado['total_secuencias']),
            0
        ) * 0.6
        
        # Factor de precisión de pases (peso 40%)
        factor_precision = (df_combinado['pct_pases_precisos'] / 100) * 0.4
        
        # Calcular ICJ
        df_combinado['ICJ'] = (proporcion_secuencias + factor_precision) * (df_combinado['IV'] + 50)
    elif all(col in df_combinado.columns for col in ['secuencias_largas', 'total_secuencias', 'IV']):
        # Versión simplificada si falta la precisión de pases
        proporcion_secuencias = np.where(
            df_combinado['total_secuencias'] > 0,
            (df_combinado['secuencias_largas'] / df_combinado['total_secuencias']),
            0
        )
        
        df_combinado['ICJ'] = proporcion_secuencias * (df_combinado['IV'] + 50)
    else:
        print("  No se pudo calcular ICJ: faltan columnas necesarias")
    
    # Seleccionar columnas para el DataFrame final
    indices_avanzados = ['jornada', 'IER', 'IV', 'IPA', 'IA', 'IDD', 'ICJ']
    indices_disponibles = [col for col in indices_avanzados if col in df_combinado.columns]
    
    if indices_disponibles:
        df_indices_avanzados = df_combinado[indices_disponibles].copy()

    return df_indices_avanzados
    
def calcular_indices_especializados(df_combinado: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula índices tácticos especializados para análisis más detallado.
    
    Args:
        df_combinado (pd.DataFrame): DataFrame con estadísticas y otros índices ya calculados
    
    Returns:
        pd.DataFrame: DataFrame con índices especializados calculados
    """
    # DataFrame para índices especializados
    df_indices_especializados = pd.DataFrame()
    
    # Verificar si hay datos suficientes
    if df_combinado.empty or 'jornada' not in df_combinado.columns:
        print("  No hay datos suficientes para calcular índices especializados")
        return df_indices_especializados
    
    # Copia del DataFrame combinado
    df_esp = df_combinado.copy()
    
    # 1. Índice de Eficiencia Ofensiva (IEO)
    if all(col in df_esp.columns for col in ['xG', 'tiros_totales']):
        df_esp['IEO'] = np.where(
            df_esp['tiros_totales'] > 0,
            (df_esp['xG'] / df_esp['tiros_totales']) * 100,
            0
        )
    
    # 2. Índice de Precisión Táctica (IPT)
    if all(col in df_esp.columns for col in ['pct_pases_precisos', 'ocasiones_claras', 'pases_campo_rival']):
        df_esp['IPT'] = (
            df_esp['pct_pases_precisos'] * 0.6 + 
            np.where(
                df_esp['pases_campo_rival'] > 0,
                (df_esp['ocasiones_claras'] / df_esp['pases_campo_rival'] * 100) * 0.4,
                0
            )
        )
    
    # 3. Índice de Amenaza por Banda (IAB)
    zonas_completas_disponibles = True
    
    # Verificar qué formato de nombres de columna está disponible
    nombres_completos = ['pases_Zona3_Derecho', 'pases_Zona3_Izquierdo', 'pases_Zona3_Central']
    nombres_abreviados = ['pases_Zona3_drch', 'pases_Zona3_izq', 'pases_Zona3_ctrl']
    
    if all(col in df_esp.columns for col in nombres_completos):
        
        suma_pases_banda = df_esp['pases_Zona3_Derecho'] + df_esp['pases_Zona3_Izquierdo']
        suma_pases_ctrl = df_esp['pases_Zona3_Central']
    elif all(col in df_esp.columns for col in nombres_abreviados):
        
        suma_pases_banda = df_esp['pases_Zona3_drch'] + df_esp['pases_Zona3_izq']
        suma_pases_ctrl = df_esp['pases_Zona3_ctrl']
    else:
        print("  No se encontraron columnas necesarias para el cálculo de IAB")
        zonas_completas_disponibles = False
    
    # Calcular IAB si tenemos las columnas necesarias
    if zonas_completas_disponibles:
        df_esp['IAB'] = (suma_pases_banda / (suma_pases_ctrl + 1)) * 100
    else:
        # Usar valor neutral si no se puede calcular
        df_esp['IAB'] = 50
    
    # 4. Índice de Control de Transiciones (ICT)
    if all(col in df_esp.columns for col in ['faltas_cometidas', 'interceptaciones', 'total_recuperaciones', 'pct_duelos_suelo_ganados']):
        df_esp['ICT'] = (
            (1 - np.minimum(df_esp['faltas_cometidas'] / 15, 1)) * 40 +
            np.where(
                df_esp['total_recuperaciones'] > 0,
                (df_esp['interceptaciones'] / df_esp['total_recuperaciones']) * 30,
                0
            ) +
            (df_esp['pct_duelos_suelo_ganados'] / 100) * 30
        )
    
    # 5. Índice de Calidad de la Posesión (ICP)
    if all(col in df_esp.columns for col in ['xG', 'posesion', 'ocasiones_claras', 'pases']):
        df_esp['ICP'] = (
            np.where(
                df_esp['posesion'] > 0,
                (df_esp['xG'] / df_esp['posesion']) * 70,
                0
            ) +
            np.where(
                df_esp['pases'] > 0,
                (df_esp['ocasiones_claras'] / df_esp['pases']) * 30,
                0
            )
        ) * 100
    
    # Seleccionar columnas de índices especializados
    indices_especializados = ['jornada']
    for indice in ['IEO', 'IPT', 'IAB', 'ICT', 'ICP']:
        if indice in df_esp.columns:
            indices_especializados.append(indice)
    
    df_indices_especializados = df_esp[indices_especializados].copy()
    
    return df_indices_especializados

def calcular_variables_contexto(df_master: pd.DataFrame, df_equipos: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula variables de contexto para cada partido.
    
    Args:
        df_master (pd.DataFrame): DataFrame con datos maestros de partidos
        df_equipos (pd.DataFrame): DataFrame con información de equipos
    
    Returns:
        pd.DataFrame: DataFrame con variables de contexto
    """
    
    # Crear DataFrame para variables de contexto
    df_contexto = pd.DataFrame()
    
    # Verificar si hay datos suficientes
    if df_master.empty:
        print("  No hay datos maestros para calcular variables de contexto")
        return df_contexto
    
    # Filtrar partidos disputados
    df_master_filtrado = df_master[~df_master['goles_a_favor'].isna()].copy()
    
    # Verificar si hay partidos registrados
    if df_master_filtrado.empty:
        print("  No hay partidos disputados registrados")
        return df_contexto
    
    # Variables de expulsión temprana
    if all(col in df_master_filtrado.columns for col in ['min_expulsion_propia', 'min_expulsion_rival']):
        df_master_filtrado['expulsion_temprana'] = (
            (df_master_filtrado['min_expulsion_propia'] <= 30) & (df_master_filtrado['min_expulsion_propia'] > 0) |
            (df_master_filtrado['min_expulsion_rival'] <= 30) & (df_master_filtrado['min_expulsion_rival'] > 0)
        ).astype(int)
    
    # Clasificar rivales por categoría
    if 'rival' in df_master_filtrado.columns:
        # Definir equipos top, medio y bajo
        equipos_top = ['Real Madrid', 'FC Barcelona', 'Atletico de Madrid']
        equipos_bajo = ['Real Valladolid', 'CD Leganes', 'Deportivo Alaves', 'RCD Espanyol', 'RCD Mallorca', 'UD Las Palmas']
        
        # Categorizar rivales
        df_master_filtrado['rival_categoria'] = 'medio'
        df_master_filtrado.loc[df_master_filtrado['rival'].isin(equipos_top), 'rival_categoria'] = 'top'
        df_master_filtrado.loc[df_master_filtrado['rival'].isin(equipos_bajo), 'rival_categoria'] = 'bajo'
        
        # Convertir a numérico
        categoria_map = {'bajo': 0, 'medio': 1, 'top': 2}
        df_master_filtrado['rival_categoria_num'] = df_master_filtrado['rival_categoria'].map(categoria_map)
    
    # Añadir si es local o visitante
    if 'local_visitante' in df_master_filtrado.columns:
        df_master_filtrado['es_local'] = (df_master_filtrado['local_visitante'] == 'L').astype(int)
    
    # Seleccionar columnas relevantes
    cols_contexto = ['jornada_num']
    
    # Añadir columnas si existen
    for col in ['expulsion_temprana', 'gol_temprano_favor', 'gol_temprano_contra',
                'rival_categoria', 'rival_categoria_num', 'es_local', 'resultado_tipo']:
        if col in df_master_filtrado.columns:
            cols_contexto.append(col)
    
    # Crear DataFrame con variables seleccionadas
    if len(cols_contexto) > 1:  # Si hay más de solo jornada_num
        df_contexto = df_master_filtrado[cols_contexto].copy()
        df_contexto.rename(columns={'jornada_num': 'jornada'}, inplace=True)
    
    return df_contexto
    
def categorizar_estilos(df_indices: pd.DataFrame) -> pd.DataFrame:
    """
    Categoriza estilos de juego basados en los índices calculados.
    
    Args:
        df_indices (pd.DataFrame): DataFrame con índices tácticos
    
    Returns:
        pd.DataFrame: DataFrame con categorías de estilo de juego añadidas
    """
    
    # Crear copia para no modificar el original
    df = df_indices.copy()
    
    # 1. Orientación general
    if 'IIJ' in df.columns and 'IEO' in df.columns:
        df['orientacion_general'] = 'Equilibrado'
        df.loc[(df['IIJ'] > 55) & (df['IEO'] > 12), 'orientacion_general'] = 'Ofensivo de calidad'
        df.loc[(df['IIJ'] > 55) & (df['IEO'] <= 12), 'orientacion_general'] = 'Ofensivo en cantidad'
        df.loc[(df['IIJ'] < 45) & (df['IER'] > 45), 'orientacion_general'] = 'Defensivo activo'
        df.loc[(df['IIJ'] < 45) & (df['IER'] <= 45), 'orientacion_general'] = 'Defensivo pasivo'
    elif 'IIJ' in df.columns:
        df['orientacion_general'] = 'Equilibrado'
        df.loc[df['IIJ'] > 55, 'orientacion_general'] = 'Ofensivo'
        df.loc[df['IIJ'] < 45, 'orientacion_general'] = 'Defensivo'
    
    # 2. Fase ofensiva
    if all(col in df.columns for col in ['IV', 'IJD', 'IPT']):
        df['fase_ofensiva'] = 'Posicional'
        df.loc[(df['IV'] > 25) & (df['IJD'] < 20), 'fase_ofensiva'] = 'Vertical-Preciso'
        df.loc[(df['IV'] > 20) & (df['IJD'] < 20) & (df['IPT'] <= 65), 'fase_ofensiva'] = 'Vertical'
        df.loc[(df['IJD'] > 15) & (df['IPT'] > 60), 'fase_ofensiva'] = 'Directo-Efectivo'
        df.loc[(df['IJD'] > 12) & (df['IPT'] <= 60), 'fase_ofensiva'] = 'Directo'
    elif all(col in df.columns for col in ['IV', 'IJD']):
        df['fase_ofensiva'] = 'Posicional'
        df.loc[(df['IV'] > 25) & (df['IJD'] < 20), 'fase_ofensiva'] = 'Vertical'  
        df.loc[df['IJD'] > 12, 'fase_ofensiva'] = 'Directo'
    
    # 3. Patrón de ataque
    if 'IAB' in df.columns:
        # Estadísticas descriptivas para entender la distribución
        media_iab = df['IAB'].mean()
        mediana_iab = df['IAB'].median()
        std_iab = df['IAB'].std()
        
        # Uso de cuartiles para crear categorías más equilibradas
        q1_iab = df['IAB'].quantile(0.33)
        q2_iab = df['IAB'].quantile(0.67)
        
        df['patron_ataque'] = 'Equilibrio pasillos de ataque'
        df.loc[df['IAB'] > q2_iab, 'patron_ataque'] = 'Enfoque pasillos exteriores'
        df.loc[df['IAB'] < q1_iab, 'patron_ataque'] = 'Enfoque pasillo central'
    elif 'IA' in df.columns:
        # De nuevo uso cuartiles para crear categorías más equilibradas
        q1_ia = df['IA'].quantile(0.33)
        q2_ia = df['IA'].quantile(0.67)
        
        df['patron_ataque'] = 'Equilibrado'
        df.loc[df['IA'] > q2_ia, 'patron_ataque'] = 'Enfoque pasillos exteriores'
        df.loc[df['IA'] < q1_ia, 'patron_ataque'] = 'Enfoque pasillo central'
    
    # 4. Intensidad defensiva
    if all(col in df.columns for col in ['IPA', 'ICT']):
        # Del mismo modo uso cuartiles para mejor distribución
        q_alto = df['IPA'].quantile(0.7)
        q_bajo = df['IPA'].quantile(0.3)
        
        df['intensidad_defensiva'] = 'Moderada'
        df.loc[(df['IPA'] > q_alto) & (df['ICT'] > 50), 'intensidad_defensiva'] = 'Cierre de trayectorias en altura'
        df.loc[(df['IPA'] > q_alto) & (df['ICT'] <= 50), 'intensidad_defensiva'] = 'Presión alta agresiva'
        df.loc[(df['IPA'] < q_bajo) & (df['ICT'] > 50), 'intensidad_defensiva'] = 'Defensa pasiva organizada'
        df.loc[(df['IPA'] < q_bajo) & (df['ICT'] <= 50), 'intensidad_defensiva'] = 'Defensa pasiva reactiva'
    elif 'IPA' in df.columns:
        # Categorías más equilibradas con cuartiles
        q_alto = df['IPA'].quantile(0.7)
        q_bajo = df['IPA'].quantile(0.3)
        
        df['intensidad_defensiva'] = 'Moderada'
        df.loc[df['IPA'] > q_alto, 'intensidad_defensiva'] = 'Presión alta'
        df.loc[df['IPA'] < q_bajo, 'intensidad_defensiva'] = 'Defensa pasiva'
    
    # 5. Altura del bloque
    if all(col in df.columns for col in ['IDD', 'IPA']):
        # Categorías más equilibradas
        q_alto_idd = df['IDD'].quantile(0.7)
        q_alto_ipa = df['IPA'].quantile(0.7)
        
        df['altura_bloque'] = 'Medio'
        df.loc[df['IDD'] > q_alto_idd, 'altura_bloque'] = 'Bajo'
        df.loc[df['IPA'] > q_alto_ipa, 'altura_bloque'] = 'Alto'
    
    # 6. Tipo de transición
    if all(col in df.columns for col in ['ICJ', 'ICP']):
        # También con cuartiles, categorías más equilibradas
        q1_icj = df['ICJ'].quantile(0.25)
        q2_icj = df['ICJ'].quantile(0.5)
        q3_icj = df['ICJ'].quantile(0.75)
        
        q_medio_icp = df['ICP'].quantile(0.5)
        
        df['tipo_transicion'] = 'Convencional'
        df.loc[(df['ICJ'] > q3_icj) & (df['ICP'] > q_medio_icp), 'tipo_transicion'] = 'Elaborada efectiva'
        df.loc[(df['ICJ'] > q3_icj) & (df['ICP'] <= q_medio_icp), 'tipo_transicion'] = 'Elaborada ineficiente'
        df.loc[(df['ICJ'] <= q1_icj) & (df['ICP'] > q_medio_icp), 'tipo_transicion'] = 'Directa efectiva'
        df.loc[(df['ICJ'] <= q1_icj) & (df['ICP'] <= q_medio_icp), 'tipo_transicion'] = 'Directa ineficiente'
        df.loc[(df['ICJ'] > q1_icj) & (df['ICJ'] <= q3_icj), 'tipo_transicion'] = 'Contragolpe adaptativo'
    elif 'ICJ' in df.columns:
        # Categorización básica usando cuartiles
        q1_icj = df['ICJ'].quantile(0.25)
        q3_icj = df['ICJ'].quantile(0.75)
        
        df['tipo_transicion'] = 'Convencional'
        df.loc[df['ICJ'] > q3_icj, 'tipo_transicion'] = 'Elaborada'  
        df.loc[df['ICJ'] < q1_icj, 'tipo_transicion'] = 'Directa'
    
    # 7. Estilo de posesión (nueva categoría) 
    cols_needed = ['ICP', 'ICJ', 'posesion']
    has_cols = all(col in df.columns for col in cols_needed)
        
    if has_cols:
        
        df['estilo_posesion'] = 'Posesión funcional'
        df.loc[(df['ICP'] > 45) & (df['ICJ'] > 45), 'estilo_posesion'] = 'Posesión dominante'
        df.loc[(df['ICP'] < 35) & (df['posesion'] > 50), 'estilo_posesion'] = 'Posesión estéril'
        df.loc[(df['ICP'] > 45) & (df['posesion'] < 48), 'estilo_posesion'] = 'Posesión eficiente'
    else:
        print("  No se pudo crear estilo_posesion, faltan columnas.")
    
    print(f"  Categorías de estilo generadas: {[col for col in df.columns if col in ['orientacion_general', 'fase_ofensiva', 'patron_ataque', 'intensidad_defensiva', 'altura_bloque', 'tipo_transicion', 'estilo_posesion']]}")
    return df
