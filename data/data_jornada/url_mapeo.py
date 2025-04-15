import streamlit as st  
import pandas as pd
from pathlib import Path

@st.cache_data(ttl=3600)
def load_partidos_master():
    """
    Carga el archivo maestro de partidos con IDs y URLs.
    
    Returns:
        pandas.DataFrame: DataFrame con información de partidos
    """
    file_path = Path("data/raw/master/partidos_master.csv")
    
    if not file_path.exists():
        st.error(f"Archivo no encontrado: {file_path}")  # Usamos st.error para visualizar el error en Streamlit
        return pd.DataFrame()
    
    try:
        
        df = pd.read_csv(file_path, sep=';')

        return df
    
    except Exception as e:

        st.error(f"Error al cargar el archivo maestro de partidos: {str(e)}")

        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_equipos_master():
    """
    Carga el archivo maestro de equipos con escudos y nombres.
    
    Returns:
        pandas.DataFrame: DataFrame con información de equipos
    """
    file_path = Path("data/raw/master/equipos_master_laliga.csv")
    
    if not file_path.exists():
        st.error(f"Archivo no encontrado: {file_path}")  # Usamos st.error para visualizar el error en Streamlit
        return pd.DataFrame()
    
    try:
        
        df = pd.read_csv(file_path, sep=';')
        
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo maestro de equipos: {e}")
        return pd.DataFrame()

def get_match_urls(match_id):
    """
    Obtiene las URLs asociadas a un ID de partido específico.
    
    Args:
        match_id (str): ID del partido
        
    Returns:
        dict: Diccionario con las URLs del partido
    """
    partidos_df = load_partidos_master()
    
    for id_col in ['id_whoscored', 'id_fotmob']:
        if id_col in partidos_df.columns and match_id in partidos_df[id_col].values:
            match_row = partidos_df[partidos_df[id_col] == match_id].iloc[0]
            urls = {}
            if 'url_whoscored' in match_row and pd.notna(match_row['url_whoscored']):
                urls['whoscored_url'] = match_row['url_whoscored']
            if 'url_fbref' in match_row and pd.notna(match_row['url_fbref']):
                urls['fbref_url'] = match_row['url_fbref']
            if 'id_fotmob' in match_row and pd.notna(match_row['id_fotmob']):
                urls['fotmob_id'] = match_row['id_fotmob']
            if 'id_understat' in match_row and pd.notna(match_row['id_understat']):
                urls['understat_id'] = match_row['id_understat']
                
            return urls
    
    return None

def get_match_by_jornada(jornada):
    """
    Obtiene la información de un partido por su jornada.
    
    Args:
        jornada (int o str): Número o etiqueta de jornada
        
    Returns:
        dict: Información del partido
    """
    partidos_df = load_partidos_master()
    
    # Buscar por número de jornada o por formato de jornada
    for jornada_col in ['Jornada', 'formato_jornada']:
        if jornada_col in partidos_df.columns:
            # Convertir jornada a string para comparar con formato_jornada
            jornada_str = str(jornada)
            
            # Si es un número y buscamos en Jornada, intentar matchear con el número
            if jornada_col == 'Jornada' and jornada_str.isdigit():
                jornada_int = int(jornada_str)
                match_row = partidos_df[partidos_df[jornada_col] == jornada_int]
            else:
                # Buscar coincidencias parciales en formato_jornada
                match_row = partidos_df[partidos_df[jornada_col].str.contains(jornada_str, na=False)]
            
            if not match_row.empty:
                return match_row.iloc[0].to_dict()
    
    return None

def get_partido_info(format_jornada):
    """
    Obtiene información detallada de un partido por su formato de jornada.
    
    Args:
        format_jornada (str): Formato de jornada (ej. "5ª ATM-VAL")
        
    Returns:
        dict: Información detallada del partido
    """
    partidos_df = load_partidos_master()
    equipos_df = load_equipos_master()
    
    if 'formato_jornada' not in partidos_df.columns:
        return None
    
    partido = partidos_df[partidos_df['formato_jornada'] == format_jornada]
    if partido.empty:
        return None
    
    partido_info = partido.iloc[0].to_dict()
    
    local_id = None
    visitante_id = None
    
    if 'equipo_local' in partido_info and 'shortname' in equipos_df.columns:
        local_name = partido_info['equipo_local']
        local_team = equipos_df[equipos_df['nombre'] == local_name]
        if not local_team.empty:
            local_id = local_team.iloc[0]['id_streamlit']
    
    if 'equipo_visitante' in partido_info and 'shortname' in equipos_df.columns:
        visitante_name = partido_info['equipo_visitante']
        visitante_team = equipos_df[equipos_df['nombre'] == visitante_name]
        if not visitante_team.empty:
            visitante_id = visitante_team.iloc[0]['id_streamlit']
    
    partido_info['local_id'] = local_id
    partido_info['visitante_id'] = visitante_id
    
    return partido_info

def get_available_jornadas():
    """
    Obtiene la lista de jornadas disponibles en formato legible.
    
    Returns:
        list: Lista de strings con formato de jornada
    """
    partidos_df = load_partidos_master()
    
    if 'formato_jornada' in partidos_df.columns:
        return partidos_df['formato_jornada'].tolist()
    elif 'Jornada' in partidos_df.columns:
        return [f"{j}ª Jornada" for j in partidos_df['Jornada'].tolist()]
    
    return []