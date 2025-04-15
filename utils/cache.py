import streamlit as st
import pandas as pd
from pathlib import Path
import os
import json
from datetime import datetime, timedelta
import LanusStats as ls

# Función cacheada para obtener datos externos
@st.cache_data(ttl=3600)  # Caché durante 1 hora
def get_fotmob_data(match_id):
    """
    Función cacheada para obtener datos de FotMob
    
    Args:
        match_id: ID del partido en FotMob
        
    Returns:
        Datos del partido en formato JSON
    """
    try:
        # Intentar importar la biblioteca para FotMob
        try:
            
            fotmob = ls.FotMob()
            response = fotmob.request_match_details(match_id)
            return response.json()
        except ImportError:
            # Si la biblioteca no está disponible, devolver un diccionario vacío
            st.warning("Biblioteca LanusStats no disponible para obtener datos de FotMob")
            return {}
    except Exception as e:
        st.error(f"Error al obtener datos de FotMob: {str(e)}")
        return None

# Inicializar caché global si no existe
def init_cache():
    """Inicializa la caché global en session_state si no existe"""
    if 'global_cache' not in st.session_state:
        st.session_state.global_cache = {}

# Función genérica para obtener datos cacheados
def get_cached_data(key, fetch_function, *args, **kwargs):
    """
    Función genérica para obtener datos cacheados o recuperarlos si no existen
    
    Args:
        key: Clave única para los datos
        fetch_function: Función para obtener los datos si no están en caché
        *args, **kwargs: Argumentos para pasar a fetch_function
        
    Returns:
        Los datos cacheados o recuperados
    """
    # Asegurar que la caché está inicializada
    init_cache()
    
    # Comprobar si los datos están en caché
    if key in st.session_state.global_cache:
        return st.session_state.global_cache[key]
    
    # Si no están en caché, recuperarlos
    data = fetch_function(*args, **kwargs)
    
    # Guardar en caché
    st.session_state.global_cache[key] = data
    
    return data

# Función para limpiar la caché
def clear_cache(key=None):
    """
    Limpia la caché
    
    Args:
        key: Clave específica a limpiar (None para limpiar toda la caché)
    """
    init_cache()
    
    if key is None:
        st.session_state.global_cache = {}
    elif key in st.session_state.global_cache:
        del st.session_state.global_cache[key]

# Función para guardar caché en disco
def save_cache_to_disk(directory="cache"):
    """
    Guarda la caché actual en el disco
    
    Args:
        directory: Directorio donde guardar los archivos de caché
    """
    init_cache()
    
    # Crear directorio si no existe
    cache_dir = Path(directory)
    cache_dir.mkdir(exist_ok=True, parents=True)
    
    # Guardar cada elemento de la caché en un archivo separado
    for key, value in st.session_state.global_cache.items():
        # Crear un nombre de archivo seguro
        safe_key = "".join(c if c.isalnum() else "_" for c in key)
        file_path = cache_dir / f"{safe_key}.json"
        
        try:
            # Intentar serializar y guardar
            with open(file_path, "w") as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "data": value
                }, f)
        except Exception as e:
            print(f"Error al guardar caché para {key}: {e}")

# Función para cargar caché desde disco
def load_cache_from_disk(directory="cache", max_age_hours=24):
    """
    Carga la caché desde el disco
    
    Args:
        directory: Directorio donde están los archivos de caché
        max_age_hours: Edad máxima en horas para considerar válida la caché
    """
    init_cache()
    
    cache_dir = Path(directory)
    if not cache_dir.exists():
        return
    
    # Obtener la fecha límite
    max_age = datetime.now() - timedelta(hours=max_age_hours)
    
    # Cargar cada archivo de caché
    for file_path in cache_dir.glob("*.json"):
        try:
            with open(file_path, "r") as f:
                cache_data = json.load(f)
            
            # Verificar la antigüedad
            timestamp = datetime.fromisoformat(cache_data["timestamp"])
            if timestamp > max_age:
                # Extraer el nombre de la clave del nombre del archivo
                key = file_path.stem
                st.session_state.global_cache[key] = cache_data["data"]
        except Exception as e:
            print(f"Error al cargar caché desde {file_path}: {e}")

@st.cache_data(ttl=3600)
def read_dataframe_with_cache(file_path, **kwargs):
    """
    Lee un DataFrame desde un archivo con caché de Streamlit.
    
    Args:
        file_path (str): Ruta al archivo
        **kwargs: Argumentos adicionales para pd.read_csv
    
    Returns:
        pd.DataFrame: DataFrame leído
    """
    try:
        return pd.read_csv(file_path, **kwargs)
    except Exception as e:
        print(f"Error al leer archivo {file_path}: {e}")
        return pd.DataFrame()