import requests
import os
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
from pathlib import Path

def load_teams_mapping():
    """
    Carga el mapeo de equipos desde el CSV
    
    Returns:
        dict: Diccionario con el mapeo de nombres y rutas a escudos
    """
    try:
        # Ruta del archivo CSV
        csv_path = Path("data/raw/master/equipos_master_laliga.csv")
        
        # Intentar cargar el CSV
        if not csv_path.exists():
            print(f"Archivo CSV no encontrado en la ruta esperada: {csv_path}")
            return {}
        
        # Leer el CSV con delimitador punto y coma
        df_tm = pd.read_csv(csv_path, sep=";", dtype=str)
       
        # Crear diccionario de mapeo de equipos
        team_mapping = {}
        for _, row in df_tm.iterrows():
            nombre_csv = row['nombre']
            shortname = row['shortname'].lower()  
            logo_path = row['ruta_escudo'].strip() if pd.notna(row['ruta_escudo']) else ""  
            
            # Verificar que la ruta del escudo no esté vacía
            if not logo_path:
                print(f"❌ Ruta de escudo vacía para el equipo {nombre_csv}. Usando logo predeterminado.")
                logo_path = 'assets/images/logos/default_logo.png'  
            
            team_mapping[nombre_csv] = {
                'original_name': nombre_csv,
                'shortname': shortname,
                'logo_path': logo_path
            }
        
        return team_mapping
    
    except Exception as e:
        print(f"Error al cargar el mapeo de equipos: {e}")
        return {}

def find_closest_team_name(api_name, team_mapping):
    """
    Encuentra el nombre de equipo más cercano en el mapeo
    
    Args:
        api_name (str): Nombre del equipo desde la API
        team_mapping (dict): Diccionario de mapeo de equipos
    
    Returns:
        str: Nombre del equipo más cercano o el nombre original
    """

    if api_name in team_mapping:
        return api_name
    
    # Simplificar nombres para comparación
    simplified_api_name = api_name.lower().replace("club", "").replace("de", "").replace("cf", "").replace("fc", "").strip()
    
    # Mapeo para casos especiales
    special_cases = {
        'espanyol': 'RCD Espanyol de Barcelona',
        'español': 'RCD Espanyol de Barcelona',  # Caso específico para español
        'rcd espanyol': 'RCD Espanyol de Barcelona',
        'rcd español': 'RCD Espanyol de Barcelona',
        'leganés': 'CD Leganés',
        'cd leganés': 'CD Leganés',
        'alavés': 'Deportivo Alavés',
        'alavés': 'Deportivo Alavés',
        'deportivo alavés': 'Deportivo Alavés',
        'real sociedad': 'Real Sociedad de Fútbol',
        'real madrid': 'Real Madrid CF',
        'valladolid': 'Real Valladolid CF',
        'athletic': 'Athletic Club',
        'atleti': 'Atletico de Madrid',
        'barcelona': 'FC Barcelona',
        'getafe': 'Getafe CF',
        'mallorca': 'RCD Mallorca',
        'celta': 'RC Celta de Vigo',
        'girona': 'Girona FC'
        # Añadir más casos si es necesario
    }
    
    # Comprobar casos especiales
    for key, value in special_cases.items():
        if key in simplified_api_name:
            if value in team_mapping:
                return value
    
    # Buscar coincidencias parciales
    best_match = None
    best_score = 0
    
    for mapped_name, details in team_mapping.items():
        # Comparaciones sin considerar mayúsculas/minúsculas
        if (mapped_name.lower() in api_name.lower() or 
            api_name.lower() in mapped_name.lower() or
            details['original_name'].lower() in api_name.lower() or
            api_name.lower() in details['original_name'].lower()):
            # Calcular un "score" simple de coincidencia
            name_length = len(api_name)
            match_length = len(mapped_name)
            score = min(name_length, match_length) / max(name_length, match_length)
            
            if score > best_score:
                best_score = score
                best_match = mapped_name
    
    if best_match:
        return best_match
    
    return api_name


def fetch_matches(api_key):
    """
    Obtiene los partidos del Atlético de Madrid desde la API de football-data.org
    
    Args:
        api_key (str): Clave de API para football-data.org
        
    Returns:
        list: Lista de partidos o None si hay un error
    """
    headers = {'X-Auth-Token': api_key}
    url = 'http://api.football-data.org/v4/teams/78/matches'
    
    # Usar caché para evitar llamadas constantes a la API
    @st.cache_data(ttl=3600) 
    def _fetch_api_data(url, headers):
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                matches = response.json()['matches']

                # Verificar los equipos en la respuesta
                for match in matches:
                    home_team = match['homeTeam']['name']
                    away_team = match['awayTeam']['name']
                    
                return matches
            else:
                print(f"Error al obtener datos: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error al conectar con la API: {e}")
            return None
    
    return _fetch_api_data(url, headers)

def process_matches(matches, team_mapping=None):
    """
    Procesa los datos de partidos para el formato requerido
    
    Args:
        matches (list): Lista de partidos desde la API
        team_mapping (dict, optional): Mapeo de equipos precargado
        
    Returns:
        pd.DataFrame: DataFrame con los datos procesados
    """
    # Cargar el mapeo de equipos si no se ha proporcionado
    if team_mapping is None:
        team_mapping = load_teams_mapping()
    
    data = []
    cumulative_points = 0
    
    for match in matches:
        try:
            # Usar la función find_closest_team_name para mapear nombres
            home_team = find_closest_team_name(match['homeTeam']['name'], team_mapping)
            away_team = find_closest_team_name(match['awayTeam']['name'], team_mapping)

            if match['competition']['name'] == 'Primera Division':
                home_team_name = match['homeTeam']['name']
                is_home = home_team_name == 'Club Atlético de Madrid'
                opponent = match['awayTeam']['name'] if is_home else home_team_name
                
                if match['score']['winner']:
                    home_goals = match['score']['fullTime']['home']
                    away_goals = match['score']['fullTime']['away']
                    
                    # Mantener formato Local-Visitante para el marcador
                    score = f"{home_goals}-{away_goals}"
                    
                    if is_home:
                        points = 3 if home_goals > away_goals else (1 if home_goals == away_goals else 0)
                        result = 'W' if home_goals > away_goals else ('D' if home_goals == away_goals else 'L')
                    else:
                        points = 3 if away_goals > home_goals else (1 if home_goals == away_goals else 0)
                        result = 'W' if away_goals > home_goals else ('D' if home_goals == away_goals else 'L')
                    
                    cumulative_points += points
                    
                    data.append({
                        'date': datetime.strptime(match['utcDate'], '%Y-%m-%dT%H:%M:%SZ'),
                        'opponent': opponent,
                        'location': 'Local' if is_home else 'Visitante',
                        'result': result,
                        'points': points,
                        'cumulative_points': cumulative_points,
                        'score': score
                    })
        except Exception as e:
            print(f"Error al procesar partido: {e}")
            continue
    
    # Si no hay datos, devolver DataFrame vacío
    if not data:
        return pd.DataFrame()
        
    return pd.DataFrame(data)

def transform_dataframe(df_tm, team_mapping=None):
    """
    Realiza transformaciones adicionales al DataFrame
    
    Args:
        df_tm (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame transformado
    """
    # Si el DataFrame está vacío, devolver como está
    if df_tm.empty:
        return df_tm
        
    # Crear copia para no modificar el original
    df_tm_new = df_tm.copy()
    
    # Convertir fecha a solo día
    df_tm_new['date'] = pd.to_datetime(df_tm_new['date']).dt.strftime('%Y-%m-%d')
        
    # Crear columna de jornada
    df_tm_new['jornada'] = range(1, len(df_tm_new) + 1)
    
    # Cargar el mapeo real de equipos
    if team_mapping is None:
        team_mapping = load_teams_mapping()
    
    # Aplicar el mapeo a los nombres de los oponentes y agregar ruta al escudo
    df_tm_new['opponent_display'] = df_tm_new['opponent'].apply(
        lambda x: find_closest_team_name(x, team_mapping)
    )
    
    # Agregar ruta al escudo (con manejo de errores)
    df_tm_new['opponent_logo'] = df_tm_new['opponent_display'].apply(
        lambda x: team_mapping.get(x, {}).get('logo_path', None) if x in team_mapping else None
    )
    
    # Reordenar columnas
    try:
        df_tm_new = df_tm_new[['jornada', 'date', 'opponent', 'opponent_display', 'opponent_logo', 
                      'location', 'result', 'points', 'cumulative_points', 'score']]
    except KeyError:
        # Si hay algún problema con las columnas, mantener el orden actual
        pass
    
    return df_tm_new

def get_atletico_matches(api_key):
    """
    Función principal que combina todas las operaciones para obtener 
    los datos procesados de los partidos del Atlético
    
    Args:
        api_key (str): Clave de API para football-data.org
        
    Returns:
        pd.DataFrame: DataFrame final con todos los datos procesados
    """
    matches = fetch_matches(api_key)
    if matches:
        # Cargar el mapeo una sola vez
        team_mapping = load_teams_mapping()
                
        # Procesar usando el mismo mapeo
        df_tm = process_matches(matches, team_mapping)
        return transform_dataframe(df_tm, team_mapping)
    return None


# Función para obtener la API key de forma segura
def get_api_key():
    """
    Obtiene la API key de forma segura desde diferentes fuentes
    
    Returns:
        str: API key o None si no se encuentra
    """
    # Intentar obtener de Streamlit secrets
    try:
        return st.secrets["FOOTBALL_DATA_API_KEY"]
    except:
        pass
    
    # Intentar obtener de variables de entorno
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("FOOTBALL_DATA_API_KEY")
        if api_key:
            return api_key
    except:
        pass
    
    # Intentar obtener de config.toml
    try:
        config_path = Path("streamlit/secrets.toml")
        if config_path.exists():
            import toml
            config = toml.load(config_path)
            api_key = config.get("football_data_api_key", None)
            if api_key:
                return api_key
    except:
        pass
    
    # Si llegamos aquí, no se encontró la API key
    return None