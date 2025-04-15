import pandas as pd
import numpy as np  
import os
from pathlib import Path
import traceback
from unidecode import unidecode
import json
import re
import glob
from data.data_jornada.url_mapeo import load_partidos_master

# Función para normalizar los nombres de equipos
def normalize_team_name(name):
    """Normaliza nombres de equipos para comparación"""
    replacements = {
        " CF": "", "FC ": "", " FC": "", " de ": " ", "RCD ": "", "RC ": "", "CD ": "",
        "UD ": "", "CA ": "", "Real ": "", "Deportivo ": "", "Alavés": "Alaves", "Leganés": "Leganes",
        "Club ": "", "Atlético": "Atletico", "Español": "Espanyol"
    }
    
    result = name
    for old, new in replacements.items():
        result = result.replace(old, new)
    
    return result.strip().lower()

# Cargar las estadísticas de un partido
def load_match_stats(match_id=None, jornada=None, partido=None):
    """
    Carga las estadísticas de los partidos desde el CSV.
    
    Args:
        match_id (str, optional): ID del partido para filtrar
        jornada (int o str, optional): Número de jornada para filtrar
        partido (str, optional): Nombre del partido (ej. "Atletico Madrid-Girona")
        
    Returns:
        pandas.DataFrame: DataFrame con las estadísticas filtradas
    """
    file_path = Path("data/raw/match_stats_24_25_pag3.csv")
    
    if not file_path.exists():
        return None
    
    try:
        df = pd.read_csv(file_path, dtype=str, skipinitialspace=True)

        numeric_columns = ['Jornada', 'Goles', 'Tiros a Puerta', 'Tiros Fuera', 'Tiros Totales', 
                          'Tiros Bloqueados', 'Tiros Dentro Area', 'Tiros Fuera Area', 
                          'Faltas', 'CÃ³rners', 'Fueras de Juego', 'PosesiÃ³n', 
                          'Tarjetas Amarillas', 'Tarjetas Rojas', 'Paradas', 
                          'Pases Totales', 'Pases Completados', 'PrecisiÃ³n de Pases', 
                          'xG', 'Goles Evitados']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if jornada is not None:
            if isinstance(jornada, str) and jornada.replace('ª', '').isdigit():
                jornada = int(jornada.replace('ª', ''))
            df = df[df['Jornada'] == jornada]
        
        if partido is not None:
            equipos = partido.split('-')
            if len(equipos) != 2:
                return df
            
            equipo_local, equipo_visitante = equipos
            local_norm = normalize_team_name(equipo_local)
            visitante_norm = normalize_team_name(equipo_visitante)
            
            matches = []
            for p in df['Partido'].unique():
                p_equipos = p.split('-')
                if len(p_equipos) != 2:
                    continue
                    
                p_local, p_visitante = p_equipos
                p_local_norm = normalize_team_name(p_local)
                p_visitante_norm = normalize_team_name(p_visitante)
                
                if (p_local_norm == local_norm and p_visitante_norm == visitante_norm) or \
                   (p_local_norm == visitante_norm and p_visitante_norm == local_norm):
                    matches.append(p)
            
            if matches:
                df = df[df['Partido'].isin(matches)]
        
        return df
    
    except Exception as e:
        print(f"Error al cargar datos CSV: {str(e)}")
        return None

def get_partido_info(match_id):
    """
    Obtiene información de un partido a partir de su ID.
    Esto debería integrarse con tu sistema de mapeo.
    
    Args:
        match_id (str): ID del partido
        
    Returns:
        dict: Información del partido (jornada, partido, etc.)
    """
    # Esta función obtiene la información desde tu archivo partidos_master.csv
    
    try:        
        
        partidos_df = load_partidos_master()
        partido = partidos_df[partidos_df['match_id'] == match_id]
        
        if not partido.empty:
            return {
                'jornada': partido.iloc[0].get('jornada'),
                'partido': partido.iloc[0].get('partido')
            }
    except Exception as e:
        print(f"Error al obtener información del partido: {str(e)}")
    
    return None

# Cargar las estadísticas de un partido específico
def load_partido_stats(jornada, partido):
    """
    Carga las estadísticas de un partido específico.
    
    Args:
        jornada (int): Número de jornada
        partido (str): Nombre del partido (ej. "Atletico Madrid-Girona")
        
    Returns:
        pandas.DataFrame: DataFrame con las estadísticas del partido
    """
    file_path = Path("data/raw/stats/match_stats_24_25_pag3.csv")
    
    try:
        df = pd.read_csv(file_path)
        
        # Convertir jornada a int si es necesario
        if isinstance(jornada, str) and jornada.replace('ª', '').isdigit():
            jornada = int(jornada.replace('ª', ''))
        
        # Filtrar por jornada
        df_jornada = df[df['Jornada'] == jornada]
        
        if df_jornada.empty:
            print(f"No hay datos para la jornada {jornada}")
            return None
        
        # Normalizar nombres para una búsqueda más flexible
        def normalize_team_name(name):
            return (name.replace(" CF", "")
                       .replace(" FC", "")
                       .replace(" de ", " ")
                       .strip())
        
        partido_norm = "-".join([normalize_team_name(team) for team in partido.split("-")])
        
        # Buscar partido con nombres normalizados
        match_df = None
        for p in df_jornada['Partido'].unique():
            p_norm = "-".join([normalize_team_name(team) for team in p.split("-")])
            if p_norm == partido_norm:
                match_df = df_jornada[df_jornada['Partido'] == p]
                
                break
        
        if match_df is None or match_df.empty:
            print(f"No se encontró el partido {partido} en la jornada {jornada}")
        
        return match_df
    except Exception as e:
        print(f"Error al cargar estadísticas del partido: {str(e)}")
        return None

# Función adaptación para red de pases

def cumulative_match_mins(events_df):
    events_out = pd.DataFrame()
    # Tiempo acumulado para los eventos
    match_events = events_df.copy()
    match_events['cumulative_mins'] = match_events['minute'] + (1/60) * match_events['second']
    # Según periodo de juego
    for period in np.arange(1, match_events['period'].max() + 1, 1):
        if period > 1:
            t_delta = match_events[match_events['period'] == period - 1]['cumulative_mins'].max() - \
                                   match_events[match_events['period'] == period]['cumulative_mins'].min()
        elif period == 1 or period == 5:
            t_delta = 0
        else:
            t_delta = 0
        match_events.loc[match_events['period'] == period, 'cumulative_mins'] += t_delta
    # Se reconstruye el DF
    events_out = pd.concat([events_out, match_events])
    return events_out

def insert_ball_carries(events_df, min_carry_length=3, max_carry_length=60, min_carry_duration=1, max_carry_duration=10):
    events_out = pd.DataFrame()
    # Conversión según opta
    min_carry_length = 3.0
    max_carry_length = 60.0
    min_carry_duration = 1.0
    max_carry_duration = 10.0

    match_events = events_df.reset_index()
    match_carries = pd.DataFrame()
    
    for idx, match_event in match_events.iterrows():

        if idx < len(match_events) - 1:
            prev_evt_team = match_event['teamId']
            next_evt_idx = idx + 1
            init_next_evt = match_events.loc[next_evt_idx]
            take_ons = 0
            incorrect_next_evt = True

            while incorrect_next_evt:

                next_evt = match_events.loc[next_evt_idx]

                if next_evt['type'] == 'TakeOn' and next_evt['outcomeType'] == 'Successful':
                    take_ons += 1
                    incorrect_next_evt = True

                elif ((next_evt['type'] == 'TakeOn' and next_evt['outcomeType'] == 'Unsuccessful')
                      or (next_evt['teamId'] != prev_evt_team and next_evt['type'] == 'Challenge' and next_evt['outcomeType'] == 'Unsuccessful')
                      or (next_evt['type'] == 'Foul')):
                    incorrect_next_evt = True

                else:
                    incorrect_next_evt = False

                next_evt_idx += 1

            # Condicionantes por ver si se cumplen
            same_team = prev_evt_team == next_evt['teamId']
            not_ball_touch = match_event['type'] != 'BallTouch'
            dx = 105*(match_event['endX'] - next_evt['x'])/100
            dy = 68*(match_event['endY'] - next_evt['y'])/100
            far_enough = dx ** 2 + dy ** 2 >= min_carry_length ** 2
            not_too_far = dx ** 2 + dy ** 2 <= max_carry_length ** 2
            dt = 60 * (next_evt['cumulative_mins'] - match_event['cumulative_mins'])
            min_time = dt >= min_carry_duration
            same_phase = dt < max_carry_duration
            same_period = match_event['period'] == next_evt['period']

            valid_carry = same_team & not_ball_touch & far_enough & not_too_far & min_time & same_phase &same_period

            if valid_carry:
                carry = pd.DataFrame()
                prev = match_event
                nex = next_evt

                carry.loc[0, 'eventId'] = prev['eventId'] + 0.5
                carry['minute'] = np.floor(((init_next_evt['minute'] * 60 + init_next_evt['second']) + (
                        prev['minute'] * 60 + prev['second'])) / (2 * 60))
                carry['second'] = (((init_next_evt['minute'] * 60 + init_next_evt['second']) +
                                    (prev['minute'] * 60 + prev['second'])) / 2) - (carry['minute'] * 60)
                carry['teamId'] = nex['teamId']
                carry['x'] = prev['endX']
                carry['y'] = prev['endY']
                carry['expandedMinute'] = np.floor(((init_next_evt['expandedMinute'] * 60 + init_next_evt['second']) +
                                                    (prev['expandedMinute'] * 60 + prev['second'])) / (2 * 60))
                carry['period'] = nex['period']
                carry['type'] = carry.apply(lambda x: {'value': 99, 'displayName': 'Carry'}, axis=1)
                carry['outcomeType'] = 'Successful'
                carry['qualifiers'] = carry.apply(lambda x: {'type': {'value': 999, 'displayName': 'takeOns'}, 'value': str(take_ons)}, axis=1)
                carry['satisfiedEventsTypes'] = carry.apply(lambda x: [], axis=1)
                carry['isTouch'] = True
                carry['playerId'] = nex['playerId']
                carry['endX'] = nex['x']
                carry['endY'] = nex['y']
                carry['blockedX'] = np.nan
                carry['blockedY'] = np.nan
                carry['goalMouthZ'] = np.nan
                carry['goalMouthY'] = np.nan
                carry['isShot'] = np.nan
                carry['relatedEventId'] = nex['eventId']
                carry['relatedPlayerId'] = np.nan
                carry['isGoal'] = np.nan
                carry['cardType'] = np.nan
                carry['isOwnGoal'] = np.nan
                carry['type'] = 'Carry'
                carry['cumulative_mins'] = (prev['cumulative_mins'] + init_next_evt['cumulative_mins']) / 2

                match_carries = pd.concat([match_carries, carry], ignore_index=True, sort=False)

    match_events_and_carries = pd.concat([match_carries, match_events], ignore_index=True, sort=False)
    match_events_and_carries = match_events_and_carries.sort_values(['period', 'cumulative_mins']).reset_index(drop=True)

    # Se reconstruye el DF
    events_out = pd.concat([events_out, match_events_and_carries])

    return events_out

def get_possession_chains(events_df, chain_check, suc_evts_in_chain):
    # Se inicia el output
    events_out = pd.DataFrame()
    match_events_df = events_df.reset_index()

    # Aislar eventos válidos de la posesión
    match_pos_events_df = match_events_df[~match_events_df['type'].isin(['OffsideGiven', 'CornerAwarded','Start', 'Card', 'SubstitutionOff',
                                                                                  'SubstitutionOn', 'FormationChange','FormationSet', 'End'])].copy()

    # Agregar resultados binarios temporales e identificadores de equipo
    match_pos_events_df['outcomeBinary'] = (match_pos_events_df['outcomeType']
                                                .apply(lambda x: 1 if x == 'Successful' else 0))
    match_pos_events_df['teamBinary'] = (match_pos_events_df['teamName']
                         .apply(lambda x: 1 if x == min(match_pos_events_df['teamName']) else 0))
    match_pos_events_df['goalBinary'] = ((match_pos_events_df['type'] == 'Goal')
                         .astype(int).diff(periods=1).apply(lambda x: 1 if x < 0 else 0))

    # Crear un marco de datos para investigar cadenas de posesiones
    pos_chain_df = pd.DataFrame()

    # Compruebe si cada evento lo completa el mismo equipo que los siguientes eventos (check_evts-1)
    for n in np.arange(1, chain_check):
        pos_chain_df[f'evt_{n}_same_team'] = abs(match_pos_events_df['teamBinary'].diff(periods=-n))
        pos_chain_df[f'evt_{n}_same_team'] = pos_chain_df[f'evt_{n}_same_team'].apply(lambda x: 1 if x > 1 else x)
    pos_chain_df['enough_evt_same_team'] = pos_chain_df.sum(axis=1).apply(lambda x: 1 if x < chain_check - suc_evts_in_chain else 0)
    pos_chain_df['enough_evt_same_team'] = pos_chain_df['enough_evt_same_team'].diff(periods=1)
    pos_chain_df[pos_chain_df['enough_evt_same_team'] < 0] = 0

    match_pos_events_df['period'] = pd.to_numeric(match_pos_events_df['period'], errors='coerce')
    # Comprueba que no hay inicios de los próximos eventos (check_evts-1)
    pos_chain_df['upcoming_ko'] = 0
    for ko in match_pos_events_df[(match_pos_events_df['goalBinary'] == 1) | (match_pos_events_df['period'].diff(periods=1))].index.values:
        ko_pos = match_pos_events_df.index.to_list().index(ko)
        pos_chain_df.iloc[ko_pos - suc_evts_in_chain:ko_pos, 5] = 1

    # Determinar los inicios de posesión válidos según el equipo del evento y los próximos reinicios
    pos_chain_df['valid_pos_start'] = (pos_chain_df.fillna(0)['enough_evt_same_team'] - pos_chain_df.fillna(0)['upcoming_ko'])

    # Sumar inicios de posesión por reinicios (cambios de período y goles)
    pos_chain_df['kick_off_period_change'] = match_pos_events_df['period'].diff(periods=1)
    pos_chain_df['kick_off_goal'] = ((match_pos_events_df['type'] == 'Goal')
                     .astype(int).diff(periods=1).apply(lambda x: 1 if x < 0 else 0))
    pos_chain_df.loc[pos_chain_df['kick_off_period_change'] == 1, 'valid_pos_start'] = 1
    pos_chain_df.loc[pos_chain_df['kick_off_goal'] == 1, 'valid_pos_start'] = 1

    # Agregar la primera posesión de forma manual
    pos_chain_df['teamName'] = match_pos_events_df['teamName']
    pos_chain_df.loc[pos_chain_df.head(1).index, 'valid_pos_start'] = 1
    pos_chain_df.loc[pos_chain_df.head(1).index, 'possession_id'] = 1
    pos_chain_df.loc[pos_chain_df.head(1).index, 'possession_team'] = pos_chain_df.loc[pos_chain_df.head(1).index, 'teamName']

    # Iterar a través de inicios de posesión válidos y asignarles identificadores
    valid_pos_start_id = pos_chain_df[pos_chain_df['valid_pos_start'] > 0].index

    possession_id = 2
    for idx in np.arange(1, len(valid_pos_start_id)):
        current_team = pos_chain_df.loc[valid_pos_start_id[idx], 'teamName']
        previous_team = pos_chain_df.loc[valid_pos_start_id[idx - 1], 'teamName']
        if ((previous_team == current_team) & (pos_chain_df.loc[valid_pos_start_id[idx], 'kick_off_goal'] != 1) &
                (pos_chain_df.loc[valid_pos_start_id[idx], 'kick_off_period_change'] != 1)):
            pos_chain_df.loc[valid_pos_start_id[idx], 'possession_id'] = np.nan
        else:
            pos_chain_df.loc[valid_pos_start_id[idx], 'possession_id'] = possession_id
            pos_chain_df.loc[valid_pos_start_id[idx], 'possession_team'] = pos_chain_df.loc[valid_pos_start_id[idx], 'teamName']
            possession_id += 1

    # Asignar los identificadores de posesión y eal equipo dentro del cuadro de eventos
    match_events_df = pd.merge(match_events_df, pos_chain_df[['possession_id', 'possession_team']], how='left', left_index=True, right_index=True)

    # Completa los identificadores de posesión y el equipo en posesión
    match_events_df[['possession_id', 'possession_team']] = (match_events_df[['possession_id', 'possession_team']].fillna(method='ffill'))
    match_events_df[['possession_id', 'possession_team']] = (match_events_df[['possession_id', 'possession_team']].fillna(method='bfill'))

    # Se aplica al df
    events_out = pd.concat([events_out, match_events_df])

    return events_out

def process_whoscored_event_data(events_file, players_file, teams_file):
    """
    Procesa los datos de eventos y jugadores de WhoScored.
    
    Args:
        events_file (str): Ruta al archivo Parquet de eventos
        players_file (str): Ruta al archivo Parquet de jugadores
        teams_file (str): Ruta al archivo Parquet con mapeo de equipos
    
    Returns:
        tuple: DataFrames procesados de eventos y jugadores, info de equipos
    """
    # Si el archivo de eventos no existe, intentar buscar un archivo similar
    if not os.path.exists(events_file):
        base_path = os.path.dirname(events_file)
        filename = os.path.basename(events_file)
        
        # Extraer la jornada y los equipos del nombre
        match = re.match(r"(\d+ª)_([A-Za-z]+)-([A-Za-z]+)_EventData_whoscored\.parquet", filename)
        
        if match:
            jornada, equipo1, equipo2 = match.groups()
            pattern = f"{base_path}/{jornada}_*_EventData_whoscored.parquet"
            possible_files = glob.glob(pattern)
            
            if possible_files:
                events_file = possible_files[0]  # Usar el primer archivo encontrado
                print(f"Usando archivo alternativo: {events_file}")
                
                # También actualizar el archivo de jugadores para mantener consistencia
                players_file = events_file.replace("_EventData_", "_PlayerData_")
    
    # Verificar si el archivo es un archivo Parquet válido
    if not events_file.endswith(".parquet"):
        print(f"El archivo {events_file} no es un archivo Parquet válido.")
        return pd.DataFrame(), pd.DataFrame(), {}

    try:
        # Intentar leer los archivos Parquet
        df_red = pd.read_parquet(events_file)
        dfp_red = pd.read_parquet(players_file)
        dft_red = pd.read_csv(teams_file, delimiter=";")
    except Exception as e:
        print(f"Error al leer los archivos Parquet: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), {}

    # Extraer nombres de equipos del nombre del archivo
    filename = os.path.basename(events_file)
    match = re.match(r"(\d+ª)_([A-Za-z]+)-([A-Za-z]+)_EventData_whoscored\.parquet", filename)
    
    file_home_team = None
    file_away_team = None
    
    if match:
        _, file_home_team, file_away_team = match.groups()
    else:
        print(f"El nombre del archivo {filename} no sigue el formato esperado.")
        return pd.DataFrame(), pd.DataFrame(), {}

    # Resto del código continúa igual
    team_id_to_name = dict(zip(dft_red['id_whoscored'], dft_red['nombre']))
    
    # Determinar equipo local y visitante basado en la frecuencia de IDs
    home_team_id = dfp_red['teamId'].mode().iloc[0]
    away_team_id = dfp_red[dfp_red['teamId'] != home_team_id]['teamId'].mode().iloc[0]
    
    # Obtener nombres de equipos
    home_team_name = team_id_to_name.get(home_team_id, f"Equipo {home_team_id}")
    away_team_name = team_id_to_name.get(away_team_id, f"Equipo {away_team_id}")
    
    # Identificar si el Atlético de Madrid es local o visitante
    is_atleti_home = False
    for team_name in [home_team_name, away_team_name]:
        if 'atleti' in team_name.lower() or 'atletico' in team_name.lower() or 'atlético' in team_name.lower():
            is_atleti_home = (team_name == home_team_name)
            break

    # Lista de jornadas donde el orden local/visitante está invertido
    jornadas_invertidas = ["1ª", "10ª", "11ª", "12ª", "14ª", "18ª", "20ª", "24ª"]

    # Extraer la jornada del nombre del archivo
    match = re.search(r'(\d+ª)', os.path.basename(events_file))
    if match:
        jornada_actual = match.group(1)
        if jornada_actual in jornadas_invertidas:
            home_team_name, away_team_name = away_team_name, home_team_name
            home_team_id, away_team_id = away_team_id, home_team_id
            is_atleti_home = not is_atleti_home
    
    # Procesar tipos de eventos y periodos
    df_red['type'] = df_red['type'].str.extract(r"'displayName': '([^']+)")
    df_red['outcomeType'] = df_red['outcomeType'].str.extract(r"'displayName': '([^']+)")
    df_red['period'] = df_red['period'].str.extract(r"'displayName': '([^']+)")

    # Mapeo de periodos
    df_red['period'] = df_red['period'].replace({
        'FirstHalf': 1, 'SecondHalf': 2, 
        'FirstPeriodOfExtraTime': 3, 'SecondPeriodOfExtraTime': 4, 
        'PenaltyShootout': 5, 'PostGame': 14, 'PreMatch': 16
    })
    
    # Procesar datos
    df_red = cumulative_match_mins(df_red)
    df_red = insert_ball_carries(df_red)
    
    # Añadir nombres de equipos al DataFrame
    df_red['teamName'] = df_red['teamId'].map({
        home_team_id: home_team_name, 
        away_team_id: away_team_name
    })
    
    # Mapa de oposición
    opposition_dict = {
        home_team_name: away_team_name,
        away_team_name: home_team_name
    }
    df_red['oppositionTeamName'] = df_red['teamName'].map(opposition_dict)
    
    # Conversión de coordenadas
    df_red['x'] = df_red['x'] * 1.05
    df_red['y'] = df_red['y'] * 0.68
    df_red['endX'] = df_red['endX'] * 1.05
    df_red['endY'] = df_red['endY'] * 0.68
    
    # Columnas a eliminar en jugadores
    columns_to_drop = ['Unnamed: 0', 'height', 'weight', 'age', 'isManOfTheMatch', 
                      'field', 'stats', 'subbedInPlayerId', 'subbedOutPeriod', 
                      'subbedOutExpandedMinute', 'subbedInPeriod', 
                      'subbedInExpandedMinute', 'subbedOutPlayerId', 'teamId']
    dfp_red.drop(columns=[col for col in columns_to_drop if col in dfp_red.columns], inplace=True)
    
    # Combinar eventos y jugadores
    df_red = df_red.merge(dfp_red, on='playerId', how='left')
    
    # Cálculo de distancias progresivas
    df_red['prog_pass'] = np.where((df_red['type'] == 'Pass'), 
                               np.sqrt((105 - df_red['x'])**2 + (34 - df_red['y'])**2) - 
                               np.sqrt((105 - df_red['endX'])**2 + (34 - df_red['endY'])**2), 0)
    
    df_red['prog_carry'] = np.where((df_red['type'] == 'Carry'), 
                                np.sqrt((105 - df_red['x'])**2 + (34 - df_red['y'])**2) - 
                                np.sqrt((105 - df_red['endX'])**2 + (34 - df_red['endY'])**2), 0)
    
    df_red['pass_or_carry_angle'] = np.degrees(np.arctan2(df_red['endY'] - df_red['y'], df_red['endX'] - df_red['x']))
    
    # Normalizar nombres
    df_red['name'] = df_red['name'].astype(str)
    df_red['name'] = df_red['name'].apply(unidecode)
    
    # Generar nombres cortos
    df_red['shortName'] = df_red['name'].apply(get_short_name)
    
    # Obtener cadenas de posesión
    df_red = get_possession_chains(df_red, 5, 3)
    
    # Restaurar nombres de periodos
    df_red['period'] = df_red['period'].replace({
        1: 'FirstHalf', 2: 'SecondHalf', 
        3: 'FirstPeriodOfExtraTime', 4: 'SecondPeriodOfExtraTime', 
        5: 'PenaltyShootout', 14: 'PostGame', 16: 'PreMatch'
    })
    
    # Filtrar periodos
    df_red = df_red[df_red['period'] != 'PenaltyShootout']
    df_red = df_red.reset_index(drop=True)
    
    # Información de equipos para la visualización
    team_info = {
        'home_team_id': home_team_id,
        'away_team_id': away_team_id,
        'home_team_name': home_team_name,
        'away_team_name': away_team_name,
        'is_atleti_home': is_atleti_home,
        'file_home_team': file_home_team,
        'file_away_team': file_away_team
    }
    
    return df_red, dfp_red, team_info

def get_short_name(full_name):
    """
    Extrae un nombre corto a partir del nombre completo
    
    Args:
        full_name (str): Nombre completo
    
    Returns:
        str: Nombre corto
    """
    if pd.isna(full_name):
        return full_name
    
    parts = full_name.split()
    if len(parts) == 1:
        return full_name
    elif len(parts) == 2:
        return parts[0][0] + ". " + parts[1]
    else:
        return parts[0][0] + ". " + parts[1][0] + ". " + " ".join(parts[2:])

def get_passes_df(df):
    """
    Obtiene un DataFrame de pases a partir del DataFrame de eventos
    
    Args:
        df (pandas.DataFrame): DataFrame de eventos
    
    Returns:
        pandas.DataFrame: DataFrame de pases
    """
    
    df1 = df[~df['type'].str.contains('SubstitutionOn|FormationChange|FormationSet|Card')]
    df = df1
    df.loc[:, "receiver"] = df["playerId"].shift(-1)
    passes_ids = df.index[df['type'] == 'Pass']
    df_passes = df.loc[passes_ids, ["index", "x", "y", "endX", "endY", "teamName", 
                                    "playerId", "receiver", "type", "outcomeType", 
                                    "pass_or_carry_angle"]]

    return df_passes

def get_passes_between_df(teamName, passes_df, players_df, events_df):
    """
    Obtiene información de pases entre jugadores y sus posiciones medias
    """
    # Filtrar eventos y pases para el equipo
    team_events = events_df[events_df['teamName'] == teamName]
    team_passes = passes_df[passes_df['teamName'] == teamName]
    
    # Calcular posiciones medias de los jugadores
    average_locs_and_count_df = team_events.groupby('playerId').agg({
        'x': 'mean',
        'y': 'mean',
        'name': 'first',
        'position': 'first',
        'isFirstEleven': 'first'
    }).reset_index()
    
    # Renombrar columnas para la visualización
    average_locs_and_count_df = average_locs_and_count_df.rename(columns={
        'x': 'pass_avg_x',
        'y': 'pass_avg_y'
    })
    
    # Filtrar solo pases completados
    team_passes = team_passes[team_passes['outcomeType'] == 'Successful']
    
    # Calcular pases entre jugadores
    passes_between_df = team_passes.groupby(['playerId', 'receiver']).size().reset_index(name='pass_count')
    
    # Añadir información de posición para origen
    passes_between_df = passes_between_df.merge(
        average_locs_and_count_df[['playerId', 'name', 'pass_avg_x', 'pass_avg_y']], 
        on='playerId',
        how='left'
    )
    
    # Añadir información de posición para destino
    passes_between_df = passes_between_df.merge(
        average_locs_and_count_df[['playerId', 'name', 'pass_avg_x', 'pass_avg_y']], 
        left_on='receiver',
        right_on='playerId',
        how='left',
        suffixes=('', '_end')
    )
    
    return passes_between_df, average_locs_and_count_df