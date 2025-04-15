import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import json
import re
import streamlit as st
from pathlib import Path

def get_atletico_data():
    """
    Función única que extrae todos los datos necesarios de Understat y los devuelve en el formato esperado
    
    Returns:
        tuple: (df_expcGL, df1) DataFrames procesados con datos de xG y partidos
    """
    
    try:
        # Primer conjunto de datos (xG, xGA, etc.) - Datos generales de la liga
       
        link = "https://understat.com/league/La_liga"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        
        res = requests.get(link, headers=headers)
                
        if res.status_code != 200:
            print(f"Error de conexión a La_liga: {res.status_code}")
            raise Exception(f"Error al conectar con Understat La_liga: {res.status_code}")
            
        soup = BeautifulSoup(res.content, 'lxml')
        scripts = soup.find_all('script')
        
        # Extraer los datos JSON de los scripts
        
        if len(scripts) < 3:
            print(f"No hay suficientes scripts, solo encontrados: {len(scripts)}")
            for i, script in enumerate(scripts):
                
                raise Exception("No se encontraron suficientes scripts en la página")
            
        strings = scripts[2].string
                
        if not strings:
            raise Exception("Script vacío o nulo")
            
        try:
            ind_start = strings.index("('")+2
            ind_end = strings.index("')")
                        
            json_data = strings[ind_start:ind_end] 
                        
            json_data = json_data.encode('utf8').decode('unicode_escape')
            data = json.loads(json_data)
                        
            # Verificar si existe el ID 143 (Atletico Madrid)
            if '143' not in data:
                print(f"ID 143 no encontrado. IDs disponibles: {list(data.keys())}")
                # Tratar de encontrar el ID correcto
                atletico_keys = [k for k, v in data.items() if 'title' in v and ('Atlético' in v['title'] or 'Atletico' in v['title'])]
                if atletico_keys:
                    
                    atletico_id = atletico_keys[0]
                else:
                    raise Exception("No se pudo encontrar los datos del Atlético de Madrid")
            else:
                atletico_id = '143'
                
                
            df_expcGL = pd.DataFrame(data[atletico_id]['history'])
                       
            df_expcGL = df_expcGL[['xG','xGA','npxG','npxGA','xpts','npxGD']]
            
            # Convertir a float para evitar problemas
            df_expcGL['xG'] = df_expcGL['xG'].astype(float)
            df_expcGL['xGA'] = df_expcGL['xGA'].astype(float)
            df_expcGL['npxG'] = df_expcGL['npxG'].astype(float)
            df_expcGL['npxGA'] = df_expcGL['npxGA'].astype(float)
            df_expcGL['xpts'] = df_expcGL['xpts'].astype(float)
            df_expcGL['npxGD'] = df_expcGL['npxGD'].astype(float)
            
            # Crear columnas adicionales
            num_matches = len(df_expcGL)
            df_expcGL['Match'] = np.arange(1, num_matches + 1)
            df_expcGL['xGdif'] = df_expcGL['xG'] - df_expcGL['xGA']
            
            
            # Segundo conjunto de datos (información de partidos)
            link = "https://understat.com/team/Atletico_Madrid/2024"
            res = requests.get(link, headers=headers)
            
            
            if res.status_code != 200:
                print(f"Error de conexión al equipo: {res.status_code}")
                raise Exception(f"Error al conectar con Understat equipo: {res.status_code}")
                
            soup = BeautifulSoup(res.content, 'lxml')
            scripts = soup.find_all('script')
                        
            if len(scripts) < 2:
                print(f"No hay suficientes scripts en página de equipo, solo encontrados: {len(scripts)}")
                raise Exception("No se encontraron suficientes scripts en la página del equipo")
                
            strings = scripts[1].string
                        
            if not strings:
                raise Exception("Script de equipo vacío o nulo")
                
            ind_start = strings.index("('")+2 
            ind_end = strings.index("')") 
                        
            json_data = strings[ind_start:ind_end] 
                       
            json_data = json_data.encode('utf8').decode('unicode_escape')
            data = json.loads(json_data)
                       
            df1 = pd.DataFrame(data)
            df_h = df1['h'].apply(pd.Series)
            df_a = df1['a'].apply(pd.Series)
                     
            # Corregir códigos de equipos
            df_h['short_title_corregido'] = df_h['short_title'].copy()
            df_a['short_title_corregido'] = df_a['short_title'].copy()
            
            # Correcciones para Valladolid
            df_h.loc[(df_h['short_title'] == 'VAL') & (df_h['title'] == 'Real Valladolid'), 'short_title_corregido'] = 'RVL'
            df_a.loc[(df_a['short_title'] == 'VAL') & (df_a['title'] == 'Real Valladolid'), 'short_title_corregido'] = 'RVL'
            
            # Correcciones para Rayo Vallecano
            df_h.loc[df_h['title'] == 'Rayo Vallecano', 'short_title_corregido'] = 'RAY'
            df_a.loc[df_a['title'] == 'Rayo Vallecano', 'short_title_corregido'] = 'RAY'
            
            # Crear columna de jornadas
            df1['short_title_h'] = df_h['short_title_corregido']
            df1['short_title_a'] = df_a['short_title_corregido']
            df1 = df1[['short_title_h','short_title_a']]
            df1['final'] = df1['short_title_h']+df1['short_title_a']
            
            # Diccionario de mapeo para nombres de partidos
            match_mapping = {
                'VILATL': 'VIL-ATM', 'ATLGIR': 'ATM-GIR', 'ATLESP': 'ATM-ESP', 'ATHATL': 'ATH-ATM', 'ATLVAL': 'ATM-VAL', 'RAYATL': 'RAY-ATM',
                'CELATL': 'CEL-ATM', 'ATLRMA': 'ATM-RMA', 'SOCATL': 'RSO-ATM', 'ATLLEG': 'ATM-LEG', 'BETATL': 'BET-ATM', 'ATLLPL': 'ATM-LPM',
                'MALATL': 'MLL-ATM', 'ATLALA': 'ATM-ALA', 'RVLATL': 'RVL-ATM', 'ATLSEV': 'ATM-SEV', 'ATLGET': 'ATM-GET', 'BARATL': 'FCB-ATM',
                'ATLOSA': 'ATM-OSA', 'LEGATL': 'LEG-ATM', 'ATLVIL': 'ATM-VIL', 'ATLMAL': 'ATM-MLL', 'RMAATL': 'RMA-ATM', 'ATLCEL': 'ATM-CEL',
                'VALATL': 'VAL-ATM', 'ATLATH': 'ATM-ATH', 'GETATL': 'GET-ATM', 'ATLBAR': 'ATM-FCB', 'ESPATL': 'ESP-ATM', 'SEVATL': 'SEV-ATM',
                'ATLRVL': 'ATM-RVL', 'LPLATL': 'LPM-ATM', 'ATLRAY': 'ATM-RAY', 'ALAATL': 'ALA-ATM', 'ATLSOC': 'ATM-RSO', 'OSAATL': 'OSA-ATM',
                'ATLBET': 'ATM-BET', 'GIRATL': 'GIR-ATM'
            }
            
            # Crear columna de jornada con formato
            df1['jornada'] = ''
            for i, row in df1.iterrows():
                match = row['final']
                match_fixed = match_mapping.get(match, match)
                jornada_match = f"J{i+1}"  # Formato simplificado: J1, J2, etc.
                df1.at[i, 'jornada'] = jornada_match
                df1.at[i, 'partido'] = match_fixed
            
            # Agregar columna de rival para compatibilidad con la visualización
            df_expcGL['jornada'] = [f"J{i+1}" for i in range(len(df_expcGL))]
            
            # Crear columna de rival con manejo de excepciones
            rivals = []
            for i in range(len(df_expcGL)):
                try:
                    if i < len(df_h) and i < len(df_a):
                        if df_a.loc[i, 'short_title'] == 'ATL':
                            rivals.append(df_h.loc[i, 'title'])
                        else:
                            rivals.append(df_a.loc[i, 'title'])
                    else:
                        rivals.append(f"Rival {i+1}")
                except Exception as e:
                    print(f"Error al obtener rival para índice {i}: {e}")
                    rivals.append(f"Rival {i+1}")
            
            df_expcGL['rival'] = rivals
            
            return df_expcGL, df1  # Devolvemos ambos DataFrames
        
        except Exception as inner_e:
            print(f"Error en la extracción de datos JSON: {inner_e}")
            raise  # Re-lanzar la excepción para que se maneje en el bloque exterior
            
    except Exception as e:
        print(f"Error al obtener datos de Understat: {e}")
        import traceback
        traceback.print_exc()
        
        # No intentar con datos inventados, solo mostrar el error
        raise Exception(f"No se pudieron obtener datos de xG: {e}")

def get_shots_data(match_id):
    """
    Obtiene datos de tiros desde Understat para un partido específico
    
    Args:
        match_id (str): ID del partido en Understat
        
    Returns:
        pd.DataFrame: DataFrame con los datos de tiros o None si hay error
    """
    # Usar caché para evitar múltiples llamadas al mismo partido
    @st.cache_data(ttl=3600)
    def _fetch_shots_data(match_id):
        
        url = f'https://understat.com/match/{match_id}'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        
        try:
            
            response = requests.get(url, headers=headers, timeout=10)
                      
            if response.status_code != 200:
                print(f"Error en la respuesta: {response.status_code}")
                return None
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Verificar si hay scripts
            scripts = soup.find_all('script')
                        
            # Buscar shotsData
            shotsData_found = False
            for script in scripts:
                if 'var shotsData' in str(script):
                    shotsData_found = True
                                        
                    # Extraer la cadena JSON usando expresiones regulares
                    pattern = r'var shotsData\s*=\s*JSON\.parse\(\'(.*?)\'\)'
                    match = re.search(pattern, str(script))
                    if match:
                        json_str = match.group(1)
                        # Decodificar la secuencia de escape
                        json_str = bytes(json_str, 'utf-8').decode('unicode_escape')
                        data = json.loads(json_str)
                        
                        shots = []
                        for team in ['h', 'a']:
                            for shot in data[team]:
                                shots.append({
                                    'x': float(shot['X']) * 100,
                                    'y': float(shot['Y']) * 100,
                                    'player': shot['player'],
                                    'minute': int(shot['minute']),
                                    'result': shot['result'],
                                    'xG': float(shot['xG']),
                                    'team': 'Local' if team == 'h' else 'Visitante'
                                })
                        
                        return pd.DataFrame(shots)
                    else:
                        print("No se pudo extraer el JSON con regex")
            
            if not shotsData_found:
                print("No se encontró el script con shotsData")
                
        except Exception as e:
            print(f"Error al obtener datos de Understat: {str(e)}")
            import traceback
            print(traceback.format_exc())
        
        return None
    
    return _fetch_shots_data(match_id)

def get_shot_map(understat_id):
    """
    Obtiene los datos de tiros separados por equipos local y visitante
    
    Args:
        understat_id (str): ID del partido en Understat
        
    Returns:
        dict: Diccionario con claves 'local' y 'visitante' conteniendo DataFrames
    """
    shots_df = get_shots_data(understat_id)
    
    if shots_df is None or shots_df.empty:
        return None
    
    return {
        'local': shots_df[shots_df['team'] == 'Local'],
        'visitante': shots_df[shots_df['team'] == 'Visitante']
    }