import sqlite3
import pandas as pd
import os
import sys

# Añadir el directorio raíz al path para importar config si es necesario
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_stats_db_connection():
    """Conectar a la base de datos de estadísticas del Atlético"""
    current_dir = os.getcwd()
    db_path = os.path.join(current_dir, "data", "raw", "base_atleti_streamlit.db")
    
    # Verificar que la base de datos existe
    if not os.path.exists(db_path):
        db_path = os.path.join(current_dir, "data", "raw", "base_Atleti_streamlit.db")
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"No se encontró la base de datos de estadísticas")
    
    return sqlite3.connect(db_path)

def get_player_stats(min_minutes=0, max_minutes=2700, position=None):
    """
    Obtiene estadísticas completas de jugadores, incluyendo métricas especiales.
    
    Args:
        min_minutes: Minutos mínimos jugados
        max_minutes: Minutos máximos jugados
        position: Posición específica (FW, MF, DF, GK)
        
    Returns:
        DataFrame con estadísticas combinadas
    """

    """Obtiene estadísticas completas de jugadores"""    
    
    # Verificar ubicación actual y bases de datos disponibles
    current_dir = os.getcwd()
    
    # Probar con ruta absoluta
    db_path = os.path.join(current_dir, "data", "raw", "base_atleti_streamlit.db")
    
    # También probar con mayúsculas/minúsculas diferentes
    alternate_path = os.path.join(current_dir, "data", "raw", "base_Atleti_streamlit.db")
    
    # Conectar a la base de datos usando la ruta que existe
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
    elif os.path.exists(alternate_path):
        conn = sqlite3.connect(alternate_path)
    else:
        raise FileNotFoundError(f"No se encontró la base de datos en {db_path} ni en {alternate_path}")
    
    # Verificar qué tablas están disponibles
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    conn = get_stats_db_connection()
    
    # Consulta SQL combinando todas las tablas
    query = """
    SELECT 
        j.id_jugador,
        j.nombre AS Jugador,
        j.nacionalidad AS Nacionalidad,
        j.posicion AS Posicion,
        j.edad AS Edad,
        ej.temporada AS Temporada,
        ej.partidos_jugados AS Partidos,
        ej.titularidades AS Titularidades,
        ej.minutos AS Minutos,
        
        -- Estadísticas ofensivas
        eo.goles AS Goles,
        eo.asistencias AS Asistencias,
        eo.goles_asistencias AS "Goles+Asistencias",
        eo.xG AS xG,
        eo.xA AS xA,
        eo.npxG AS npxG,
        eo.tiros AS Tiros,
        eo.tiros_puerta AS "Tiros a puerta",
        eo.porcentaje_tiros_puerta AS "%Tiros a puerta",
        
        -- Estadísticas de pases
        ep.porcentaje_pases_completados AS "%Pases completados",
        ep.pases_clave AS "Pases clave",
        ep.pases_ultimo_tercio AS "Pases último tercio",
        ep.pases_area_penal AS "Pases área penal",
        ep.pases_progresivos AS "Pases progresivos",
        
        -- Estadísticas de posesión
        epos.toques AS Toques,
        epos.intentos_regate AS "Intentos regate",
        epos.regates_exitosos AS "Regates exitosos",
        epos.porcentaje_regate AS "%Regate",
        epos.conducciones_progresivas AS "Conducciones progresivas",
        
        -- Estadísticas defensivas
        ed.entradas AS Entradas, 
        ed.intercepciones AS Intercepciones,
        ed.despejes AS Despejes,
        ed.bloqueos AS Bloqueos,
        ed.recuperaciones AS Recuperaciones,
        
        -- Estadísticas de disciplina
        edis.tarjetas_amarillas AS Amarillas,
        edis.tarjetas_rojas AS Rojas,
        edis.faltas_cometidas AS "Faltas cometidas",
        edis.faltas_recibidas AS "Faltas recibidas",
        
        -- Métricas especiales
        me.p_adj_tackles_interceptions_per90 AS "Ent+Int aj. a pos/90",
        me.p_adj_clearances_per90 AS "Despj. aj. a pos/90",
        me.p_adj_shot_blocks_per90 AS "Bloqueos tiro aj. a pos/90",
        me.p_adj_pass_blocks_per90 AS "Bloqueos pase aj. a pos/90",
        me.p_adj_tackles_win_possession_per90 AS "Recup. por posesión/90",
        me.p_adj_dribbled_past_per90 AS "Superado por regate/90",
        me.touch_centrality AS "Centralidad toques",
        me.tkl_int_per_600_opp_touch AS "TKL+INT/600 toques rival",
        me.carries_per_50_touches AS "Conducciones/50 toques",
        me.prog_carries_per_50_touches AS "Cond.progresivas/50t",
        me.prog_passes_per_50_cmp_passes AS "Pases prog/50 pases"
        
    FROM jugadores j
    JOIN estadisticas_jugador ej ON j.id_jugador = ej.id_jugador
    LEFT JOIN estadisticas_ofensivas eo ON ej.id_estadistica = eo.id_estadistica
    LEFT JOIN estadisticas_pases ep ON ej.id_estadistica = ep.id_estadistica
    LEFT JOIN estadisticas_posesion epos ON ej.id_estadistica = epos.id_estadistica
    LEFT JOIN estadisticas_defensivas ed ON ej.id_estadistica = ed.id_estadistica
    LEFT JOIN estadisticas_disciplina edis ON ej.id_estadistica = edis.id_estadistica
    LEFT JOIN metricas_especiales me ON ej.id_estadistica = me.id_estadistica
    WHERE ej.minutos BETWEEN ? AND ?
    """
    
    params = [min_minutes, max_minutes]
    
    # Añadir filtro de posición si se especifica
    if position:
        query += " AND j.posicion = ?"
        params.append(position)
    
    # Ordenar por minutos jugados (descendente)
    query += " ORDER BY ej.minutos DESC"
    
    # Ejecutar consulta
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    return df

def get_metric_groups():
    """
    Devuelve los grupos de métricas disponibles y sus columnas correspondientes
    
    Returns:
        Diccionario con grupos de métricas
    """
    return {
        "Ataque": ["Goles", "Asistencias", "Goles+Asistencias", "xG", "xA", "npxG", "Tiros", "Tiros a puerta", "%Tiros a puerta"],
        "Posesión": ["Toques", "Intentos regate", "Regates exitosos", "%Regate", "Conducciones progresivas"],
        "Pases": ["%Pases completados", "Pases clave", "Pases último tercio", "Pases área penal", "Pases progresivos"],
        "Defensa": ["Entradas", "Intercepciones", "Despejes", "Bloqueos", "Recuperaciones"],
        "Disciplina": ["Partidos", "Titularidades", "Minutos", "Amarillas", "Rojas", "Faltas cometidas", "Faltas recibidas"],
        "Métricas avanzadas": ["Ent+Int aj. a pos/90", "Despj. aj. a pos/90", "Bloqueos tiro aj. a pos/90", "Bloqueos pase aj. a pos/90", 
                              "Recup. por posesión/90", "Superado por regate/90", "Centralidad toques", "TKL+INT/600 toques rival", 
                              "Conducciones/50 toques", "Cond.progresivas/50t", "Pases prog/50 pases"]
    }

def get_ranking_by_metric(metric, limit=5, min_minutes=0):
    """
    Obtiene un ranking de jugadores según una métrica específica
    
    Args:
        metric: Nombre de la columna para rankear
        limit: Número máximo de jugadores a mostrar
        min_minutes: Minutos mínimos jugados para filtrar
        
    Returns:
        DataFrame con los mejores jugadores según la métrica
    """
    # Obtener todos los datos
    df = get_player_stats(min_minutes=min_minutes)
    
    # Verificar que la métrica existe
    if metric not in df.columns:
        return pd.DataFrame(columns=['Jugador', metric])
    
    # Filtrar por la métrica seleccionada, ordenar y limitar
    result = df[['Jugador', metric]].sort_values(by=metric, ascending=False).head(limit)
    
    return result