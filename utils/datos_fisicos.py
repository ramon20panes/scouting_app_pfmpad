import streamlit as st
import unicodedata
import pandas as pd
import sqlite3
from pathlib import Path
import streamlit as st
from datetime import datetime
import plotly.graph_objects as go

def get_player_id_mapping():
    """
    Retorna un diccionario con el mapeo de nombres de jugadores a sus IDs de Sofascore
    """
    player_mapping = {
        # Mapeo directo de nombres normalizados a IDs
        'julian alvarez': 944656,
        'julian': 944656,
        'alvarez': 944656,
        'jorge resurreccion': 84539,
        'koke': 84539,
        'jan oblak': 56651,
        'oblak': 56651,
        'antoine griezmann': 85859,
        'griezmann': 85859,
        'jose maria gimenez': 142489,
        'gimenez': 142489,
        'axel witsel': 35612,
        'witsel': 35612,
        'rodrigo de paul': 188586,
        'de paul': 188586,
        'conor gallagher': 837448,
        'gallagher': 837448,
        'marcos llorente': 263651,
        'llorente': 263651,
        'cesar azpilicueta': 69768,
        'azpilicueta': 69768,
        'angel correa': 325355,
        'correa': 325355,
        'pablo barrios': 1063610,
        'barrios': 1063610,
        'alexander sorloth': 309078,
        'sorloth': 309078,
        'thomas lemar': 227236,
        'lemar': 227236,
        'samuel lino': 995293,
        'lino': 995293,
        'clement lenglet': 214866,
        'lenglet': 214866,
        'nahuel molina': 580550,
        'molina': 580550,
        'rodrigo riquelme': 900981,
        'riquelme': 900981,
        'javi galan': 825133,
        'galan': 825133,
        'giuliano simeone': 1099352,
        'giuliano': 1099352,
        'simeone': 1099352,
        'reinildo mandava': 831424,
        'reinildo': 831424,
        'robin le normand': 787751,
        'le normand': 787751,
        'ilias kostis': 1105173,
        'kostis': 1105173,
        'aitor gismera': 1178542,
        'gismera': 1178542,
        'javi serrano': 1133962,
        'serrano': 1133962,
        'antonio gomis': 1167358,
        'gomis': 1167358,
        'adrian nino': 1402927,
        'nino': 1402927,
        'alejandro iturbe': 1134077,
        'iturbe': 1134077,
        'rayane belaid': 1099526,
        'rayane': 1099526,
        'belaid': 1099526,
        'carlos gimenez': 1007254,
        'carlos': 1007254,
        'geronimo spina': 1271307,
        'spina': 1271307,
        'juan musso': 286448,
        'musso': 286448
    }
    return player_mapping

# Función para normalizar nombres
def normalize_name(name):
    """Normaliza un nombre (quita acentos, convierte a minúsculas)"""
    if not name:
        return ""
    
    name = str(name).lower()
    # Comentar o quitar esta línea para evitar reemplazar 'ø'
    # name = name.replace('ø', 'o')
    name = name.replace('æ', 'ae')
    name = name.replace('å', 'a')

    return ' '.join(unicodedata.normalize('NFKD', str(name).lower())
                    .encode('ASCII', 'ignore')
                    .decode('ASCII')
                    .split())

def get_direct_name_mapping():
    """
    Proporciona un mapeo directo entre diferentes variantes de nombres de jugadores
    """
    name_variants = {
        # Actualizar estas cuatro entradas problemáticas
        "Alexander Sørloth": ["Alexander Sorloth", "Alekxander Sørloth", "Sorloth", "Sørloth"],
        "José María Giménez": ["Jose Maria Gimenez", "Giménez", "Gimenez"],
        "Koke": ["Jorge Resurrección", "Jorge Resurreccion", "Koke"],
        "Clément Lenglet": ["Clement Lenglet", "Lenglet"],
        
        # Mantener el resto como estaba
        "Julian Alvarez": ["Julián Álvarez", "Julian Alvarez", "Julián"],
        "Geronimo Spina": ["Gerónimo Spina", "Spina"],
        "Angel Correa": ["Ángel Correa", "Correa"],
        "Rodrigo De Paul": ["Rodrigo de Paul", "De Paul"],
        "Cesar Azpilicueta": ["César Azpilicueta", "Azpilicueta"],
        "Giuliano Simeone": ["Giuliano", "Simeone"],
        "Robin Le Normand": ["Le Normand"],
        "Jan Oblak": ["Oblak"],
        "Juan Musso": ["Juan Musso", "Musso"],
        "Marcos Llorente": ["Llorente"],
        "Antoine Griezmann": ["Griezmann"],
        "Thomas Lemar": ["Lemar"],
        "Samuel Lino": ["Lino"],
        "Nahuel Molina": ["Molina"],
        "Rodrigo Riquelme": ["Riquelme"],
        "Javi Galan": ["Javi Galán", "Galán", "Galan"],
        "Axel Witsel": ["Witsel"],
        "Conor Gallagher": ["Gallagher"],
        "Reinildo Mandava": ["Reinildo"],
        "Pablo Barrios": ["Barrios"],
        "Ilias Kostis": ["Kostis"],
        "Aitor Gismera": ["Gismera"],
        "Javi Serrano": ["Serrano"],
        "Antonio Gomis": ["Gomis"],
        "Adrian Nino": ["Adrián Niño", "Niño", "Nino"],
        "Alejandro Iturbe": ["Iturbe"],
        "Rayane Belaid": ["Rayane", "Belaid"],
        "Carlos Gimenez": ["Carlos Giménez", "Carlos"]
    }
    return name_variants

# Función para cargar datos físicos
@st.cache_data(ttl=3600)
def load_physical_data():
    """Carga datos físicos desde la base de datos SQLite"""
    try:
        db_path = Path("data/raw/ATM_condic_24_25.db")
        if db_path.exists():
            conn = sqlite3.connect(db_path)
            
            # Consulta para obtener datos físicos desde la tabla datos_fisicos
            query = """
            SELECT 
                id as id_jugador,
                Nombre as nombre, 
                Jornada as jornada,
                Jornada as fecha, 
                Distancia as distancia_total, 
                Sprints_Rel_m as distancia_sprint,
                Max_Speed as velocidad_max, 
                Aceleraciones as aceleraciones, 
                Deceleraciones as desaceleraciones, 
                Sprints_Abs_Cnt as sprints,
                HSR_Abs_m as distancia_alta_intensidad,
                AVG_HR as carga_cardio,
                Total_Impacts as impactos,
                ROUND(Distancia / CASE WHEN Min > 0 THEN Min ELSE 1 END, 2) as distancia_por_min,
                Distancia as distancia,
                Distancia_Explosiva as distancia_explosiva,
                HIBD as hibd,
                MAX_Acc as max_acc,
                MAX_Dec as max_dec,
                AVG_Acc as avg_acc,
                AVG_Dec as avg_dec,
                High_Acc_Cnt as high_acc_cnt,
                High_Dec_Cnt as high_dec_cnt,
                High_Acc_m as high_acc_m,
                High_Dec_m as high_dec_m,
                Max_HR as max_hr,
                AVG_HR as avg_hr,
                High_HR_m as high_hr_m,
                High_HR_Cnt as high_hr_cnt,
                Sprints_Abs_Cnt as sprints_abs_cnt,
                Sprints_Rel_Cnt as sprints_rel_cnt,
                HSR_Abs_Cnt as hsr_abs_cnt,
                HSR_Rel_Cnt as hsr_rel_cnt,
                Sprints_Rel_m as sprints_rel_m,
                Sprints_Abs_m as sprints_abs_m,
                HSR_Abs_m as hsr_abs_m,
                HSR_Rel_m as hsr_rel_m,
                AVG_Speed as avg_speed,
                Steps_Count as steps_count,
                Step_Balance as step_balance,
                Jumps_Count as jumps_count,
                HMLD as hmld,
                HML_Cnt as hml_cnt
            FROM datos_fisicos
            ORDER BY Jornada DESC
            """
            
            df = pd.read_sql(query, conn)
            conn.close()
            
            # Convertir jornadas a formato fecha
            base_date = datetime(2024, 8, 1)  # Fecha base para la jornada 1
            df['fecha'] = pd.to_datetime([base_date + pd.Timedelta(days=(int(j)-1)*7) for j in df['jornada']])
            
            return df
        else:
            st.error(f"No se encontró la base de datos en {db_path}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error al cargar datos físicos: {e}")
        return pd.DataFrame()

# Función para cargar datos de jugadores con nombres normalizados
@st.cache_data(ttl=3600)
def load_players_data():
    """Carga información de jugadores desde el archivo CSV con normalización de nombres"""
    try:
        players_path = Path("data/raw/master/jugadores_master.csv")
        if players_path.exists():
            delimiters = [';', ',', '\t']

            for delimiter in delimiters:
                try:
                    df = pd.read_csv(players_path, delimiter=delimiter, encoding='utf-8')
                    
                    if len(df.columns) == 1 and delimiter in df.iloc[0, 0]:
                        primera_col = df.columns[0]
                        nuevas_cols = primera_col.split(delimiter)
                        valores_split = [row.split(delimiter) for row in df[primera_col]]
                        df = pd.DataFrame(valores_split, columns=nuevas_cols)
                    
                    if 'nombre_completo' not in df.columns:
                        if 'short_name' in df.columns:
                            df['nombre_completo'] = df['short_name']
                        elif 'nombre' in df.columns:
                            df['nombre_completo'] = df['nombre']

                    df['nombre_completo_norm'] = df['nombre_completo'].apply(normalize_name)
                    if 'short_name' in df.columns:
                        df['short_name_norm'] = df['short_name'].apply(normalize_name)

                    return df
                except Exception as e:
                    st.warning(f"Error al cargar con delimitador {delimiter}: {e}")
                    continue

            st.error(f"No se pudo cargar correctamente el archivo de jugadores en {players_path}")
            return pd.DataFrame()
        else:
            st.error(f"No se encontró el archivo de jugadores en {players_path}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error al cargar datos de jugadores: {e}")
        return pd.DataFrame()

# Cálculo de tendencias
def calculate_trend(player_data, metric):
    """Calcula la tendencia de una métrica específica"""
        
    # Convertir nombres a minúsculas para comparación más flexible
    metric_lower = metric.lower()
    columns_lower = {col.lower(): col for col in player_data.columns}
    
    # Buscar coincidencia aproximada
    if metric_lower in columns_lower:
        actual_metric = columns_lower[metric_lower]
        
        metric = actual_metric
    else:
        
        return "stable", 0
    
    if len(player_data) < 2:
        return "stable", 0
    
    # Orden por jornada
    sorted_data = player_data.sort_values('jornada')
    
    # Diferencia entre último y penúltimo valor
    last_value = sorted_data[metric].iloc[-1]
    previous_value = sorted_data[metric].iloc[-2]
    
    diff = last_value - previous_value
    percent_change = (diff / previous_value) * 100 if previous_value > 0 else 0
    
    if abs(percent_change) < 2:
        return "stable", percent_change
    elif percent_change > 0:
        return "up", percent_change
    else:
        return "down", percent_change

# Función para mostrar el valor con indicador de tendencia
def display_metric_with_trend(label, value, trend, change_pct):
    """Muestra una métrica con su tendencia"""
    if trend == "up":
        trend_icon = "↑"
        trend_class = "metric-up"
    elif trend == "down":
        trend_icon = "↓"
        trend_class = "metric-down"
    else:
        trend_icon = "→"
        trend_class = "metric-stable"
    
    st.markdown(f"""
    <div>
        <span>{label}:</span>
        <span class="{trend_class}">{value} {trend_icon} ({change_pct:.1f}%)</span>
    </div>
    """, unsafe_allow_html=True)

# Función para crear gráfico de evolución
def create_evolution_chart(player_data, metrics):
    """Crea un gráfico de evolución para las métricas seleccionadas"""
    if player_data.empty or len(metrics) == 0:
        return None
    
    # Orden por jornada
    df_plot = player_data.sort_values('jornada')
    
    # Figura
    fig = go.Figure()
    
    # Colores para las líneas
    colors = ['#e41a1c', '#377eb8', '#4daf4a']
    
    # Añadir cada métrica
    for i, metric in enumerate(metrics):
        fig.add_trace(go.Scatter(
            x=df_plot['jornada'],
            y=df_plot[metric],
            mode='lines+markers',
            name=metric.replace('_', ' ').title(),
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=8)
        ))
    
    # Configurar layout
    fig.update_layout(
        title="Evolución por Jornada",
        xaxis_title="Jornada",
        yaxis_title="Valor",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

# Función para crear gráfico de barras comparativo
def create_comparison_chart(player_data, metrics, jornada=None):
    """Crea un gráfico de barras para comparar métricas en una jornada específica"""
    if player_data.empty or len(metrics) == 0:
        return None
    
    # Si no se especifica jornada, usar la última
    if jornada is None:
        jornada = player_data['jornada'].max()
    
    # Filtrar por jornada
    df_jornada = player_data[player_data['jornada'] == jornada]
    
    if df_jornada.empty:
        return None
    
    # Preparar datos para el gráfico
    valores = []
    for metric in metrics:
        if metric in df_jornada.columns:
            valores.append(df_jornada[metric].iloc[0])
        else:
            valores.append(0)
    
    # Crear gráfico
    fig = go.Figure()
    
    # Colores para las barras
    colors = ['#e41a1c', '#377eb8', '#4daf4a']
    
    # Añadir barras
    for i, (metric, valor) in enumerate(zip(metrics, valores)):
        fig.add_trace(go.Bar(
            x=[metric.replace('_', ' ').title()],
            y=[valor],
            name=metric.replace('_', ' ').title(),
            marker_color=colors[i % len(colors)]
        ))
    
    # Configurar layout
    fig.update_layout(
        title=f"Comparativa - Jornada {jornada}",
        yaxis_title="Valor",
        template="plotly_white",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    
    return fig

# Función para calcular la edad a partir de la fecha de nacimiento
def calculate_age(birth_date_str):
    try:
        birth_date = datetime.strptime(str(birth_date_str), '%Y-%m-%d')
        today = datetime.now()
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        return age
    except:
        return None
    
def map_name_to_database(nombre):
    """
    Mapea un nombre desde el CSV a como aparece en la base de datos (tabla datos_fisicos)
    """
    # Normalizar el nombre primero
    nombre_norm = normalize_name(nombre)
    
    # Diccionario de mapeo de nombres normalizados a nombres exactos en la base de datos
    db_name_mapping = {
        "julian alvarez": "Julián Álvarez",
        "jorge resurreccion": "Koke",
        "jorge resurrección": "Koke",
        "koke": "Koke",
        "jose maria gimenez": "José María Giménez",
        "gimenez": "José María Giménez",
        "giménez": "José María Giménez",
        "jan oblak": "Jan Oblak",
        "oblak": "Jan Oblak",
        "antoine griezmann": "Antoine Griezmann",
        "griezmann": "Antoine Griezmann",
        "axel witsel": "Axel Witsel",
        "witsel": "Axel Witsel",
        "rodrigo de paul": "Rodrigo De Paul",
        "de paul": "Rodrigo De Paul",
        "conor gallagher": "Conor Gallagher",
        "gallagher": "Conor Gallagher",
        "marcos llorente": "Marcos Llorente",
        "llorente": "Marcos Llorente",
        "cesar azpilicueta": "César Azpilicueta",
        "césar azpilicueta": "César Azpilicueta",
        "azpilicueta": "César Azpilicueta",
        "angel correa": "Ángel Correa",
        "ángel correa": "Ángel Correa",
        "correa": "Ángel Correa",
        "pablo barrios": "Pablo Barrios",
        "barrios": "Pablo Barrios",
        "alexander sorloth": "Alexander Sørloth",
        "aleksander sorloth": "Alexander Sørloth",
        "sørloth": "Alexander Sørloth",
        "sorloth": "Alexander Sørloth",
        "alexander sorloth": "Alexander Sørloth",
        "thomas lemar": "Thomas Lemar",
        "lemar": "Thomas Lemar",
        "samuel lino": "Samuel Lino",
        "samu lino": "Samuel Lino",
        "lino": "Samuel Lino",
        "clement lenglet": "Clément Lenglet",
        "clément lenglet": "Clément Lenglet",
        "lenglet": "Clément Lenglet",
        "clement lenglet": "Clément Lenglet",
        "nahuel molina": "Nahuel Molina",
        "molina": "Nahuel Molina",
        "rodrigo riquelme": "Rodrigo Riquelme",
        "riquelme": "Rodrigo Riquelme",
        "javi galan": "Javi Galán",
        "javi galán": "Javi Galán",
        "galan": "Javi Galán",
        "galán": "Javi Galán",
        "giuliano simeone": "Giuliano Simeone",
        "giuliano": "Giuliano Simeone",
        "simeone": "Giuliano Simeone",
        "reinildo mandava": "Reinildo Mandava",
        "reinildo": "Reinildo Mandava",
        "robin le normand": "Robin Le Normand",
        "le normand": "Robin Le Normand",
        "adrian niño": "Adrian Niño",
        "adrián niño": "Adrian Niño",
        "niño": "Adrian Niño",
        "nino": "Adrian Niño",
        "juan musso": "Juan Musso",
        "musso": "Juan Musso"
    }
    
    # Buscar en el mapeo
    if nombre_norm in db_name_mapping:
        return db_name_mapping[nombre_norm]
    
    # Si no se encuentra, buscar coincidencias parciales
    for key, value in db_name_mapping.items():
        if key in nombre_norm or nombre_norm in key:
            return value
    
    # Si no hay coincidencia, devolver el nombre original
    return nombre

def get_player_physical_data(player_name):
    """
    Obtiene los datos físicos para un jugador específico, usando el mapeo de nombres
    """
    # Cargar todos los datos físicos
    all_physical_data = load_physical_data()
    
    if all_physical_data.empty:
        return pd.DataFrame()
    
    # Mapear el nombre del jugador al formato de la base de datos
    db_name = map_name_to_database(player_name)
    
    # Filtrar los datos para este jugador
    player_data = all_physical_data[all_physical_data['nombre'] == db_name]
    
    if player_data.empty:
        print(f"No se encontraron datos físicos para {player_name} (mapeado a {db_name})")
        
        # Segundo intento: buscar coincidencias parciales
        for name in all_physical_data['nombre'].unique():
            if normalize_name(name) == normalize_name(db_name) or normalize_name(name) in normalize_name(db_name) or normalize_name(db_name) in normalize_name(name):
                print(f"Se encontró una posible coincidencia: {name}")
                return all_physical_data[all_physical_data['nombre'] == name]
    
    return player_data