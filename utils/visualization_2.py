import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.ticker as ticker
import matplotlib.image as mpimg
import highlight_text
from highlight_text import fig_text
from PIL import Image
from pathlib import Path

# Definir función para crear un gráfico bumpy chart
def create_bumpy_chart(df, highlight_teams=None):
    """
    Crea un gráfico bumpy chart para visualizar la evolución de posiciones de equipos en La Liga.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos de posiciones
        highlight_teams (list): Lista de equipos a destacar
        
    Returns:
        fig, ax: Figura y ejes de matplotlib
    """
    # Equipos por defecto
    if highlight_teams is None or len(highlight_teams) == 0:
        highlight_teams = ["Club Atlético de Madrid", "Real Madrid CF", "FC Barcelona"]
    
    # Paleta de colores para equipos
    team_colors = {
        "Club Atlético de Madrid": "#272E61",
        "Real Madrid CF": "white",
        "FC Barcelona": "#CB3524",
        "Athletic Club": "#282828",
        "Villarreal CF": "#FFD700",
        "Real Betis Balompié": "#00A650",
        "Sevilla FC": "#C41E3A",
        "Valencia CF": "#FF7500",
        "Real Sociedad de Fútbol": "#0066CC",
        "Girona FC": "#C91C2E",
        "CA Osasuna": "#AB1311",        
        "CD Leganés": "#2C3D98",       
        "Deportivo Alavés": "#0067B1",  
        "Getafe CF": "#005999",         
        "RC Celta de Vigo": "#8AD2F0",  
        "RCD Espanyol de Barcelona": "#0070B2", 
        "RCD Mallorca": "#E10D2B",      
        "Rayo Vallecano de Madrid": "#E53027", 
        "Real Valladolid CF": "#6A256F", 
        "UD Las Palmas": "#FFD700"      
    }
    
    # Crear un diccionario de colores solo para los equipos a destacar
    highlight_dict = {team: team_colors.get(team, "#A0A0A0") for team in highlight_teams}
    
    # Determinar la última jornada con datos diferentes
    ultima_jornada = None
    for i in range(1, 39):
        columna = f'J{i}'
        if columna in df.columns:
            # Comparar con la columna anterior si existe
            if i > 1 and (df[columna] == df[f'J{i-1}']).all():
                ultima_jornada = i-1
                break
        else:
            ultima_jornada = i-1
            break

    if not ultima_jornada:
        ultima_jornada = 38
    
    # Filtrar el DataFrame
    df_filtered = df.iloc[:, :ultima_jornada+1]  # +1 para incluir la columna "Equipo"
    
    # Transponer el dataframe para el formato que necesita el gráfico
    df_plot = df_filtered.set_index('Equipo').T
    
    # Crear las etiquetas de jornadas
    jornada_labels = ['Jornada ' + str(num) for num in range(1, len(df_plot)+1)]
    
    # Crear figura y ejes
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#d4d4d4')
    ax.set_facecolor('#d4d4d4')

    # Quitar todos los bordes
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Añadir un borde azul del Atleti
    fig.patch.set_edgecolor('#272E61')
    fig.patch.set_linewidth(2)  # Grosor del borde
    
    # Dibujar líneas grises para todos los equipos como fondo
    for equipo in df_plot.columns:
        if equipo not in highlight_teams:
            ax.plot(jornada_labels, df_plot[equipo], 
                    color='lightgray', alpha=0.5, linewidth=1, 
                    marker='o', markersize=5)
    
    # Dibujar líneas de colores para equipos destacados
    for equipo in highlight_teams:
        if equipo in df_plot.columns:
            color = highlight_dict[equipo]
            ax.plot(jornada_labels, df_plot[equipo], 
                    color=color, linewidth=3, marker='o', markersize=8, 
                    label=equipo)
                
    # Configuración del gráfico
    ax.set_ylabel('Posición', color='#272E61', fontweight='bold', size=12)
    ax.set_xlabel('Jornadas', color='#272E61', fontweight='bold', size=12)
    
    # Configurar etiquetas de eje X (jornadas)
    plt.xticks(rotation=45, color="#272E61", weight='bold', fontsize=10)
    
    ax.set_ylim(21, 0)

    # Configurar etiquetas de eje Y (posiciones)
    ax.set_yticks(range(1, 21))
    ax.set_yticklabels([str(i) for i in range(1, 21)], color="#272E61", weight='bold', fontsize=10)

    ax.grid(True, alpha=0.3)

    # Título
    plt.title('Progresión LaLiga 24/25', 
             color='darkblue', 
             fontsize=18, 
             fontweight='bold', 
             pad=20)
    
    # Subtítulo con colores según equipo
    highlight_text_str = "Comparación "
    highlight_textprops = []
    
    for i, team in enumerate(highlight_teams):
        if i > 0:
            highlight_text_str += ", "
        highlight_text_str += f"<{team}>"
        highlight_textprops.append({"color": highlight_dict[team]})
    
    highlight_text.fig_text(
        x=.5, 
        y=.93,
        s=highlight_text_str,
        highlight_textprops=highlight_textprops,
        fontsize=12,
        color='black',
        ha='center'
    )
    
    plt.tight_layout()
    return fig, ax

def get_team_logo(team_name, team_mapping, default_scale=0.08):
    """
    Obtiene el logo de un equipo con escalado personalizado.
    El nombre del equipo se normaliza para resolver variaciones (con y sin tildes, abreviaciones).
    """
    # Diccionario de escalas personalizadas por equipo
    scales = {
        'Girona FC': 0.024,
        'Athletic Club': 0.03,
        'Valencia CF': 0.028,
        'Real Madrid': 0.027,  
        'Real Sociedad': 0.027,  
        'Real Betis': 0.027,
        'FC Barcelona': 0.022,
        'RCD Español': 0.047,
        'Villarreal CF': 0.08,
        'Sevilla FC': 0.053,
        'Rayo Vallecano': 0.06,
        'RC Celta de Vigo': 0.075,
        'CD Leganes': 0.07,
        'UD Las Palmas': 0.118,
        'RCD Mallorca': 0.12,
        'CA Osasuna': 0.095,
        'Deportivo Alaves': 0.09,
        'Real Valladolid': 0.095,
        'Getafe CF': 0.099,
    }

    # Normalizar el nombre del equipo: Eliminar espacios extras y convertir a minúsculas
    team_name_normalized = team_name.strip().lower()

    # Si el nombre del equipo está en el mapeo, usar el nombre normalizado
    if team_name_normalized in team_mapping:
        team_name = team_mapping[team_name_normalized]
    
    # Buscar el nombre del equipo en el mapeo para obtener la información
    team_info = None
    for mapped_name, info in team_mapping.items():
        if mapped_name.lower() == team_name.lower():
            team_info = info
            break
    
    if team_info is None:
        return None
    
    # Obtener la ruta del logo
    logo_path = team_info.get('logo_path', '')
    
    # Eliminar comillas simples si existen
    if logo_path.startswith("'") and logo_path.endswith("'"):
        logo_path = logo_path[1:-1]
    
    # Obtener el zoom para el escalado
    zoom = scales.get(team_name, default_scale)
    
    try:
        if logo_path:
            # Verificar si la ruta es relativa
            path_obj = Path(logo_path)
            if not path_obj.is_absolute():
                # Buscar en diferentes ubicaciones relativas
                possible_paths = [
                    path_obj,
                    Path("assets/escudos") / path_obj.name,
                    Path("assets") / path_obj,
                    Path("assets/escudos") / Path(logo_path).name
                ]
                
                for p in possible_paths:
                    if p.exists():
                        logo_path = str(p)                        
                        break
            
            if Path(logo_path).exists():
                img = plt.imread(logo_path)
                return OffsetImage(img, zoom=zoom, alpha=1)
            else:
                print(f"❌ No se pudo encontrar el archivo: {logo_path}")
        else:
            print(f"❌ Ruta de logo vacía para {team_name}")
    except Exception as e:
        import traceback
        traceback.print_exc()
    
    return None

def create_match_timeline(df_tm, team_mapping):
    """
    Crea un timeline de partidos del Atlético usando matplotlib
    
    Args:
        df (pd.DataFrame): DataFrame con los datos de partidos
        team_mapping (dict): Diccionario de mapeo de equipos
    
    Returns:
        fig: Figura de matplotlib
    """
    # Diccionario manual que vincula la jornada con el escudo correspondiente
    journey_teams = {
        3: "assets/images/logos/esp.png",  # Jornada 3, Escudo del RCD Espanyol
        10: "assets/images/logos/leg.png", # Jornada 10, Escudo del CD Leganés
        14: "assets/images/logos/ala.png", # Jornada 14, Escudo del Deportivo Alavés
        20: "assets/images/logos/leg.png", # Jornada 19, Escudo del CD Leganés
        29: "assets/images/logos/esp.png"  # Jornada 21, Escudo del RCD Espanyol
    }

    # Crear figura y ejes explícitamente
    fig, ax = plt.subplots(figsize=(12, 9), facecolor='#d4d4d4')
    ax.set_facecolor('#d4d4d4')

    # Añadir un borde azul del Atleti
    fig.patch.set_edgecolor('#272E61')
    fig.patch.set_linewidth(2)  # Grosor del borde

    # Configuración de colores y barras
    colors = {'W': 'green', 'D': 'orange', 'L': 'red'}
    bar_width = 0.5

    # Ajuste de zoom para escudos específicos
    zoom_levels = {
        3: 0.05,  # Jornada 3, Español
        10: 0.05, # Jornada 10, Leganés
        14: 0.05, # Jornada 14, Alavés
        20: 0.05, # Jornada 20, Leganés
        29: 0.05  # Jornada 29, Español
    }

    # Dibujar barras y escudos
    for idx, row in df_tm.iterrows():
        height = row['points']
        ax.bar(row['jornada'], height, color=colors[row['result']], alpha=0.7, width=bar_width)
        
        # Añadir barra roja para derrotas
        if row['result'] == 'L':
            ax.vlines(x=row['jornada'], ymin=-0.3, ymax=0, color='red', linewidth=8)

        y_pos = height + 0.1
        
        # Verificar si la jornada tiene un mapeo directo
        if row['jornada'] in journey_teams:
            logo_path = journey_teams[row['jornada']]  # Usar el logo de la jornada específica
            zoom = zoom_levels.get(row['jornada'], 0.07)  # Usar el zoom correspondiente para cada jornada
        else:
            # Si no tiene un mapeo específico, usar la función normal de get_team_logo
            logo_path = get_team_logo(row['opponent_display'], team_mapping)
            zoom = 0.08  # Tamaño por defecto

        if logo_path:
            # Verificar si logo_path es una cadena, porque OffsetImage no tiene el método exists
            if isinstance(logo_path, str) and Path(logo_path).exists():
                # Leer la imagen usando mpimg.imread() para obtener la imagen
                img = mpimg.imread(logo_path)
                # Crear un objeto OffsetImage con la imagen leída
                ab = AnnotationBbox(OffsetImage(img, zoom=zoom), 
                                    (row['jornada'], y_pos), frameon=False, box_alignment=(0.5, 0.5))
                ax.add_artist(ab)
            else:
                # Si logo_path ya es un OffsetImage, directamente añadimos al gráfico
                ab = AnnotationBbox(logo_path, 
                                    (row['jornada'], y_pos), frameon=False, box_alignment=(0.5, 0.5))
                ax.add_artist(ab)

        # Para local/visitante
        ax.text(row['jornada'], -1.05,
               'L' if row['location'] == 'Local' else 'V',
               ha='center', va='center',
               fontsize=10,
               weight='bold',
               color='darkblue')

        result_color = colors[row['result']]
        ax.text(row['jornada'], height + 0.6,
               row['score'],
               ha='center',
               va='bottom',
               fontsize=10,
               weight='bold',
               color=result_color)
        
        # Añadir fecha del partido
        ax.text(row['jornada'], -2,
               row['date'],
               ha='center',
               va='center',
               fontsize=8,
               weight='bold',
               color='darkblue',
               rotation=60)

    # Configuraciones adicionales del gráfico
    # Eliminar spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Configurar eje Y - quitar las marcas
    ax.set_yticks([0, 1, 3])
    ax.set_yticklabels(['0', '1', '3'])  # Etiquetas vacías para quitar los números
    ax.tick_params(axis='y', colors='darkblue', size=0)

    # Ajustar eje X para mostrar solo jornadas del 1 al 25
    ax.set_xticks(range(1, 27))
    ax.set_xticklabels(range(1, 27), color='darkblue', size=8, weight='bold') 
    ax.tick_params(axis='x', colors='darkblue', size=0)

    # Título personalizado
    plt.title('Atlético de Madrid 24/25', 
             color='darkblue', 
             fontsize=18, 
             fontweight='bold', 
             pad=65)  # Aumentar el pad para subir el título

    # Calcular estadísticas
    total_matches = len(df_tm)
    total_points = df_tm['points'].sum()
    wins = len(df_tm[df_tm['result'] == 'W'])
    draws = len(df_tm[df_tm['result'] == 'D'])
    losses = len(df_tm[df_tm['result'] == 'L'])

    # Añadir estadísticas en la parte inferior
    stats_text = (
        f"Partidos disputados: {total_matches}\n"
        f"Puntos totales: {total_points}\n"
        f"Victorias: {wins} | Empates: {draws} | Derrotas: {losses}"
    )

    plt.text(0.9, 1.4, 
             stats_text, 
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax.transAxes,
             color='darkblue',
             fontsize=12)

    plt.tight_layout()
    
    return fig

def plot_atletico_xg_differential(df_expcGL, df1):
    """
    Genera un gráfico de barras horizontales para visualizar la diferencia de xG por partido
    del Atlético de Madrid con las jornadas invertidas, considerando solo partidos jugados.
    """
    # Asegurarnos de que trabajamos solo con los partidos jugados
    df_expcGL = df_expcGL.copy()
    
    # Obtener las etiquetas de jornadas para los partidos jugados
    jornada_labels = df1['jornada'].tolist()[:len(df_expcGL)]
    
    # Invertir el orden para mostrar jornada 1 arriba
    df_expcGL_inverted = df_expcGL.iloc[::-1].reset_index(drop=True)
    jornada_labels_inverted = jornada_labels.copy()
    jornada_labels_inverted.reverse()
    
    # Crear posiciones Y para el gráfico
    y_positions = np.arange(1, len(df_expcGL_inverted) + 1)
    
    # Separar diferencias positivas y negativas
    df_expcGL_pos = df_expcGL_inverted[df_expcGL_inverted['xGdif'] > 0].copy()
    df_expcGL_neg = df_expcGL_inverted[df_expcGL_inverted['xGdif'] < 0].copy()
    
    # Asignar posiciones Y a cada fila
    df_expcGL_inverted['y_pos'] = y_positions
    df_expcGL_pos = df_expcGL_inverted[df_expcGL_inverted['xGdif'] > 0].copy()  
    df_expcGL_neg = df_expcGL_inverted[df_expcGL_inverted['xGdif'] < 0].copy()  
    
    # Configurar gráfico
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(figsize=(6, 3))
    
    # Cambiar fondo
    fig.patch.set_facecolor('#d4d4d4')
    ax.set_facecolor('#d4d4d4')

    # Añadir un borde azul del Atleti
    fig.patch.set_edgecolor('#272E61')
    fig.patch.set_linewidth(1)  # Grosor del borde

    # Eliminar el borde del gráfico
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Dibujar barras
    bar_height = 0.7
    plt.hlines(y=df_expcGL_pos['y_pos'], xmin=0, xmax=df_expcGL_pos['xGdif'], 
               color='green', alpha=0.7, linewidth=bar_height*9)
    plt.hlines(y=df_expcGL_neg['y_pos'], xmin=0, xmax=df_expcGL_neg['xGdif'], 
               color='red', alpha=0.7, linewidth=bar_height*9)
    
    # Configuración de ejes
    ax.tick_params(axis='x', colors='darkblue')
    ax.tick_params(axis='y', colors='darkblue')
    plt.xticks([-3, -2, -1, 0, 1, 2, 3, 4, 5], fontsize=6, weight='bold')
    
    # Aplicar formato correcto a las jornadas (solo J1, J2, J3...)
    jornada_labels_inverted = [f"J{j}" for j in range(1, len(jornada_labels_inverted) + 1)]
    plt.yticks(y_positions, jornada_labels_inverted, rotation='horizontal', fontsize=6, color='darkblue', weight='bold')
    
    # Ajustes visuales
    ax.tick_params(axis='y', pad=1, which='both', left=True)
    plt.gca().xaxis.grid(False)
    plt.gca().yaxis.grid(False)
    
    # Anotaciones
    for i, row in df_expcGL_pos.iterrows():
        plt.annotate(f"{row['xGdif']:.1f}", 
                     (row['xGdif'] + 0.2, row['y_pos']),
                     c='green', size=5, ha='center', va='center', weight='bold')
    
    for i, row in df_expcGL_neg.iterrows():
        plt.annotate(f"{row['xGdif']:.1f}", 
                     (row['xGdif'] - 0.2, row['y_pos']),
                     c='red', size=5, ha='center', va='center', weight='bold')
    
    # Títulos
    fig_text(0.25, 1.01, s="Atletico de Madrid 24-25", fontsize=12, weight='bold', color="darkblue")
    fig_text(0.25, 0.95, s=" <Negativo xG>     <Positivo xG>", 
             highlight_textprops=[{"color":'red'}, {'color':"green"}], 
             fontsize=8, fontweight="bold")
    
    # Etiqueta eje X
    fig_text(0.33, 0.005, s="Diferencial xG", fontsize=5, fontweight="bold", color="darkblue")
    
    # Ajustar márgenes
    plt.subplots_adjust(left=0.1, right=0.92, top=0.90, bottom=0.05)
   
    return fig