import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Función para cargar los datos de los jugadores
def load_player_data(file_path):
    try:
        # Cargar el archivo CSV de los jugadores
        df = pd.read_csv(file_path)
        # Convertir las columnas de la jornada a enteros si es necesario
        df['Jornada'] = df['Jornada'].astype(int)
        return df
    except Exception as e:
        raise Exception(f"Error al cargar el archivo de datos: {e}")

# Función para limpiar los datos (manejar valores NaN, errores de tipo, etc.)
def clean_player_data(df):
    df = df.dropna(subset=['Nombre'])  # Eliminar filas sin nombres de jugadores
    df['Edad'] = pd.to_numeric(df['Edad'], errors='coerce')  # Asegurar que la edad sea numérica
    df['Min'] = pd.to_numeric(df['Min'], errors='coerce')  # Asegurar que los minutos sean numéricos
    df['Gls'] = pd.to_numeric(df['Gls'], errors='coerce')  # Asegurar que los goles sean numéricos
    df['Asist'] = pd.to_numeric(df['Asist'], errors='coerce')  # Asegurar que las asistencias sean numéricas
    # Añadir más columnas de métricas si es necesario
    return df

# Función para calcular la diferencia de una métrica entre jornadas
def calculate_improvement(df, metric):
    df_sorted = df.sort_values(by=['Nombre', 'Jornada'])
    df_sorted['metric_diff'] = df_sorted.groupby('Nombre')[metric].diff()
    return df_sorted

# Función para calcular el promedio de una métrica para cada jugador
def calculate_player_avg(df, metric):
    return df.groupby('Nombre')[metric].mean()

# Función para generar gráfico de evolución de una métrica por jornada
def plot_metric_evolution(df, player_name, metric):
    player_data = df[df['Nombre'] == player_name]
    plt.figure(figsize=(10, 6))
    plt.plot(player_data['Jornada'], player_data[metric], marker='o', label=player_name)
    plt.title(f"Evolución de {metric} para {player_name}")
    plt.xlabel("Jornada")
    plt.ylabel(metric)
    plt.grid(True)
    plt.legend()
    plt.show()

# Función para generar gráfico de barras comparando jugadores
def plot_player_metric(df, metric):
    players_avg = calculate_player_avg(df, metric)
    players_avg_sorted = players_avg.sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    players_avg_sorted.plot(kind='barh', color='green')
    plt.title(f"Comparación de jugadores por {metric}")
    plt.xlabel(metric)
    plt.ylabel("Jugador")
    plt.show()

# Función para generar ranking de jugadores basado en una métrica
def generate_rankings(df, metric):
    player_avg = calculate_player_avg(df, metric)
    player_ranking = player_avg.sort_values(ascending=False)
    return player_ranking

# Función para generar indicadores de mejora/empeoramiento
def improvement_indicator(df, player_name, metric):
    player_data = df[df['Nombre'] == player_name]
    player_data['Improvement'] = player_data[metric].diff().apply(lambda x: '⬆️' if x > 0 else '⬇️')
    return player_data[['Jornada', metric, 'Improvement']]

# Función para generar un heatmap de las métricas de los jugadores
def plot_metric_heatmap(df, metric):
    pivot_df = df.pivot_table(index='Nombre', columns='Jornada', values=metric)
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title(f"Heatmap de {metric} por jugador y jornada")
    plt.show()
