import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import base64
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import unicodedata
from utils.auth import logout

# Configuración de la página debe ser lo primero
st.set_page_config(
    page_title="Jugadores Similares - Atlético de Madrid",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded")

from utils.auth import check_auth, logout
from utils.styles import load_all_styles

from utils.export_pdf import export_to_pdf, dataframe_a_pdf_contenido
from utils.auth import check_auth, check_player_access, get_user_role
 
# Obtener la ruta absoluta al directorio raíz del proyecto
current_dir = os.path.dirname(os.path.abspath(__file__))  # Directorio de la página 7
project_root = os.path.dirname(current_dir)  # Directorio raíz del proyecto

# Añadir el directorio raíz al path de Python
sys.path.insert(0, project_root)

# Cargar estilos al principio del archivo
load_all_styles()

# Reducir espacios verticales
st.markdown("""
<style>
div.block-container {padding-top: 0.5rem; padding-bottom: 0rem;}
div[data-testid="stVerticalBlock"] > div {margin-bottom: 0rem;}
.element-container {margin-bottom: 0.5rem !important;}
h3 {margin-top: 0.5rem !important; margin-bottom: 0.5rem !important;}
h4 {margin-top: 0.5rem !important; margin-bottom: 0.5rem !important;}
p {margin-bottom: 0.3rem !important;}
</style>
""", unsafe_allow_html=True)

# Importar módulos de predicción
try:
    from models.similar_players.predict import (
        cargar_recursos,
        obtener_jugadores_atletico,
        obtener_jugadores_similares,
        generar_grafico_similares,
        verificar_modelos,
        entrenar_modelos_si_necesario
    )
    
except ImportError as e:
    st.error(f"Error al importar módulos: {e}")
    print(f"Error detallado: {e}")

# Autenticación
if not check_auth():
    st.warning("No estás autenticado.")
    st.stop()

# Bloquear acceso si el rol es "player", no lo pueden ver los jugadores
if get_user_role() == "player":
    st.markdown("<br>", unsafe_allow_html=True)
    st.error("⛔ Esta sección no está disponible para jugadores.")
    st.stop()

# Agregar ruta raíz al path para importar módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def app():
    # Título y logo
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<h2 class="main-title">Búsqueda jugadores similares</h2>', unsafe_allow_html=True)
        st.markdown("<h4>Alternativas integrantes actual plantilla</h4>", unsafe_allow_html=True)
    
    with col2:
        # Cargar el logo del Atlético de Madrid
        try:
            logo = Image.open("assets/images/logos/atm.png")
            st.image(logo, width=100)
        except FileNotFoundError:
            try:
                # Intentar rutas alternativas
                alt_paths = [
                    "assets/escudos/atm.png",
                    "assets/logos/atm.png",
                    "assets/images/atm.png"
                ]
                for path in alt_paths:
                    if Path(path).exists():
                        logo = Image.open(path)
                        st.image(logo, width=100)
                        break
                else:
                    st.warning("Logo no encontrado. Verifica la ruta de los logos.")
            except Exception as e:
                st.warning(f"No se pudo cargar el logo: {str(e)}")
    
    # Verificar si los modelos están entrenados
    if not verificar_modelos():
        with st.spinner("Entrenando modelos. Por favor, espere..."):
            success = entrenar_modelos_si_necesario()
            if not success:
                st.error("No se pudieron entrenar los modelos. Verifica los datos de entrada.")
                st.stop()
    
    # Cargar recursos para predicciones
    with st.spinner("Cargando recursos..."):
        recursos = cargar_recursos()
    
    if not recursos:
        st.error("No se pudieron cargar los recursos necesarios.")
        st.stop()
    
    # Obtener jugadores del Atlético
    df_atletico = obtener_jugadores_atletico()
    
    if df_atletico.empty:
        st.error("No se encontraron datos de jugadores del Atlético.")
        st.stop()
    
    # Preparar lista de jugadores para el selector
    jugadores_atletico = df_atletico['Nombre'].tolist()
    jugadores_atletico.sort()  # Ordenar alfabéticamente
    
    # Selector de jugador (con Julián Álvarez como predeterminado)
    jugador_default = "Julian Alvarez" if "Julian Alvarez" in jugadores_atletico else jugadores_atletico[0]

    jugador_seleccionado = st.selectbox(
        "Selecciona un jugador del Atlético de Madrid:",
        jugadores_atletico,
        index=jugadores_atletico.index(jugador_default) if jugador_default in jugadores_atletico else 0
    )

    # Correcciones manuales para nombres con alias o letras especiales
    mapeo_nombres_similares = {
        "Koke": "Jorge Resurrección",
        "Jorge Resurrección": "Koke",
        "Sorloth": "Alexander Sørloth",
        "Alexander Sorloth": "Alexander Sørloth",
        "Aleksander Sorloth": "Alexander Sørloth"
    }

    # Normalizar si hay coincidencia
    if jugador_seleccionado in mapeo_nombres_similares:
        jugador_seleccionado = mapeo_nombres_similares[jugador_seleccionado]    
    
    # Número de jugadores similares a mostrar
    num_similares = st.slider(
        "Número de jugadores similares a mostrar:",
        min_value=4, 
        max_value=12, 
        value=8
    )
    
    # Botón para buscar jugadores similares
    if st.button("Buscar Jugadores Similares", key="search_button", type="primary"):
        with st.spinner(f"Buscando jugadores similares a {jugador_seleccionado}..."):
            # Obtener jugadores similares
            result = obtener_jugadores_similares(
                jugador_seleccionado,
                recursos,
                top_n=num_similares
            )
            
            if not result:
                st.error(f"No se encontraron jugadores similares para {jugador_seleccionado}.")
                st.stop()
            
            # Mostrar información del jugador seleccionado
            jugador_info = df_atletico[df_atletico['Nombre'] == jugador_seleccionado].iloc[0]
            
            # Diseño de dos columnas: info del jugador y gráfico (proporción ajustada)
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.markdown(f"<h3 style='margin-bottom:5px;'>{jugador_seleccionado}</h3>", unsafe_allow_html=True)
                
                # Foto del jugador
                ruta_foto = jugador_info.get('ruta_foto', None)
                if ruta_foto and os.path.exists(ruta_foto):
                    imagen = Image.open(ruta_foto)
                    st.image(imagen, width=250, use_container_width=True)
                else:
                    # Intentar buscar la foto en assets/players
                    posibles_rutas = [
                        f"assets/players/{jugador_seleccionado.lower().replace(' ', '_')}.png",
                        f"assets/players/{jugador_seleccionado.lower().replace(' ', '_')}.jpg",
                        f"assets/players/{jugador_info.get('Nombre_corto', '').lower()}.png",
                        f"assets/players/{jugador_info.get('Nombre_corto', '').lower()}.jpg"
                    ]
                    
                    foto_encontrada = False
                    for ruta in posibles_rutas:
                        if os.path.exists(ruta):
                            try:
                                imagen = Image.open(ruta)
                                st.image(imagen, width=250, use_column_width=True)
                                foto_encontrada = True
                                break
                            except:
                                continue
                    
                    if not foto_encontrada:
                        st.info(f"No se encontró la foto de {jugador_seleccionado}")
                
                # Información básica del jugador en formato más compacto
                st.markdown("<h4 style='margin-top:5px; margin-bottom:5px;'>Datos básicos</h4>", unsafe_allow_html=True)
                
                # Usar una tabla más compacta para los datos
                col1a, col1b = st.columns(2)
                with col1a:
                    st.markdown("**Posición:**")
                    st.markdown("**Dorsal:**")
                    st.markdown("**Nacionalidad:**")
                    st.markdown("**Año Nacimiento:**")
                with col1b:
                    st.markdown(f"{jugador_info.get('Posicion', 'N/A')}")
                    st.markdown(f"{jugador_info.get('Numero', 'N/A')}")
                    st.markdown(f"{jugador_info.get('Nacionalidad', 'N/A')}")
                    st.markdown(f"{jugador_info.get('Nacimiento', 'N/A')}")
            
            with col2:
                # Generar gráfico de jugadores similares a mayor tamaño
                st.markdown("<div style='height:400px;'>", unsafe_allow_html=True)
                fig = generar_grafico_similares(result, recursos)
                if fig:
                    st.pyplot(fig, use_container_width=True)
                else:
                    st.warning("No se pudo generar el gráfico de jugadores similares.")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Mostrar jugadores similares como tarjetas (4 por fila)
            st.markdown("<h2 style='margin-top:10px;'>Jugadores Similares</h2>", unsafe_allow_html=True)
            
            # Obtener todos los jugadores similares
            similares_df = result['similares']
            total_jugadores = min(len(similares_df), 12)  # Limitamos a 12 jugadores máximo
            
            # Definir cuántos jugadores mostrar por fila
            jugadores_por_fila = 4
            
            # Usar un diseño más compacto para las tarjetas
            st.markdown("""
            <style>
            .compact-card {
                padding: 8px;
                margin-bottom: 10px;
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .jugador-nombre {
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 2px;
                color: #272E61;
            }
            .jugador-equipo {
                font-size: 14px;
                color: #666;
                margin-bottom: 5px;
            }
            .dato-label {
                font-size: 12px;
                color: #777;
                margin-bottom: 0;
            }
            .dato-valor {
                font-size: 14px;
                font-weight: bold;
                margin-bottom: 5px;
            }
            .similitud-valor {
                font-size: 16px;
                font-weight: bold;
                color: #d81e05;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Crear filas de jugadores
            for fila in range((total_jugadores + jugadores_por_fila - 1) // jugadores_por_fila):
                # Crear columnas para esta fila
                cols = st.columns(jugadores_por_fila)
                
                # Para cada columna en esta fila
                for col_idx in range(jugadores_por_fila):
                    # Calcular el índice global del jugador
                    jugador_idx = fila * jugadores_por_fila + col_idx
                    
                    # Verificar si todavía hay jugadores disponibles
                    if jugador_idx < total_jugadores:
                        jugador = similares_df.iloc[jugador_idx]
                        
                        with cols[col_idx]:
                            # Usar el st.expander para crear tarjetas compactas
                            with st.container():
                                # Encabezado con nombre y posición
                                st.markdown(f"""
                                <div class="compact-card">
                                    <div class="jugador-nombre">{jugador['Nombre']}</div>
                                    <div class="jugador-equipo">{jugador['Equipo']} | {result['posicion']}</div>
                                    <hr style="margin: 5px 0;">
                                    <div style="display: flex; justify-content: space-between;">
                                        <div style="width: 48%;">
                                            <p class="dato-label">Nacimiento</p>
                                            <p class="dato-valor">{int(jugador['Nacimiento']) if 'Nacimiento' in jugador else 'N/A'}</p>
                                        </div>
                                        <div style="width: 48%;">
                                            <p class="dato-label">Contrato</p>
                                            <p class="dato-valor">{jugador['Contrato'] if 'Contrato' in jugador else 'N/A'}</p>
                                        </div>
                                    </div>
                                    <p class="dato-label">Similitud</p>
                                    <p class="similitud-valor">{100 - round(jugador['distancia'] * 20, 1)}%</p>
                                """, unsafe_allow_html=True)
                                
                                # Métricas específicas
                                if 'metricas' in result:
                                    metricas_mostradas = 0
                                    for metrica in result['metricas']:
                                        if metrica in jugador and metricas_mostradas < 2:
                                            nombre_metrica = metrica.replace('_original', '').replace('_', ' ').title()
                                            valor_metrica = round(jugador[metrica], 2)
                                            
                                            # Mostrar métrica
                                            st.markdown(f"""
                                            <p class="dato-label">{nombre_metrica}</p>
                                            <p class="dato-valor">{valor_metrica}</p>
                                            """, unsafe_allow_html=True)
                                            
                                            metricas_mostradas += 1
                                
                                st.markdown("</div>", unsafe_allow_html=True)
            
            # Nombre del archivo PDF
            pdf_filename = f"similares_{jugador_seleccionado.replace(' ', '_').lower()}.pdf"
            
            # Crear DataFrame para exportar (solo los similares)
            similares_df = result['similares'].copy()
            
            # Seleccionar solo columnas que existen en similares_df
            columnas_disponibles = similares_df.columns.tolist()
            columnas_deseadas = ['Nombre', 'Equipo', 'Contrato', 'Nacimiento', 'distancia'] + result.get('metricas', [])
            columnas_exportar = [col for col in columnas_deseadas if col in columnas_disponibles]
            
            # Ahora usamos solo columnas que existen
            df_export = similares_df[columnas_exportar]
            
            # Renombrar columnas para que se vean más chulas en el PDF
            nombres_nuevos = {
                'Nombre': 'Jugador',
                'Equipo': 'Equipo',
                'distancia': 'Similitud'
            }
            
            # Añadir renombres condicionales
            if 'Nacimiento' in df_export.columns:
                nombres_nuevos['Nacimiento'] = 'Año nacimiento'
            if 'Contrato' in df_export.columns:
                nombres_nuevos['Contrato'] = 'Fin de contrato'
                
            df_export = df_export.rename(columns=nombres_nuevos)
            
            # Formato de similitud en %
            if 'Similitud' in df_export.columns:
                df_export['Similitud'] = (100 - df_export['Similitud'] * 20).round(1).astype(str) + '%'
            
            contenido_pdf = dataframe_a_pdf_contenido(
                df_export,
                columnas=df_export.columns.tolist()
            )       
            
            # Guardar los datos para generación de PDF posterior
            st.session_state.df_export_similares = df_export
            st.session_state.pdf_filename_similares = pdf_filename
    else:
        st.info("🔍 Para generar el informe PDF, primero selecciona un jugador y haz clic en 'Buscar jugadores similares'.")
    
    # Explicación de la metodología
    with st.expander("Metodología"):
        st.markdown("""
        ### Metodología CRISP-DM para encontrar jugadores similares
        
        #### 1. Comprensión del Negocio
        - Objetivo: Encontrar jugadores similares a los del Atlético de Madrid como posibles reemplazos
        - Contexto: Apoyo a la dirección deportiva para la planificación de plantilla
        
        #### 2. Comprensión de los Datos
        - Fuentes: Estadísticas completas de las 5 grandes ligas europeas (24/25)
        - Variables: Métricas de rendimiento específicas por posición
        
        #### 3. Preparación de los Datos
        - Limpieza de datos y manejo de valores faltantes
        - Creación de características derivadas por posición
        - Normalización de variables numéricas
        
        #### 4. Modelado
        - Agrupamiento con K-means para identificar perfiles de jugadores similares
        - Algoritmo KNN para encontrar los jugadores más cercanos en el espacio de características
        
        #### 5. Evaluación
        - Métricas de similitud basadas en distancia euclidiana
        - Validación cualitativa por expertos en fútbol
        
        #### 6. Despliegue
        - Interfaz visual para la dirección deportiva
        - Actualización periódica con nuevas estadísticas
        """)

        # --- BARRA INFERIOR CON BACK, NOMBRE, PDF, EXIT ---
    st.markdown("""<hr style='margin-top: 2rem; margin-bottom: 1rem;'>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([3, 2, 1])

    with col1:
        st.markdown("""
            <div style="text-align: center; font-size: 18px; font-weight: bold; color: #232E61;">
                Ramón González<br>Mod11 MPAD
            </div>
        """, unsafe_allow_html=True)

    with col2:
        generar_pdf = st.button("📄 Generar PDF", key="pdf_button", use_container_width=True)

        if generar_pdf:
            if "df_export_similares" in st.session_state and not st.session_state.df_export_similares.empty:
                contenido_pdf = dataframe_a_pdf_contenido(
                    st.session_state.df_export_similares,
                    columnas=st.session_state.df_export_similares.columns.tolist()
                )

                export_to_pdf(
                    nombre_pagina="Jugadores Similares",
                    contenido=contenido_pdf,
                    autor="Ramón González",
                    output_path=st.session_state.pdf_filename_similares
                )

                with open(st.session_state.pdf_filename_similares, "rb") as f:
                    pdf_bytes = f.read()

                st.success("✅ Informe PDF generado correctamente.")
                st.download_button(
                    label="⬇️ Descargar PDF",
                    data=pdf_bytes,
                    file_name=st.session_state.pdf_filename_similares,
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.warning("No hay datos disponibles para exportar. Haz una búsqueda primero.")

    with col3:
        if st.button("⏻ Exit", use_container_width=True):
            logout()


# Punto de entrada para ejecución directa
if __name__ == "__main__":
    app()