import streamlit as st

def load_all_styles():
    """Carga todos los estilos CSS de la aplicación"""

    st.markdown("""
        <style>
        /* Variables CSS para mantener consistencia */
        :root {
            --primary-color: #272E61;  /* Azul Atlético de Madrid */
            --primary-font: 'Roboto Slab', serif;
            --spacing-sm: 0.5rem;
            --spacing-md: 1rem;
            --spacing-lg: 1.5rem;
            --border-radius: 8px;  /* Radio de los bordes */
            --background-color: #D0DAD8;  /* Gris claro - fondo principal */
            --secondary-background: #B5BFC0;  /* Gris más oscuro - fondo secundario */
            --text-color: #272E61;  /* Azul para texto */
            --sidebar-background-color: #9BA3A4;  /* Gris más oscuro para la sidebar */
            --sidebar-text-color: #272E61;  /* Azul para texto en sidebar */
            --selector-border-color: #272E61;  /* Azul para bordes de selectores */
            --button-background-color: #9BA3A4;  /* Gris oscuro para botones */
            --button-text-color: #272E61;  /* Azul para texto de botones */
            --button-hover-color: #8B9394;  /* Gris más oscuro para hover */
            --pdf-button-background: #272E61;  /* Azul para botón PDF */
            --pdf-button-text: #FFFFFF;  /* Blanco para texto botón PDF */
        }
        
        /* Forzar el fondo en la aplicación completa */
        .stApp {
            background-color: var(--background-color) !important;
        }
        
        /* Configuración global para texto */
        h1, h2, h3, h4, h5, h6, p, div, span, label, input, button {
            color: var(--text-color) !important;
            font-weight: bold !important;
        }
        
        /* Estilo para la sidebar */
        [data-testid="stSidebar"] {
            background-color: var(--sidebar-background-color) !important;
            border-radius: var(--border-radius) !important;
            border-right: 6px !important;
            padding: 20px;
        }

        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3, 
        [data-testid="stSidebar"] p {
            color: var(--sidebar-text-color) !important;
        }
        
        /* Forzar el color de fondo en todos los contenedores */
        .block-container, div.stTabs [data-baseweb="tab-panel"] {
            background-color: var(--background-color) !important;
        }
        
        /* Eliminar espacios extras */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
        }
        
        div[data-testid="stVerticalBlock"] > div {
            margin-bottom: 0.3rem;
            background-color: var(--background-color) !important;
        }
        
        /* Contenedores específicos */
        .metric-card {
            background-color: white !important;
            border-radius: var(--border-radius) !important;
            padding: 15px !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24) !important;
            margin-bottom: 10px !important;
        }
        
        .metric-title {
            font-size: 14px;
            color: #555;
            margin-bottom: 5px;
        }
        
        .metric-value {
            font-size: 24px;
            font-weight: bold;
        }
        
        .player-selector {
            background-color: var(--secondary-background) !important;
            padding: 15px !important;
            border-radius: var(--border-radius) !important;
            margin-bottom: 15px !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24) !important;
        }
        
        .visualization-container {
            background-color: white !important;
            padding: 15px !important;
            border-radius: var(--border-radius) !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24) !important;
        }
        
        /* Estilo para los selectores */
        .stSelectbox, .stSlider, .stTextInput input, .stMultiSelect {
            background-color: white !important;
            color: var(--text-color) !important;
            border: 6px !important;
            border-radius: var(--border-radius) !important;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1) !important;
            padding: 8px !important;
        }

        /* Estilo para texto dentro de selectores */
        .stSelectbox div[data-baseweb="select"] span,
        .stMultiSelect div[data-baseweb="select"] span,
        div[role="listbox"] ul li {
            color: var(--text-color) !important;
            background-color: white !important;
        }

        /* Estilos para opciones en listas desplegables */
        div[data-baseweb="popover"],
        div[data-baseweb="select"] div[role="listbox"],
        div[data-baseweb="select"] div[role="listbox"] ul {
            background-color: white !important;
        }

        /* Botones generales */
        .stButton button, .stFormSubmitButton button {
            background-color: var(--button-background-color) !important;
            color: var(--text-color) !important;
            font-weight: bold !important;
            border-radius: var(--border-radius) !important;
            padding: 0.6rem 1.2rem !important;
            border: 6px !important;
            transition: background-color 0.3s ease-in-out !important;
        }

        .stButton button:hover, .stFormSubmitButton button:hover {
            background-color: var(--button-hover-color) !important;
        }
        
        /* Estilo específico para botón PDF */
        .pdf-button button {
            background-color: var(--pdf-button-background) !important;
            color: var(--pdf-button-text) !important;
        }
        
        .pdf-button button:hover {
            background-color: var(--primary-color) !important;
            opacity: 0.9 !important;
        }
        
        /* Estilos para los tabs */
        .stTabs [data-baseweb="tab-list"] {
            background-color: var(--background-color) !important;
            gap: 1px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: var(--secondary-background) !important;
            color: var(--text-color) !important;
            padding-top: 10px;
            padding-bottom: 10px;
            border-radius: var(--border-radius) !important;
        }
        
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: var(--sidebar-background-color) !important;
            font-weight: bold !important;
        }
        
        /* Estilos para tablas */
        .stTable, [data-testid="stTable"], .stDataFrame [data-testid="stDataFrameResizable"] {
            border: 6px !important;
            border-radius: var(--border-radius) !important;
            overflow: hidden !important;
        }

        /* Para los encabezados de tabla */
        .stDataFrame thead tr th {
            background-color: var(--sidebar-background-color) !important;
            color: var(--text-color) !important;
            font-weight: bold !important;
        }

        /* Para los encabezados de tabla */
        .stDataFrame thead tr th {
            background-color: var(--sidebar-background-color) !important;
            color: var(--text-color) !important;
            font-weight: bold !important;
        }

        /* Para las filas alternadas */
        .stDataFrame tbody tr:nth-child(even) {
            background-color: var(--secondary-background) !important;
        }

        /* Ajustes para encabezados y textos principales */
        h1, h2, h3, .main-title {
            color: var(--text-color) !important;
            font-weight: bold !important;
        }

        /* Espacio para título y logo */
        div.stHorizontalBlock {
            padding-top: 1.5rem !important;
            margin-bottom: 1.5rem !important;
        }
        
        /* Clases para tendencias */
        .trend-up {
            color: #4CAF50 !important;
        }
        
        .trend-down {
            color: #F44336 !important;
        }
        
        /* Estilos para dataframes */
        .stDataFrame {
            border-radius: var(--border-radius) !important;
        }
        
        /* Forzar que el footer aparezca y tenga el color correcto */
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            padding: 10px 20px;
            background-color: var(--secondary-background);
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: var(--text-color) !important;
        }
        
        .footer-container div {
            color: var(--text-color) !important;
            font-size: 14px !important;
        }

        /* Ajustes responsivos */
        @media (max-width: 768px) {
            .main-title { font-size: 1.8em; }
            div[data-testid="stImage"] img { max-width: 80px !important; }
        }
                
        /* Estilos para tarjetas de jugadores */
        .player-card {
            background-color: white;
            border-radius: var(--border-radius);
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 15px;
        }

        .player-name {
            font-size: 18px;
            font-weight: bold;
            color: var(--text-color);
        }

        .player-position {
            background-color: var(--pdf-button-background);
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
        }

        .player-team {
            font-size: 14px;
            color: #555;
            margin-bottom: 10px;
        }

        .player-stat {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
        }

        .player-stat-label {
            font-size: 14px;
            color: #555;
        }

        .player-stat-value {
            font-size: 14px;
            font-weight: bold;
            color: var(--text-color);
        }

        .player-info {
            background-color: white;
            border-radius: var(--border-radius);
            padding: 15px;
            margin-bottom: 15px;
        }

        </style>
    """, unsafe_allow_html=True)