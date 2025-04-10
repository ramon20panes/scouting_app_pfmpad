import streamlit as st

def load_all_styles():
    """Carga todos los estilos CSS de la aplicación"""
    
    st.markdown("""
        <style>
        /* Variables CSS para mantener consistencia */
        :root {
            --primary-color: #272E61;
            --primary-font: 'Roboto Slab', serif;
            --spacing-sm: 0.5rem;
            --spacing-md: 1rem;
            --spacing-lg: 1.5rem;
        }
        
        /* Control inmediato de sidebar y navegación */
        [data-testid="stSidebar"][aria-expanded="false"],
        [data-testid="stSidebar"][aria-expanded="true"],
        div[data-testid="collapsedControl"],
        section[data-testid="stSidebarNav"],
        button[kind="menuButton"],
        .stDeployButton,
        div[class^="stToolbar"] {
            display: none !important;
            opacity: 0 !important;
            visibility: hidden !important;
            height: 0 !important;
            position: absolute !important;
            z-index: -1 !important;
            margin: 0 !important;
            padding: 0 !important;
        }        
        
        /* Importar fuente similar a Rockwell */
        @import url('https://fonts.googleapis.com/css2?family=Roboto+Slab:wght@400;700&display=swap');
        
        /* Aplicar fuente a toda la aplicación */
        * {
            font-family: var(--primary-font) !important;
        }
        
        /* Estilos principales */
        .main-title {
            text-align: left;
            font-size: 2.5em;
            margin-bottom: 0;
            color: var(--primary-color);
            font-weight: bold;
        }
        .subtitle {
            text-align: left;
            font-size: 1.8em;
            color: var(--primary-color);
            margin-top: 0;
            font-weight: bold;
        }
        
        /* Contenedor del autor */
        .author-container {
            position: fixed;
            right: 20px;
            bottom: 60px;
            text-align: right;
            color: var(--primary-color);
            font-weight: bold;
            z-index: 1000;
        }
        
        /* Contenedor del footer */
        .footer-container {
            position: fixed;
            bottom: 0;
            right: 0;
            padding: 10px;
            z-index: 1000;
        }
        .footer img {
            max-width: 200px;
        }
        
        /* Estilos de login */
        .login-box {
            max-width: 300px;
            margin: 0 auto;
            padding: 10px;
        }
        div[data-testid="stTextInput"] input {
            max-width: 200px !important;
            display: block;
        }
        div[data-testid="stTextInput"] {
            max-width: 280px !important;
            margin: 0 auto !important;
        }
        button[kind="primary"] {
            max-width: 120px !important;
            margin: 0 auto !important;
            display: block !important;
        }
        div[data-baseweb="notification"] {
            width: 100% !important;
            max-width: 300px !important;
            margin: 0 auto !important;
        }
        
        /* Estilos globales */
        .stTextInput input, .stTextInput label {
            color: var(--primary-color) !important;
            font-size: 0.9em !important;
            max-width: 200px !important;
        }
        
        /* Color global de texto */
        .st-emotion-cache-*, div, p, h1, h2, h3, label {
            color: var(--primary-color) !important;
        }
        
        /* Contenedores flexibles responsivos */
        .responsive-container {
            display: flex;
            flex-wrap: wrap;
            gap: var(--spacing-md);
        }
        
        /* Reducir espaciado general en toda la aplicación */
        .block-container {
            padding-top: var(--spacing-sm) !important;
            padding-bottom: 0 !important;
            margin-top: -15px !important;
        }
    
        /* Reducir espacios entre elementos */
        div[data-testid="stVerticalBlock"] > div {
            margin-bottom: 0.2rem !important;
        }
    
        /* Reducir tamaño de títulos */
        h1 {
            font-size: 1.8rem !important;
            margin-bottom: 0.2rem !important;
        }
    
        h2 {
            font-size: 1.5rem !important;
            margin-bottom: 0.2rem !important;
        }
    
        h3, h4, h5 {
            font-size: 1.2rem !important;
            margin-bottom: 0.1rem !important;
        }
    
        /* Reducir espacio en elementos de formulario */
        div[data-testid="stFormSubmitButton"] {
            margin-top: var(--spacing-sm) !important;
        }
    
        /* Compactar selectores múltiples */
        div[data-testid="stMultiSelect"] {
            margin-bottom: 0.3rem !important;
        }
                
        /* Ajustar espacio del encabezado */
        div.stHorizontalBlock:first-child {
            margin-top: 15px !important;
            padding-top: 10px !important;
        }
    
        /* Ajustar espacio para el logo */
        div.stHorizontalBlock:first-child div.column:nth-child(2) {
            padding-top: 10px !important;
        }
        
        /* Estilos para botones de navegación y acción */
        .nav-button, .action-button {
            background-color: var(--primary-color) !important;
            color: white !important;
            font-weight: bold !important;
            border-radius: 5px !important;
            padding: 0.5rem 1rem !important;
            margin: 0.3rem !important;
            text-align: center !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
        }
        
        .nav-button:hover, .action-button:hover {
            opacity: 0.9 !important;
            transform: scale(1.02) !important;
        }
        
        /* Estilos específicos para componentes de análisis */
        .metrics-container {
            background-color: #f5f5f5 !important;
            border-radius: 10px !important;
            padding: 1rem !important;
            margin-bottom: 1rem !important;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1) !important;
        }
        
        /* Ajustes responsivos mejorados */
        @media (max-width: 1200px) {
            .main-title { font-size: 2.2em; }
            .subtitle { font-size: 1.6em; }
            .responsive-container > div {
                flex: 1 1 48%;
            }
        }
        
        @media (max-width: 768px) {
            .main-title { font-size: 1.8em; }
            .subtitle { font-size: 1.4em; }
            div[data-testid="stImage"] img { max-width: 150px !important; }
            .responsive-container > div {
                flex: 1 1 100%;
            }
        }
        
        @media (max-width: 480px) {
            .main-title { font-size: 1.5em; }
            .subtitle { font-size: 1.2em; }
            div[data-testid="stImage"] img { max-width: 100px !important; }
            
            /* Ajustes para footer en móvil */
            .footer-container {
                position: relative;
                margin-top: 2rem;
                right: auto;
                bottom: auto;
            }
            .footer img {
                max-width: 150px;
            }
            
            /* Ajustes para autor en móvil */
            .author-container {
                position: relative;
                margin-top: 1rem;
                right: auto;
                bottom: auto;
            }
        }
        </style>
    """, unsafe_allow_html=True)