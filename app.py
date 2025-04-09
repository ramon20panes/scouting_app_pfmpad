# app.py
import streamlit as st
import os
import base64
from pathlib import Path
from utils.styles import load_all_styles
from utils.auth import login, check_auth
from common.session import init_session_state

# Configuración de la página
st.set_page_config(
    page_title="Atlético de Madrid - Scouting",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Cargar estilos al principio del archivo
load_all_styles()

# Estilo global para letras y reducir espacios verticales
st.markdown("""
    <style>
    h1, h2, h3, h4, h5, h6, p, div, span, label, input, button {
        color: #001F3F !important;
        font-weight: bold !important;
    }
    
    /* Reducir espacios verticales */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
    }
    
    .stButton button {
        margin-top: 10px !important;
    }
    
    /* Forzar que el footer aparezca */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        padding: 10px 20px;
        background-color: #f8f9fa;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    </style>
""", unsafe_allow_html=True)

# Rutas de las imágenes
ESCUDO_PATH = Path("assets/images/logos/atm.png")
FOOTER_PATH = Path("assets/images/others/footer.png")

def base64_image(image_path):
    """Convierte imagen a base64 para insertarla en HTML"""
    if not os.path.exists(image_path):
        return ""
    
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Inicializar estado de sesión
init_session_state()

# Ocultar menú de hamburguesa y demás elementos
hide_menu = """
<style>
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_menu, unsafe_allow_html=True)

# Ocultar la sidebar
st.markdown("""
    <style>
    [data-testid="stSidebar"][aria-expanded="false"],
    [data-testid="stSidebar"][aria-expanded="true"],
    div[data-testid="collapsedControl"] {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# Nueva estructura: escudo a la izquierda y título centrado arriba
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    if os.path.exists(ESCUDO_PATH):
        st.image(str(ESCUDO_PATH), width=200) 
    else:
        st.error(f"No se encontró la imagen: {ESCUDO_PATH}")

with col2:
    st.markdown("<h1 style='text-align: center; margin-top: -30px;'>Club Atlético de Madrid</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; margin-top: -15px;'>Dirección Deportiva</h2>", unsafe_allow_html=True)

# Reducir espacio antes del formulario
st.markdown("<div style='margin-top: -30px;'></div>", unsafe_allow_html=True)

# Login y navegación
if not check_auth():
    # Formulario de login centrado
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        login()
        st.markdown('</div>', unsafe_allow_html=True)
else:
    # Ya autenticado, realizar redirección si es necesario
    if "redirect_to" in st.session_state and st.session_state.redirect_to:
        target_page = st.session_state.redirect_to
        st.session_state.redirect_to = None  # Limpiar
        st.switch_page(f"pages/{target_page}.py")

# Footer
st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; position: fixed; bottom: 0; right: 0; width: 100%; padding: 10px 20px; background-color: #f8f9fa;">
        <div>
            <strong>Ramón González</strong><br>
            Mod11 MPAD
        </div>
""", unsafe_allow_html=True)
