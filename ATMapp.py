import streamlit as st
import os
import base64
from pathlib import Path
from utils.styles import load_all_styles
from utils.auth import login, check_auth, get_user_role
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

# Forzar el color de fondo principal para toda la aplicación
st.markdown("""
    <style>
    .stApp {
        background-color: #D0DAD8 !important;
    }
    
    /* Estilo para el formulario de login */
    form[data-testid="stForm"] {
        background-color: #B5BFC0 !important;
        padding: 20px !important;
        border-radius: 6px !important;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1) !important;
        margin-top: -10px !important;
    }
    
    /* Hacer los campos de entrada más pequeños */
    [data-testid="stForm"] [data-testid="stTextInput"] {
        margin-bottom: 10x !important;
    }
    
    [data-testid="stForm"] input {
        padding: 5px !important;
        height: 35px !important;
    }
    
    /* Ajuste para el botón de login */
    [data-testid="stForm"] [data-testid="baseButton-secondary"] {
        background-color: #272E61 !important;
        color: white !important;
        font-weight: bold !important;
        padding: 6px 12px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Inicializar estado de sesión
init_session_state()

# Rutas de las imágenes
ESCUDO_PATH = Path("assets/images/logos/atm.png")
FOOTER_PATH = Path("assets/images/others/footer.png")

def base64_image(image_path):
    """Convierte imagen a base64 para insertarla en HTML"""
    if not os.path.exists(image_path):
        return ""
    
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()


# Ocultar menú de hamburguesa y demás elementos
hide_menu = """
<style>
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_menu, unsafe_allow_html=True)

# Sidebar
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
        st.image(str(ESCUDO_PATH), width=120) 
    else:
        st.error(f"No se encontró la imagen: {ESCUDO_PATH}")

with col2:
    
    st.markdown('<h1 style="text-align: center; color: #272E61; font-size: 3.0rem;">Club Atlético de Madrid</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #272E61; margin-top: 0; font-size: 1.8rem;">Dirección Deportiva</h2>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Login y navegación
if not check_auth():
    # Formulario de login centrado
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        login()
        
else:
    from utils.auth import get_user_role

    # Ocultar página 7 si el usuario es jugador
    if get_user_role() == "player":
        page_7_path = "pages/7_Similar_players.py"
        hidden_path = page_7_path + ".hide"

        if os.path.exists(page_7_path):
            os.rename(page_7_path, hidden_path)
    else:
        # Restaurar la página si está oculta y el usuario no es jugador
        hidden_path = "pages/7_Similar_players.py.hide"
        original_path = hidden_path.replace(".hide", "")

        if os.path.exists(hidden_path):
            os.rename(hidden_path, original_path)

    # Redirección tras login
    if "redirect_to" in st.session_state and st.session_state.redirect_to:
        try:
            # El formato correcto es con el prefijo "pages/"
            st.switch_page("pages/1_Team_stats.py")
            st.session_state.redirect_to = None
        except Exception as e:
            st.error(f"Error al redirigir: {e}")

# Espacio antes del footer
st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

# Footer con tu código original (mantenido tal cual lo tenías)
footer_col1, footer_col2 = st.columns(2)

with footer_col1:
    st.markdown("<strong>Ramón González</strong><br>Mod11 MPAD", unsafe_allow_html=True)

with footer_col2:
    if os.path.exists(FOOTER_PATH):
        footer_img = base64_image(FOOTER_PATH)
        st.markdown(f"""
            <div style="text-align: right;">
                <img src="data:image/png;base64,{footer_img}" style="max-height: 50px;">
            </div>
        """, unsafe_allow_html=True)