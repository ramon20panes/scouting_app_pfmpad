# common/session.py
import streamlit as st

def init_session_state():
    """Inicializa todas las variables de estado de la sesión"""
    
    # Estado de autenticación
    if "authentication_status" not in st.session_state:
        st.session_state.authentication_status = False
    
    # Información del usuario
    if "username" not in st.session_state:
        st.session_state.username = None
    
    if "name" not in st.session_state:
        st.session_state.name = None
    
    if "role" not in st.session_state:
        st.session_state.role = None
    
    if "player_id" not in st.session_state:
        st.session_state.player_id = None
    
    # Variable para redirecciones
    if "redirect_to" not in st.session_state:
        st.session_state.redirect_to = None
    
    # Otras variables
    if "current_page" not in st.session_state:
        st.session_state.current_page = None
    
    if "selected_metrics" not in st.session_state:
        st.session_state.selected_metrics = []

    if "page_history" not in st.session_state:
        st.session_state.page_history = []
        
    # Actividad de la sesión
    if "last_activity" not in st.session_state:
        st.session_state.last_activity = None