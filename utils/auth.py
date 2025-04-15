# utils/auth.py
import streamlit as st
import time
from datetime import datetime
import logging
from utils.db import check_user, get_user_by_username

# Configuración de logging
logging.basicConfig(
    filename='streamlit_auth.log', 
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constantes
SESSION_TIMEOUT = 1800  # 30 minutos en segundos

def check_session_timeout():
    """Verifica si la sesión ha expirado"""
    if 'last_activity' in st.session_state and st.session_state.last_activity is not None:
        last_activity = st.session_state.last_activity
        if (datetime.now() - last_activity).total_seconds() > SESSION_TIMEOUT:
            # Limpiar estado de sesión
            st.session_state.authentication_status = False
            st.session_state.last_activity = None
            return False
        return True
    return False        

def update_last_activity():
    """Actualiza el timestamp de última actividad"""
    st.session_state.last_activity = datetime.now()

def login():
    """Maneja la autenticación del usuario"""
        
    # Creamos un formulario que se puede enviar con Enter
    with st.form(key="login_form"):
        username = st.text_input("Usuario", key="username_input", placeholder="Ingresa tu usuario")
        password = st.text_input("Contraseña", type="password", key="password_input", placeholder="Ingresa tu contraseña")
        
        # Centrar el botón
        cols = st.columns([1, 2, 1])
        with cols[1]:
            # El botón de submit del formulario (se activa con Enter o clic)
            submit_button = st.form_submit_button("Iniciar Sesión")
        
        # Procesamiento del formulario (se ejecuta cuando se presiona Enter o se hace clic en el botón)
        if submit_button:
            user = check_user(username, password)
            
            if user:
                st.session_state.authentication_status = True
                st.session_state.username = username
                st.session_state.name = user['name']
                st.session_state.role = user['role']
                st.session_state.player_id = user['player_id']
                st.session_state.last_activity = datetime.now()

                # Añadir parámetro a la URL para ayudar a mantener la sesión
                st.query_params["authenticated"] = "true"
                
                # Mensaje de éxito
                st.success(f"¡Bienvenido, {user['name']}!", icon="✅")
                time.sleep(0.8)
                
                # Redirección a la primera página
                st.session_state.redirect_to = "pages/1_Team_stats.py"
                st.rerun()
            else:
                logging.warning(f"Intento de login fallido para usuario: {username}")
                st.error('Usuario o contraseña incorrectos', icon="🚨")

def logout():
    """Cierra la sesión del usuario"""
    for key in ['authentication_status', 'username', 'name', 'role', 'player_id', 'redirect_to', 'last_activity']:
        if key in st.session_state:
            del st.session_state[key]
    
    st.switch_page("ATMapp")

def check_auth():
    """Verifica autenticación y timeout"""
    if "authentication_status" in st.session_state and st.session_state.authentication_status:
        
        if check_session_timeout():
            update_last_activity()
            return True
        
        else:
            st.write("Autenticación fallida o expiración de sesión")  # Mensaje de depuración
    return False

def get_user_role():
    """Obtiene el rol del usuario autenticado"""
    if "role" in st.session_state:
        return st.session_state.role
    return None

def get_user_name():
    """Obtiene el nombre del usuario autenticado"""
    if "name" in st.session_state:
        return st.session_state.name
    return None

def get_player_id():
    """Obtiene el ID del jugador si el usuario es un jugador"""
    if "player_id" in st.session_state:
        return st.session_state.player_id
    return None

def check_admin_access():
    """Verifica si el usuario tiene acceso de administrador"""
    role = get_user_role()
    return role in ["admin", "Profesor", "Alumno"]

def check_player_access():
    """Verifica si el usuario es un jugador"""
    role = get_user_role()
    return role == "player"