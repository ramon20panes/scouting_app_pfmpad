import streamlit as st  

def create_header():
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("Título de tu página")
    with col2:
        st.image("assets/images/logos/atm.png", width=100)  # Ajusta el tamaño según necesites