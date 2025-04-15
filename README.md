# 🧠 Scouting App PFMPAD – Atlético de Madrid 24/25

Aplicación desarrollada en **Python + Streamlit** como parte del módulo del proyecto fin de Máster de Python aplicado al deporte (PFMPAD). Su propósito es ofrecer una plataforma visual e interactiva para **el análisis físico, técnico y táctico del primer equipo del Atlético de Madrid**, como ejemplo para ejecutar en cualquier institución deportiva o empresa.

---

## 📊 Funcionalidades

### 0. **Login por roles**
- Acceso o no según rol dentro de club

### 1. **Stats acumuladas**
- Datos por jugador de la presente temporada
- Ránkings y tabla

### 2. **Visualización Avanzada de la Temporada**
- Evolución de la clasificación en LaLiga
- Timeline de partidos oficiales
- Gráficos de Expected Goals (xG)

### 3. **Análisis por jornada**
- Estadísticas comparativas
- Match Momentum
- Red de pases
- xG
- Mapa de tiros

### 4. **Detección de modelo de juego**
- Índices valorativos de aspectos del juego
- Por localía, resultado o rival

### 5. **Ranking y Comparativas**
- Rankings personalizados por métrica y jornada
- Filtros por posición, nacionalidad o edad
- Visualización con gráficos de barras y circulares

### 6. **Análisis Individual Físico (ficticios)**
- Acceso personalizado por jugador
- Métricas como distancia total, sprints, HR, impactos, etc.
- Evolución por jornada
- Exportación en PDF de los informes

### 7. **Jugadores Similares**
- Búsqueda de perfiles comparables por posición
- Análisis mediante KMeans + KNN
- Mapeo inteligente para nombres con alias (Koke, Sørloth, etc.)
- Exportación de jugadores similares en PDF

---

## 🔐 Control de Acceso

Sistema de autenticación por roles:
- **Admins / Staff técnico**: acceso completo a todas las páginas
- **Jugadores**: acceso solo a sus propios datos físicos

---

## 📁 Estructura del Proyecto

📦 scouting_app_pfmpad 
├── pages/ │ ├── 1_Fisico_Individual.py │ ├── 2_Season_viz.py │ ├── ... 

├── utils/ │ ├── auth.py │ ├── datos_fisicos.py │ ├── export_pdf.py │ ├── ... 

├── models/ │ └── similar_players/ │ ├── predict.py │ ├── train_model.py 

├── assets/ │ ├── images/ │ ├── heatmaps/ │ ├── players/ 

├── data/ │ ├── raw/ │ ├── processed/ 

├── streamlit/ │ └── secrets.toml

![Estructura](image.png)


---

## ▶️ Cómo Ejecutar la App

# 1. Crear entorno virtual
python -m venv ent_mod11
ent_mod11\Scripts\activate # o en Mac ssource ent_mod11/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar Streamlit
streamlit run ATMapp.py


## 🌐 Despliegue

# ✅ Preparada para Streamlit Cloud

# 🐳 Lista para montar en Docker

# 📦 Organización modular para integraciones externas


### **📄 Autores y Créditos**

👨‍🏫 Ramón González

Máster PFMPAD – Módulo 11


### **Agradeciimientos**

[Lucas Bracamonte](https://www.linkedin.com/in/lucas-braca?originalSubdomain=ar)


[Sport Data Campus](https://www.linkedin.com/school/sports-data-campus/?trk=publ)


