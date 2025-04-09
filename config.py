# config.py
# config.py
import os
from pathlib import Path

# Rutas importantes
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = DATA_DIR / "auth"  # Subdirectorio para archivos de autenticación

# Asegurar que los directorios existan
DATA_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(exist_ok=True)

# Configuración de la base de datos
DB_CONFIG = {
    "db_path": str(DB_DIR / "scouters_auth.db")
}