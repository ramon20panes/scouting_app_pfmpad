# utils/db.py
import sqlite3
import hashlib
import os
from pathlib import Path
import sys

# Añadir el directorio raíz al path para importar config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_CONFIG

def get_db_connection():
    """Establece una conexión con la base de datos"""
    # Asegurarse que el directorio data existe
    data_dir = Path(os.path.dirname(DB_CONFIG["db_path"]))
    data_dir.mkdir(exist_ok=True)
    
    conn = sqlite3.connect(DB_CONFIG["db_path"])
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Inicializa la base de datos con tablas y usuarios de ejemplo"""
    conn = get_db_connection()
    
    # Crear tabla de usuarios
    conn.execute('''
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT NOT NULL,
        name TEXT NOT NULL,
        role TEXT NOT NULL,
        player_id INTEGER
    )
    ''')
    
    # Crear algunos usuarios de ejemplo
    users = [
        ('admin', hash_password('admin'), 'Administrador', 'admin', None),
        ('Lucas Bracamonte', hash_password('lucbra10'), 'Lucas', 'Director', None),
        ('Ramongonex', hash_password('ragoex20'), 'Ramón', 'Coordinador', None),
        ('j_musso', hash_password('j_musso1'), 'Juan Musso', 'player', 1),
        ('gimenez', hash_password('gimenez2'), 'José María Giménez', 'player', 2),
        ('oblak', hash_password('oblak13'), 'Jan Oblak', 'player', 13),
        ('griezmann', hash_password('griezmann7'), 'Antoine Griezmann', 'player', 7),
        ('de_paul', hash_password('de_paul5'), 'Rodrigo De Paul', 'player', 5),
        ('p_barrios', hash_password('p_barrios8'), 'Pablo Barrios', 'player', 8),
        ('azpilicueta', hash_password('azpilicueta3'), 'César Azpilicueta', 'player', 3),
        ('gallagher', hash_password('gallagher4'), 'Conor Gallagher', 'player', 4),
        ('koke', hash_password('koke6'), 'Jorge Resurrección', 'player', 6),
        ('sorloth', hash_password('sorloth9'), 'Aleksander Sorloth', 'player', 9),
        ('correa', hash_password('correa10'), 'Ángel Correa', 'player', 10),
        ('lemar', hash_password('lemar11'), 'Thomas Lemar', 'player', 11),
        ('lino', hash_password('s_lino12'), 'Samuel Lino', 'player', 12),
        ('llorente', hash_password('llorente14'), 'Marcos Llorente', 'player', 14),
        ('lenglet', hash_password('lenglet15'), 'Clement Lenglet', 'player', 15),
        ('molina', hash_password('n_molina16'), 'Nahuel Molina', 'player', 16),
        ('riquelme', hash_password('r_riquelme17'), 'Rodrigo Riquelme', 'player', 17),
        ('j_alvarez', hash_password('j_alvarez19'), 'Julián Álvarez', 'player', 19),
        ('witsel', hash_password('witsel_20'), 'Axel Witsel', 'player', 20),
        ('j_galan', hash_password('j_galan21'), 'Javi Galán', 'player', 21),
        ('g_simeone', hash_password('g_simeone22'), 'Giuliano Simeone', 'player', 22),
        ('reinildo', hash_password('reinildo23'), 'Reinildo Mandava', 'player', 23),
        ('le_normand', hash_password('le_normand24'), 'Robin Le Normand', 'player', 24),
        ('a_niño', hash_password('a_niño32'), 'Adrián Niño', 'player', 32),
        ('i_kostis', hash_password('i_kostis27'), 'Ilias Kostis', 'player', 27)
    ]
    
    # Verificar si ya existen usuarios para no duplicar
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM users")
    count = cursor.fetchone()[0]
    
    if count == 0:  # Solo insertar si no hay usuarios
        conn.executemany(
            "INSERT INTO users (username, password, name, role, player_id) VALUES (?, ?, ?, ?, ?)",
            users
        )
        conn.commit()
    
    conn.close()
    print("Base de datos inicializada con éxito")

def hash_password(password):
    """Genera un hash seguro para la contraseña"""
    return hashlib.sha256(password.encode()).hexdigest()

def check_user(username, password):
    """Verifica las credenciales del usuario"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    hashed_password = hash_password(password)
    cursor.execute(
        "SELECT * FROM users WHERE username = ? AND password = ?",
        (username, hashed_password)
    )
    
    user = cursor.fetchone()
    conn.close()
    
    return dict(user) if user else None

def get_user_by_username(username):
    """Obtiene los datos de un usuario por su username"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    
    conn.close()
    
    return dict(user) if user else None