import mariadb
import mysql.connector
import os
import dotenv
import uuid

# Load environment variables from prod.env
if os.environ.get("ENVIRONMENT") != "prod" :
    dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), 'prod.env'))

conn_params = {
    "host": os.environ.get("HOST"),
    "port": os.environ.get("PORT"),
    "user": os.environ.get("DB_USER", "root"),
    "password": os.environ.get("PASSWORD"),
    "database": os.environ.get("DATABASE", "cineai"),
}

def mariadb_get_connection():
    try:
        conn = mariadb.connect(**conn_params)
        return conn
    except mariadb.Error as e:
        print(f"Error connecting to MariaDB Platform: {e}")
        return None

def mysql_get_connection():
    try:
        conn = mysql.connector.connect(**conn_params)
        return conn
    except mysql.connector.Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def get_connection():
    if os.environ.get("DB_TYPE") == "mariadb":
        return mariadb_get_connection()
    else:
        return mysql_get_connection()

def init_tables() :
    conn = get_connection()
    if conn is None:
        return

    cursor = conn.cursor()

    # models : id:int, name:str, weights:bytes, created_at:datetime
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS models (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255) NOT NULL UNIQUE,
            weights LONGBLOB NOT NULL,
            model_type VARCHAR(50) NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # files : id:int, uuid:str (unique), contnt: LONGBLOB, created_at:datetime
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS files (
            id INT AUTO_INCREMENT PRIMARY KEY,
            uuid VARCHAR(36) NOT NULL UNIQUE,
            content LONGBLOB NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    cursor.close()
    conn.close()

def upload_file(content: bytes) -> uuid.uuid4 :
    conn = get_connection()
    if conn is None:
        return None

    cursor = conn.cursor()
    file_uuid = str(uuid.uuid4())
    cursor.execute("INSERT INTO files (uuid, content) VALUES (%s, %s)", (file_uuid, content))
    conn.commit()
    cursor.close()
    conn.close()

    return file_uuid

def retrieve_file(uuid: str) -> bytes:
    conn = get_connection()
    if conn is None:
        return None

    cursor = conn.cursor()
    cursor.execute("SELECT content FROM files WHERE uuid = %s", (uuid,))
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if row is None:
        return None

    return row[0]  # Return the content as bytes


def list_models():
    conn = get_connection()
    if conn is None:
        return []

    cursor = conn.cursor()
    cursor.execute("SELECT id, name, created_at FROM models")
    models = cursor.fetchall()
    cursor.close()
    conn.close()

    return models

def upload_model(name, weights, model_type):
    conn = get_connection()
    if conn is None:
        return None

    cursor = conn.cursor()
    cursor.execute("INSERT INTO models (name, weights, model_type) VALUES (%s, %s, %s)", (name, weights, model_type))
    conn.commit()
    model_id = cursor.lastrowid
    cursor.close()
    conn.close()

    return model_id

def get_weights_by_id(model_id):
    conn = get_connection()
    if conn is None:
        return None

    cursor = conn.cursor()
    cursor.execute("SELECT weights FROM models WHERE id = %s", (model_id,))
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if row is None:
        return None

    return row[0]  # Return the weights as bytes

def get_weughts_by_model_name(model_name):
    conn = get_connection()
    if conn is None:
        return None

    cursor = conn.cursor()
    cursor.execute("SELECT weights,model_type FROM models WHERE name = %s", (model_name,))
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if row is None:
        return None

    return row[0], row[1]  # Return the weights as bytes and model type

def reset_database() :
    conn = get_connection()
    if conn is None:
        return

    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS models")
    cursor.execute("DROP TABLE IF EXISTS files")
    conn.commit()
    cursor.close()
    conn.close()

if __name__ == "__main__":
    import sys
    if "force" in sys.argv:
        print("Resetting database...")
        reset_database()
    print("Initializing tables...")
    init_tables()
