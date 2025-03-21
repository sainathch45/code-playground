import psycopg2

def get_db_connection():
    conn = psycopg2.connect(
        dbname="smart_grid_db",
        user="postgres",
        password="yourpassword",
        host="localhost"
    )
    return conn
