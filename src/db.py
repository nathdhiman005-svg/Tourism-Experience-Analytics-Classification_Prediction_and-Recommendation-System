import os
import psycopg2
import bcrypt
import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

@st.cache_resource
def get_db_connection():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url or db_url == "YOUR_DIRECT_CONNECTION_STRING_HERE":
        st.error("DATABASE_URL not found or not configured in .env file.")
        st.stop()
    try:
        conn = psycopg2.connect(db_url)
        # Ensure the table exists on first connection
        init_db(conn)
        return conn
    except Exception as e:
        st.error(f"Failed to connect to the database: {e}")
        st.stop()

def init_db(conn):
    """Creates the users table if it does not already exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(100),
                email VARCHAR(100) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                role VARCHAR(20) DEFAULT 'User'
            );
        """)
    conn.commit()

def create_user(conn, username, email, password, role="User"):
    """Hashes the password with bcrypt and inserts a new user."""
    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (username, email, password_hash, role) VALUES (%s, %s, %s, %s)",
                (username, email.lower().strip(), password_hash, role)
            )
        conn.commit()
        return True, "User created successfully."
    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        return False, "Email already exists."
    except Exception as e:
        conn.rollback()
        return False, f"An error occurred: {e}"

def authenticate_user(conn, email, password):
    """Checks the database for the email and validates the bcrypt hash."""
    with conn.cursor() as cur:
        cur.execute("SELECT password_hash, role FROM users WHERE email = %s", (email.lower().strip(),))
        result = cur.fetchone()
        
    if result:
        stored_hash, role = result
        if bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8')):
            return True, role
    return False, None
