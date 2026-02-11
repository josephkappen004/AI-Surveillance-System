import sqlite3
import numpy as np
import uuid
import os

DB_PATH = "faces.db"

def get_conn():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            id TEXT PRIMARY KEY,
            name TEXT,
            embedding BLOB
        )
    """)
    conn.commit()
    conn.close()

def add_face(name, embedding):
    face_id = str(uuid.uuid4())
    emb_bytes = embedding.astype(np.float32).tobytes()

    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "INSERT INTO faces VALUES (?, ?, ?)",
        (face_id, name, emb_bytes)
    )
    conn.commit()
    conn.close()

def delete_face(face_id):
    conn = get_conn()
    c = conn.cursor()
    c.execute("DELETE FROM faces WHERE id = ?", (face_id,))
    conn.commit()
    conn.close()

def get_faces():
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT id, name FROM faces")
    rows = c.fetchall()
    conn.close()

    return [{"id": r[0], "name": r[1]} for r in rows]

def load_embeddings():
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT name, embedding FROM faces")
    rows = c.fetchall()
    conn.close()

    data = []
    for name, emb in rows:
        emb_array = np.frombuffer(emb, dtype=np.float32)
        data.append({"name": name, "embedding": emb_array})
    return data

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
