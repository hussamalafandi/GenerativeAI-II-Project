# context_memory.py
import json
import os

SESSION_FILE = "session_memory.json"

def load_memory():
    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_memory(memory):
    with open(SESSION_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)

def append_message(role, content):
    memory = load_memory()
    memory.append({"role": role, "content": content})
    save_memory(memory)

def get_context():
    memory = load_memory()
    return "\n".join([f"{m['role']}: {m['content']}" for m in memory[-10:]])  # letzte 10
