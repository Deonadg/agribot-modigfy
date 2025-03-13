import sqlite3
from datetime import datetime

def init_db():
    conn = sqlite3.connect('agribot.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS messages
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_message TEXT NOT NULL,
                  bot_response TEXT NOT NULL,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

def save_message(user_message, bot_response):
    conn = sqlite3.connect('agribot.db')
    c = conn.cursor()
    c.execute('INSERT INTO messages (user_message, bot_response) VALUES (?, ?)',
              (user_message, bot_response))
    conn.commit()
    conn.close()

def get_history():
    conn = sqlite3.connect('agribot.db')
    c = conn.cursor()
    c.execute('SELECT user_message, bot_response, timestamp FROM messages ORDER BY timestamp DESC LIMIT 50')
    messages = c.fetchall()
    conn.close()
    return [{'user_message': msg[0], 
             'bot_response': msg[1], 
             'timestamp': msg[2]} for msg in messages]

def save_unknown_query(user_message):
    conn = sqlite3.connect('agribot.db')
    c = conn.cursor()
    c.execute('INSERT INTO unknown_queries (user_message) VALUES (?)', (user_message,))
    conn.commit()
    conn.close()
