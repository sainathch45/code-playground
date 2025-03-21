from flask import Flask, request, render_template_string
import sqlite3
import re

app = Flask(__name__)

# In-memory SQLite database for testing
def init_db():
    conn = sqlite3.connect("test.db")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT, password TEXT)")
    cursor.execute("INSERT INTO users (username, password) VALUES ('admin', 'password123')")
    conn.commit()
    conn.close()

init_db()

# HTML page for testing firewall
html_page = """
<!DOCTYPE html>
<html>
<head>
    <title>Firewall Test Page</title>
</head>
<body>
    <h2>Login Page</h2>
    <form action="/login" method="POST">
        <label for="username">Username:</label>
        <input type="text" id="username" name="username" required>
        <br><br>
        <label for="password">Password:</label>
        <input type="password" id="password" name="password" required>
        <br><br>
        <button type="submit">Login</button>
    </form>
</body>
</html>
"""

# Simple firewall function to detect SQL injection
def detect_sql_injection(input_string):
    sql_patterns = [
        "(--|#|\\/\\*)",  # Comments
        "(\' OR |\" OR )",  # OR-based injections
        "(UNION SELECT|SELECT \* FROM)",  # UNION attacks
        "(DROP TABLE|DELETE FROM|INSERT INTO|UPDATE .* SET)",  # Dangerous SQL commands
    ]
    for pattern in sql_patterns:
        if re.search(pattern, input_string, re.IGNORECASE):
            return True
    return False

@app.route('/')
def index():
    return render_template_string(html_page)

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username', '')
    password = request.form.get('password', '')
    
    # Check for SQL injection attempt
    if detect_sql_injection(username) or detect_sql_injection(password):
        return "SQL Injection Detected! Firewall Blocked the Request."
    
    conn = sqlite3.connect("test.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return "Login Successful!"
    else:
        return "Invalid Credentials."

if __name__ == '__main__':
    app.run(debug=True)