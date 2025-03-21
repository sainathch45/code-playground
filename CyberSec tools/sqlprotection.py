import sqlite3
import re
from flask import Flask, request, render_template_string, redirect, url_for

app = Flask(__name__)

# 🚀 Dummy Database Setup (Creates Table & Inserts Admin User)
def init_db():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT, password TEXT)''')
    cursor.execute("INSERT OR IGNORE INTO users (id, username, password) VALUES (1, 'admin', 'password123')")
    conn.commit()
    conn.close()

init_db()  # Initialize Database

# 🛡️ Simple SQL Injection Firewall
def is_sql_injection_attempt(user_input):
    sql_keywords = ["'", '"', "--", ";", "/*", "*/", "xp_", "exec", "UNION", "SELECT", "INSERT", "DELETE", "DROP", "UPDATE"]
    pattern = r"|".join(re.escape(keyword) for keyword in sql_keywords)
    return re.search(pattern, user_input, re.IGNORECASE)

# 📌 HTML Template for Login Page
login_page = """
<!DOCTYPE html>
<html>
<head>
    <title>Secure Login</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }
        form { display: inline-block; background: #f8f8f8; padding: 20px; border-radius: 8px; }
        input { padding: 8px; margin: 5px; width: 200px; }
        button { padding: 10px 15px; background: green; color: white; border: none; cursor: pointer; }
        .error { color: red; font-weight: bold; }
    </style>
</head>
<body>
    <h2>Login to Your Account</h2>
    <form method="POST">
        <input type="text" name="username" placeholder="Username" required><br>
        <input type="password" name="password" placeholder="Password" required><br>
        <button type="submit">Login</button>
    </form>
    {% if error %}
        <p class="error">{{ error }}</p>
    {% endif %}
</body>
</html>
"""

# 🔐 Login Route
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        # 🛑 Block SQL Injection Attempts
        if is_sql_injection_attempt(username) or is_sql_injection_attempt(password):
            return render_template_string(login_page, error="⚠️ SQL Injection Detected! Access Denied.")

        # ✅ Check Credentials Securely (Preventing SQL Injection)
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            return "✅ Login Successful! Welcome, " + username
        else:
            return render_template_string(login_page, error="❌ Invalid Username or Password!")

    return render_template_string(login_page)

# 🚀 Run the App
if __name__ == "__main__":
    app.run(debug=True)
