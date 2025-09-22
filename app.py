import os
import sqlite3
import secrets
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_mail import Mail, Message

# -----------------------
# Config
# -----------------------
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "bec_lost_found.db"
UPLOAD_DIR = BASE_DIR / "uploads"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}
MAX_CONTENT_LENGTH = 8 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["JSON_SORT_KEYS"] = False

# --- Email Config ---
app.config.update(
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USERNAME='sohailhangaragids@gmail.com',
    MAIL_PASSWORD='mxxbiitpbfslsenp'
)
mail = Mail(app)

CORS(app, resources={r"/*": {"origins": "*"}})
UPLOAD_DIR.mkdir(exist_ok=True)


# -----------------------
# DB helpers
# -----------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as con:
        cur = con.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email_verified BOOLEAN DEFAULT FALSE,
                verification_token TEXT,
                created_at TEXT NOT NULL
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                type TEXT CHECK(type IN ('lost','found')) NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                location TEXT,
                date_str TEXT,
                contact_email TEXT,
                status TEXT DEFAULT 'open' CHECK(status IN ('open', 'pending', 'returned')),
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(item_id) REFERENCES items(id)
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS claims (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id INTEGER NOT NULL,
                claimer_user_id INTEGER NOT NULL,
                claimer_name TEXT NOT NULL,
                claimer_contact TEXT NOT NULL,
                proof TEXT,
                status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'approved', 'denied')),
                created_at TEXT NOT NULL,
                FOREIGN KEY(item_id) REFERENCES items(id),
                FOREIGN KEY(claimer_user_id) REFERENCES users(id)
            );
            """
        )
        con.commit()


def row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {k: row[k] for k in row.keys()} if row else None


# -----------------------
# Utilities
# -----------------------
def now_str() -> str:
    # Standard ISO format for the database
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def validate_password(password: str) -> tuple[bool, str]:
    """
    Validate password strength.
    Returns (is_valid, error_message)
    Requirements:
    - At least 8 characters
    - At least one uppercase letter
    - At least one special character
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"
    
    special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    if not any(c in special_chars for c in password):
        return False, "Password must contain at least one special character (!@#$%^&*()_+-=[]{}|;:,.<>?)"
    
    return True, ""

def generate_verification_token() -> str:
    """Generate a secure verification token"""
    return secrets.token_urlsafe(32)

def send_verification_email(email: str, name: str, token: str):
    """Send email verification email"""
    verification_url = f"http://127.0.0.1:5000/verify_email?token={token}"
    
    msg = Message(
        "Verify Your Email - BEC Lost & Found",
        sender=app.config['MAIL_USERNAME'],
        recipients=[email]
    )
    msg.html = f"""
    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
        <h2 style="color: #10b981;">Welcome to BEC Lost & Found!</h2>
        <p>Hi {name},</p>
        <p>Thank you for signing up! Please verify your email address to complete your registration and start using our platform.</p>
        <div style="text-align: center; margin: 30px 0;">
            <a href="{verification_url}" style="background-color: #10b981; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; display: inline-block;">Verify Email Address</a>
        </div>
        <p>If the button doesn't work, copy and paste this link into your browser:</p>
        <p style="word-break: break-all; color: #666;">{verification_url}</p>
        <p>This verification link will expire in 24 hours.</p>
        <p>Best regards,<br>BEC Lost & Found Team</p>
    </div>
    """
    try:
        mail.send(msg)
        return True
    except Exception as e:
        print(f"Failed to send verification email: {e}")
        return False

# NEW: Filename-safe timestamp function
def now_for_filename() -> str:
    # Returns a timestamp safe for filenames (replaces colons)
    return datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')

def item_with_images_dict(con, item_row: sqlite3.Row) -> Dict[str, Any]:
    item = row_to_dict(item_row)
    cur = con.execute("SELECT filename FROM images WHERE item_id = ?", (item["id"],))
    files = [r["filename"] for r in cur.fetchall()]
    item["images"] = [f"/uploads/{fn}" for fn in files]
    return item

# -----------------------
# Routes
# -----------------------
@app.route("/")
def health(): return jsonify({"status": "ok"})

@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json()
    if not data or not data.get('email') or not data.get('password') or not data.get('name'):
        return jsonify({"error": "Name, email and password are required"}), 400
    
    # Validate password strength
    is_valid, error_message = validate_password(data['password'])
    if not is_valid:
        return jsonify({"error": error_message}), 400
    
    password_hash = generate_password_hash(data['password'])
    verification_token = generate_verification_token()
    
    with get_db() as con:
        # Check for existing username and email before inserting
        existing_user = con.execute(
            "SELECT name, email FROM users WHERE name = ? OR email = ?", 
            (data.get('name', ''), data['email'].lower())
        ).fetchone()
        
        if existing_user:
            if existing_user['name'] == data.get('name', ''):
                return jsonify({"error": "Username already exists. Please choose a different name."}), 409
            elif existing_user['email'] == data['email'].lower():
                return jsonify({"error": "Email already exists"}), 409
        
        try:
            cur = con.execute(
                "INSERT INTO users (name, email, password_hash, email_verified, verification_token, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (data.get('name', ''), data['email'].lower(), password_hash, False, verification_token, now_str())
            )
            con.commit()
            
            # Send verification email
            email_sent = send_verification_email(data['email'], data.get('name', ''), verification_token)
            
            return jsonify({
                "user_id": cur.lastrowid,
                "message": "Account created! Please check your email to verify your account.",
                "email_sent": email_sent
            }), 201
        except sqlite3.IntegrityError:
            return jsonify({"error": "User already exists"}), 409

@app.route('/verify_email', methods=['GET'])
def verify_email():
    token = request.args.get('token')
    if not token:
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Email Verification - BEC Lost & Found</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; padding: 50px; background: #f0f0f0; }
                .container { background: white; padding: 30px; border-radius: 10px; max-width: 500px; margin: 0 auto; }
                .error { color: #e74c3c; }
                .success { color: #27ae60; }
                .btn { background: #10b981; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block; margin: 10px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h2 class="error">Verification Failed</h2>
                <p>No verification token provided.</p>
                <a href="http://127.0.0.1:5000" class="btn">Go to App</a>
            </div>
        </body>
        </html>
        ''', 400
    
    with get_db() as con:
        user_row = con.execute("SELECT * FROM users WHERE verification_token = ?", (token,)).fetchone()
        if not user_row:
            return '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Email Verification - BEC Lost & Found</title>
                <style>
                    body { font-family: Arial, sans-serif; text-align: center; padding: 50px; background: #f0f0f0; }
                    .container { background: white; padding: 30px; border-radius: 10px; max-width: 500px; margin: 0 auto; }
                    .error { color: #e74c3c; }
                    .success { color: #27ae60; }
                    .btn { background: #10b981; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block; margin: 10px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h2 class="error">Verification Failed</h2>
                    <p>Invalid or expired verification token.</p>
                    <a href="http://127.0.0.1:5000" class="btn">Go to App</a>
                </div>
            </body>
            </html>
            ''', 400
        
        if user_row['email_verified']:
            return '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Email Verification - BEC Lost & Found</title>
                <style>
                    body { font-family: Arial, sans-serif; text-align: center; padding: 50px; background: #f0f0f0; }
                    .container { background: white; padding: 30px; border-radius: 10px; max-width: 500px; margin: 0 auto; }
                    .error { color: #e74c3c; }
                    .success { color: #27ae60; }
                    .btn { background: #10b981; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block; margin: 10px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h2 class="success">Already Verified</h2>
                    <p>Your email is already verified!</p>
                    <a href="http://127.0.0.1:5000" class="btn">Go to App</a>
                </div>
            </body>
            </html>
            ''', 200
        
        # Mark email as verified and clear token
        con.execute(
            "UPDATE users SET email_verified = TRUE, verification_token = NULL WHERE id = ?",
            (user_row['id'],)
        )
        con.commit()
        
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Email Verification - BEC Lost & Found</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; padding: 50px; background: #f0f0f0; }
                .container { background: white; padding: 30px; border-radius: 10px; max-width: 500px; margin: 0 auto; }
                .error { color: #e74c3c; }
                .success { color: #27ae60; }
                .btn { background: #10b981; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block; margin: 10px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h2 class="success">Email Verified Successfully!</h2>
                <p>Your email has been verified. You can now log in to your account.</p>
                <a href="http://127.0.0.1:5000" class="btn">Go to App</a>
            </div>
        </body>
        </html>
        ''', 200

@app.route('/resend_verification', methods=['POST'])
def resend_verification():
    data = request.get_json()
    if not data or not data.get('email'):
        return jsonify({'error': 'Email is required'}), 400
    
    with get_db() as con:
        user_row = con.execute("SELECT * FROM users WHERE email = ?", (data['email'].lower(),)).fetchone()
        if not user_row:
            return jsonify({'error': 'User not found'}), 404
        
        if user_row['email_verified']:
            return jsonify({'message': 'Email already verified'}), 200
        
        # Generate new token
        new_token = generate_verification_token()
        con.execute(
            "UPDATE users SET verification_token = ? WHERE id = ?",
            (new_token, user_row['id'])
        )
        con.commit()
        
        # Send verification email
        email_sent = send_verification_email(user_row['email'], user_row['name'], new_token)
        
        return jsonify({
            'message': 'Verification email sent! Please check your inbox.',
            'email_sent': email_sent
        }), 200

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data or not data.get('email') or not data.get('password'):
        return jsonify({'error': 'Email and password are required'}), 400

    with get_db() as con:
        user_row = con.execute("SELECT * FROM users WHERE email = ?", (data['email'].lower(),)).fetchone()
        if not user_row or not check_password_hash(user_row["password_hash"], data['password']):
            return jsonify({'error': 'Invalid email or password'}), 401
        
        # Check if email is verified
        if not user_row['email_verified']:
            return jsonify({'error': 'Please verify your email before logging in. Check your inbox for the verification link.'}), 403
        
        user_dict = row_to_dict(user_row)
        del user_dict['password_hash']
        del user_dict['verification_token']
        return jsonify({'message': 'Login successful', 'user': user_dict})

@app.route("/add_item", methods=["POST"])
def add_item():
    form = request.form
    if not form.get("user_id") or not form.get("type"):
        return jsonify({"error": "user_id and type are required"}), 400
    
    with get_db() as con:
        cur = con.execute(
            "INSERT INTO items (user_id, type, title, description, location, date_str, contact_email, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (form["user_id"], form["type"], form.get("title", "Untitled"), form.get("description", ""), form.get("location", ""), form.get("date"), form.get("contact_email"), now_str())
        )
        item_id = cur.lastrowid
        for key, file in request.files.items():
            if file and file.filename:
                # UPDATED: Use the new filename-safe timestamp function
                filename = f"{item_id}_{now_for_filename()}_{secure_filename(file.filename)}"
                file.save(UPLOAD_DIR / filename)
                con.execute("INSERT INTO images (item_id, filename, created_at) VALUES (?, ?, ?)", (item_id, filename, now_str()))
        con.commit()
        item_row = con.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
        return jsonify(item_with_images_dict(con, item_row)), 201

@app.route("/search", methods=["GET"])
def search_items():
    q = request.args.get("q", "").strip()
    with get_db() as con:
        sql = "SELECT * FROM items WHERE status IN ('open', 'pending')"
        params = []
        if q:
            like = f"%{q}%"
            sql += " AND (title LIKE ? OR description LIKE ?)"
            params.extend([like, like])
        sql += " ORDER BY created_at DESC LIMIT 100"
        rows = con.execute(sql, tuple(params)).fetchall()
        items = [item_with_images_dict(con, r) for r in rows]
        return jsonify({"count": len(items), "items": items})

@app.route("/claim_item", methods=["POST"])
def claim_item():
    data = request.get_json()
    if not data or not data.get('item_id') or not data.get('claimer_user_id'):
        return jsonify({"error": "item_id and claimer_user_id are required"}), 400

    with get_db() as con:
        item = con.execute("SELECT * FROM items WHERE id = ? AND status = 'open'", (data['item_id'],)).fetchone()
        if not item:
            return jsonify({"error": "Item not found or already has a pending claim"}), 404

        con.execute("UPDATE items SET status = 'pending' WHERE id = ?", (data['item_id'],))
        con.execute(
            "INSERT INTO claims (item_id, claimer_user_id, claimer_name, claimer_contact, proof, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (data['item_id'], data['claimer_user_id'], data.get('claimer_name', ''), data.get('claimer_contact', ''), data.get('proof', ''), now_str())
        )
        con.commit()

        owner = con.execute("SELECT u.email, i.title FROM users u JOIN items i ON u.id = i.user_id WHERE i.id = ?", (data['item_id'],)).fetchone()
        if owner:
            msg = Message(
                f"Action Required: A claim has been made on your item '{owner['title']}'",
                sender=app.config['MAIL_USERNAME'],
                recipients=[owner['email']]
            )
            msg.body = f"A claim has been submitted for your item '{owner['title']}'. Please log in to your dashboard to review and approve or deny the claim."
            try:
                mail.send(msg)
            except Exception as e:
                print(f"Email failed to send: {e}")

    return jsonify({"message": "Claim submitted for review"}), 201

@app.route("/my_items", methods=["GET"])
def my_items():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    with get_db() as con:
        item_rows = con.execute("SELECT * FROM items WHERE user_id = ? ORDER BY created_at DESC", (user_id,)).fetchall()
        response_items = []
        for item_row in item_rows:
            item_dict = item_with_images_dict(con, item_row)
            claim_rows = con.execute(
                "SELECT * FROM claims WHERE item_id = ? AND status = 'pending'", (item_dict['id'],)
            ).fetchall()
            item_dict['claims'] = [row_to_dict(r) for r in claim_rows]
            response_items.append(item_dict)
    
    return jsonify({"items": response_items})

@app.route("/resolve_claim", methods=["POST"])
def resolve_claim():
    data = request.get_json()
    if not data or not data.get('claim_id') or not data.get('user_id') or not data.get('resolution'):
        return jsonify({"error": "claim_id, user_id, and resolution ('approved' or 'denied') are required"}), 400

    resolution = data['resolution']
    if resolution not in ['approved', 'denied']:
        return jsonify({"error": "Invalid resolution"}), 400

    with get_db() as con:
        claim = con.execute("SELECT * FROM claims WHERE id = ?", (data['claim_id'],)).fetchone()
        if not claim:
            return jsonify({"error": "Claim not found"}), 404
        
        item = con.execute("SELECT * FROM items WHERE id = ?", (claim['item_id'],)).fetchone()
        if item['user_id'] != int(data['user_id']):
            return jsonify({"error": "Unauthorized"}), 403

        if resolution == 'approved':
            con.execute("UPDATE claims SET status = 'approved' WHERE id = ?", (data['claim_id'],))
            con.execute("UPDATE items SET status = 'returned' WHERE id = ?", (claim['item_id'],))
            con.execute("UPDATE claims SET status = 'denied' WHERE item_id = ? AND status = 'pending'", (claim['item_id'],))
        else:
            con.execute("UPDATE claims SET status = 'denied' WHERE id = ?", (data['claim_id'],))
            other_claims = con.execute("SELECT COUNT(*) as count FROM claims WHERE item_id = ? AND status = 'pending'", (claim['item_id'],)).fetchone()
            if other_claims['count'] == 0:
                con.execute("UPDATE items SET status = 'open' WHERE id = ?", (claim['item_id'],))

        con.commit()
        
    return jsonify({"message": f"Claim {resolution}"})

@app.route("/uploads/<path:filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_DIR, filename)

if __name__ == "__main__":
    init_db()
    app.run(host="127.0.0.1", port=5000, debug=True)