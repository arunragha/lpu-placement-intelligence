"""
LPU Placement Intelligence Platform - Authentication & Authorization Service
Version: 1.0.0

Implements secure role-based access control (Student, Mentor, Admin),
cryptographic password hashing with PBKDF2-HMAC-SHA256, HMAC-signed session tokens,
and database-backed mentor-student assignment management.
"""

import os
import time
import base64
import hashlib
import secrets
import hmac
import json
import sqlite3

SECRET_KEY = os.getenv("AUTH_SECRET_KEY", "lpu-placement-secret-key-production-seed-2026")
TOKEN_TTL_SECONDS = 86400 * 7  # 7 days

class AuthService:
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.environ.get("DATABASE_PATH")
        if db_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path = os.path.join(base_dir, "data", "assessment_history.db")
        self.db_path = db_path
        self._init_tables()
        self._seed_default_users()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_tables(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        salt TEXT NOT NULL,
                        role TEXT NOT NULL,
                        full_name TEXT NOT NULL,
                        email TEXT,
                        student_reference_id TEXT,
                        created_at TEXT NOT NULL
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS mentor_assignments (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        mentor_username TEXT NOT NULL,
                        student_id TEXT NOT NULL,
                        assigned_at TEXT NOT NULL,
                        UNIQUE(mentor_username, student_id)
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS mentor_feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        assessment_id INTEGER NOT NULL,
                        student_id TEXT NOT NULL,
                        mentor_username TEXT NOT NULL,
                        feedback_notes TEXT NOT NULL,
                        recommended_focus TEXT,
                        created_at TEXT NOT NULL
                    )
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_mentor_assign ON mentor_assignments(mentor_username)")
        finally:
            conn.close()

    def _hash_password(self, password: str, salt: str = None) -> tuple:
        if not salt:
            salt = secrets.token_hex(16)
        pwd_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            100000
        ).hex()
        return pwd_hash, salt

    def _seed_default_users(self):
        """Seeds initial demo accounts for Student, Mentor, and Admin roles."""
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM users")
                if cursor.fetchone()[0] == 0:
                    now = time.strftime("%Y-%m-%d %H:%M:%S")
                    demo_users = [
                        ("student1", "Student@123", "student", "Aarav Sharma (B.Tech CSE)", "aarav.sharma@lpu.in", "LPU_DEMO_ADV01"),
                        ("student2", "Student@123", "student", "Priya Verma (B.Tech IT)", "priya.verma@lpu.in", "LPU_DEMO_RDY02"),
                        ("mentor1", "Mentor@123", "mentor", "Dr. Rajesh Sharma (Placement Chair)", "rajesh.mentor@lpu.in", None),
                        ("admin1", "Admin@123", "admin", "TPO Central Placement Administrator", "tpo.admin@lpu.in", None)
                    ]
                    for uname, pwd, role, fname, email, stuid in demo_users:
                        phash, salt = self._hash_password(pwd)
                        cursor.execute("""
                            INSERT INTO users (username, password_hash, salt, role, full_name, email, student_reference_id, created_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (uname, phash, salt, role, fname, email, stuid, now))

                    # Seed initial mentor assignments
                    cursor.execute("""
                        INSERT OR IGNORE INTO mentor_assignments (mentor_username, student_id, assigned_at)
                        VALUES (?, ?, ?)
                    """, ("mentor1", "LPU_DEMO_ADV01", now))
                    cursor.execute("""
                        INSERT OR IGNORE INTO mentor_assignments (mentor_username, student_id, assigned_at)
                        VALUES (?, ?, ?)
                    """, ("mentor1", "LPU_DEMO_RDY02", now))
                    cursor.execute("""
                        INSERT OR IGNORE INTO mentor_assignments (mentor_username, student_id, assigned_at)
                        VALUES (?, ?, ?)
                    """, ("mentor1", "LPU_DEMO_DEV03", now))
                    cursor.execute("""
                        INSERT OR IGNORE INTO mentor_assignments (mentor_username, student_id, assigned_at)
                        VALUES (?, ?, ?)
                    """, ("mentor1", "LPU_DEMO_FND04", now))
        finally:
            conn.close()

    def generate_token(self, user_dict: dict) -> str:
        """Generates an HMAC-signed token containing user id, username, role, and expiry."""
        payload = {
            "uid": user_dict["id"],
            "username": user_dict["username"],
            "role": user_dict["role"],
            "student_reference_id": user_dict.get("student_reference_id"),
            "exp": int(time.time()) + TOKEN_TTL_SECONDS
        }
        json_bytes = json.dumps(payload, separators=(',', ':')).encode("utf-8")
        b64_payload = base64.urlsafe_b64encode(json_bytes).decode("utf-8").rstrip("=")
        sig = hmac.new(SECRET_KEY.encode("utf-8"), b64_payload.encode("utf-8"), hashlib.sha256).hexdigest()
        return f"{b64_payload}.{sig}"

    def verify_token(self, token: str) -> dict:
        """Verifies HMAC signature and expiration of a session token."""
        if not token or "." not in token:
            return None
        parts = token.split(".")
        if len(parts) != 2:
            return None
        b64_payload, signature = parts
        expected_sig = hmac.new(SECRET_KEY.encode("utf-8"), b64_payload.encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected_sig):
            return None
        try:
            # Add padding back if necessary
            rem = len(b64_payload) % 4
            padded = b64_payload + ("=" * ((4 - rem) % 4))
            raw_json = base64.urlsafe_b64decode(padded.encode("utf-8")).decode("utf-8")
            payload = json.loads(raw_json)
            if payload.get("exp", 0) < time.time():
                return None  # Expired
            return payload
        except Exception:
            return None

    def register_user(self, username: str, password: str, role: str, full_name: str, email: str = "", student_reference_id: str = "") -> tuple:
        """Registers a new user account."""
        username = username.strip().lower()
        if len(username) < 3:
            return False, "Username must be at least 3 characters long.", None
        if len(password) < 6:
            return False, "Password must be at least 6 characters long.", None
        if role not in ["student", "mentor", "admin"]:
            role = "student"
            
        phash, salt = self._hash_password(password)
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO users (username, password_hash, salt, role, full_name, email, student_reference_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (username, phash, salt, role, full_name, email, student_reference_id or username.upper(), now))
                user_id = cursor.lastrowid
                user_data = {
                    "id": user_id,
                    "username": username,
                    "role": role,
                    "full_name": full_name,
                    "email": email,
                    "student_reference_id": student_reference_id or username.upper()
                }
                token = self.generate_token(user_data)
                return True, "User registered successfully.", {"token": token, "user": user_data}
        except sqlite3.IntegrityError:
            return False, f"Username '{username}' is already registered.", None
        finally:
            conn.close()

    def authenticate_user(self, username: str, password: str) -> tuple:
        """Authenticates user credentials and returns an access token."""
        username = username.strip().lower()
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            user = cursor.fetchone()
            if not user:
                return False, "Invalid username or password.", None
            
            pwd_hash, _ = self._hash_password(password, user["salt"])
            if not hmac.compare_digest(pwd_hash, user["password_hash"]):
                return False, "Invalid username or password.", None
                
            user_data = {
                "id": user["id"],
                "username": user["username"],
                "role": user["role"],
                "full_name": user["full_name"],
                "email": user["email"],
                "student_reference_id": user["student_reference_id"]
            }
            token = self.generate_token(user_data)
            return True, "Authentication successful.", {"token": token, "user": user_data}
        finally:
            conn.close()

    def get_mentor_assigned_students(self, mentor_username: str) -> list:
        """Returns the list of student IDs assigned to a specific mentor."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT student_id, assigned_at
                FROM mentor_assignments
                WHERE mentor_username = ?
            """, (mentor_username,))
            rows = cursor.fetchall()
            return [{"student_id": r[0], "assigned_at": r[1]} for r in rows]
        finally:
            conn.close()

    def is_student_assigned_to_mentor(self, mentor_username: str, student_id: str) -> bool:
        """Verifies whether a student is authorized to be viewed by a mentor."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 1 FROM mentor_assignments
                WHERE mentor_username = ? AND student_id = ?
            """, (mentor_username, student_id))
            return cursor.fetchone() is not None
        finally:
            conn.close()

    def add_mentor_feedback(self, mentor_username: str, assessment_id: int, student_id: str, feedback_notes: str, recommended_focus: str = "") -> int:
        """Saves mentor feedback on an assessment."""
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO mentor_feedback (assessment_id, student_id, mentor_username, feedback_notes, recommended_focus, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (assessment_id, student_id, mentor_username, feedback_notes, recommended_focus, now))
                return cursor.lastrowid
        finally:
            conn.close()

    def get_feedback_for_student(self, student_id: str) -> list:
        """Retrieves mentor feedback notes for a student."""
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, assessment_id, mentor_username, feedback_notes, recommended_focus, created_at
                FROM mentor_feedback
                WHERE student_id = ?
                ORDER BY id DESC
            """, (student_id,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

