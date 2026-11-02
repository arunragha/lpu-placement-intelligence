"""
LPU Placement Intelligence Platform - Lightweight Assessment History Store
Version: 2.1.0

Persists student assessments, transparent readiness scores, supervised placement predictions,
and recommendation snapshots in SQLite, supporting historical progress tracking and mentor reviews.
"""

import os
import sqlite3
import json
from datetime import datetime

class AssessmentHistoryStore:
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.environ.get("DATABASE_PATH")
        if db_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path = os.path.join(base_dir, "data", "assessment_history.db")
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS assessment_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        student_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        branch TEXT NOT NULL,
                        target_role TEXT,
                        cgpa REAL,
                        readiness_tier TEXT NOT NULL,
                        completeness_pct REAL,
                        input_payload TEXT,
                        prediction_result TEXT,
                        owner_username TEXT,
                        readiness_score REAL,
                        readiness_band TEXT,
                        readiness_category TEXT,
                        placement_pred INTEGER,
                        placement_prob REAL,
                        scoring_version TEXT,
                        model_version TEXT,
                        is_demo INTEGER DEFAULT 0
                    )
                """)
                # Check and add columns if upgrading from earlier schema
                cursor.execute("PRAGMA table_info(assessment_history)")
                existing_cols = [c[1] for c in cursor.fetchall()]
                new_cols = [
                    ("owner_username", "TEXT"),
                    ("readiness_score", "REAL"),
                    ("readiness_band", "TEXT"),
                    ("readiness_category", "TEXT"),
                    ("placement_pred", "INTEGER"),
                    ("placement_prob", "REAL"),
                    ("scoring_version", "TEXT"),
                    ("model_version", "TEXT"),
                    ("is_demo", "INTEGER DEFAULT 0")
                ]
                for col_def in new_cols:
                    col_name = col_def[0]
                    col_type = col_def[1]
                    if col_name not in existing_cols:
                        try:
                            cursor.execute(f"ALTER TABLE assessment_history ADD COLUMN {col_name} {col_type}")
                        except Exception:
                            pass

                cursor.execute("CREATE INDEX IF NOT EXISTS idx_student_id ON assessment_history(student_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_owner_user ON assessment_history(owner_username)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_is_demo ON assessment_history(is_demo)")
        finally:
            conn.close()

    def save_assessment(self, student_dict: dict, prediction_result: dict, owner_username: str = None, is_demo: int = 0) -> int:
        """Saves a student assessment record to SQLite with distinct readiness and prediction metrics."""
        student_id = str(student_dict.get("student_id") or "ANONYMOUS_STUDENT").strip()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        branch = str(student_dict.get("branch") or "Computer Science & Engineering")
        target_role = str(student_dict.get("target_role") or "Software Development Engineer (SDE)")
        cgpa = float(student_dict.get("cgpa") or 0.0)
        
        pred = prediction_result.get("prediction", {})
        readiness_tier = pred.get("readiness_tier", "Foundation")
        completeness = prediction_result.get("completeness", {})
        completeness_pct = float(completeness.get("completeness_pct", 100.0))
        
        # Transparent Readiness Breakdown
        readiness = prediction_result.get("readiness_assessment", {})
        readiness_score = float(readiness.get("total_score") or 0.0)
        readiness_band = str(readiness.get("readiness_band") or readiness_tier)
        
        # Standardized 3-tier canonical category
        if readiness_band in ["Needs attention", "Developing", "Strong preparation"]:
            readiness_category = readiness_band
        elif readiness_tier == "Foundation":
            readiness_category = "Needs attention"
        elif readiness_tier == "Developing":
            readiness_category = "Developing"
        else:
            readiness_category = "Strong preparation"
            
        scoring_version = str(readiness.get("scoring_version") or "2.2.0_3tier")
        
        # Supervised Placement Prediction
        placement_prob = float(pred.get("simulated_placement_prob") or pred.get("raw_placement_prob") or 0.0)
        placement_pred = 1 if placement_prob >= 50.0 else 0
        model_version = str(prediction_result.get("model_version") or "2.1.0")
        
        # Auto-flag demo records if student ID starts with demo prefix
        if student_id.startswith("LPU_DEMO_") or student_id.startswith("DEMO_"):
            is_demo = 1
            
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO assessment_history 
                    (student_id, timestamp, branch, target_role, cgpa, readiness_tier, completeness_pct, 
                     input_payload, prediction_result, owner_username, readiness_score, readiness_band, 
                     readiness_category, placement_pred, placement_prob, scoring_version, model_version, is_demo)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    student_id,
                    timestamp,
                    branch,
                    target_role,
                    cgpa,
                    readiness_tier,
                    completeness_pct,
                    json.dumps(student_dict),
                    json.dumps(prediction_result),
                    owner_username or student_id,
                    readiness_score,
                    readiness_band,
                    readiness_category,
                    placement_pred,
                    placement_prob,
                    scoring_version,
                    model_version,
                    is_demo
                ))
                return cursor.lastrowid
        finally:
            conn.close()

    def get_assessment_by_id(self, assessment_id: int) -> dict:
        """Retrieves a single assessment record by its unique database ID."""
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM assessment_history WHERE id = ?", (int(assessment_id),))
            row = cursor.fetchone()
            if not row:
                return None
            res = dict(row)
            # Ensure canonical readiness_category is populated
            if not res.get("readiness_category"):
                band = res.get("readiness_band")
                tier = res.get("readiness_tier")
                if band in ["Needs attention", "Developing", "Strong preparation"]:
                    res["readiness_category"] = band
                elif tier == "Foundation":
                    res["readiness_category"] = "Needs attention"
                elif tier == "Developing":
                    res["readiness_category"] = "Developing"
                else:
                    res["readiness_category"] = "Strong preparation"
                    
            if res.get("input_payload"):
                try:
                    res["input_payload"] = json.loads(res["input_payload"])
                except Exception:
                    pass
            if res.get("prediction_result"):
                try:
                    res["prediction_result"] = json.loads(res["prediction_result"])
                except Exception:
                    pass
            return res
        finally:
            conn.close()

    def get_student_history(self, student_id: str, owner_username: str = None) -> list:
        """Returns chronological assessment history for a specific student ID or username."""
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if owner_username:
                cursor.execute("""
                    SELECT id, student_id, timestamp, branch, target_role, cgpa, readiness_tier, 
                           completeness_pct, readiness_score, readiness_band, readiness_category, 
                           placement_prob, scoring_version, model_version, is_demo
                    FROM assessment_history
                    WHERE student_id = ? OR owner_username = ?
                    ORDER BY id ASC
                """, (student_id.strip(), owner_username.strip()))
            else:
                cursor.execute("""
                    SELECT id, student_id, timestamp, branch, target_role, cgpa, readiness_tier, 
                           completeness_pct, readiness_score, readiness_band, readiness_category, 
                           placement_prob, scoring_version, model_version, is_demo
                    FROM assessment_history
                    WHERE student_id = ?
                    ORDER BY id ASC
                """, (student_id.strip(),))
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                if not item.get("readiness_category"):
                    band = item.get("readiness_band")
                    tier = item.get("readiness_tier")
                    if band in ["Needs attention", "Developing", "Strong preparation"]:
                        item["readiness_category"] = band
                    elif tier == "Foundation":
                        item["readiness_category"] = "Needs attention"
                    elif tier == "Developing":
                        item["readiness_category"] = "Developing"
                    else:
                        item["readiness_category"] = "Strong preparation"
                results.append(item)
            return results
        finally:
            conn.close()

    def get_recent_assessments(self, limit: int = 50, offset: int = 0, is_guest: bool = False,
                               category_filter: str = None, search: str = None) -> dict:
        """
        Returns paginated assessment runs across the platform with optional filters.
        Enforces guest isolation: guests only see demo records.
        """
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            conditions = []
            params = []
            
            if is_guest:
                conditions.append("(is_demo = 1 OR student_id LIKE 'LPU_DEMO_%')")
                
            if category_filter and category_filter != "all":
                cf = category_filter.strip()
                if cf == "Needs attention":
                    conditions.append("(readiness_category = 'Needs attention' OR readiness_band = 'Needs attention' OR readiness_tier = 'Foundation')")
                elif cf == "Developing":
                    conditions.append("(readiness_category = 'Developing' OR readiness_band = 'Developing' OR readiness_tier = 'Developing')")
                elif cf == "Strong preparation":
                    conditions.append("(readiness_category = 'Strong preparation' OR readiness_band = 'Strong preparation' OR readiness_tier IN ('Placement Ready', 'Advanced'))")
                    
            if search and search.strip():
                kw = f"%{search.strip()}%"
                conditions.append("(student_id LIKE ? OR branch LIKE ? OR target_role LIKE ?)")
                params.extend([kw, kw, kw])
                
            where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
            
            # Count total matching
            count_sql = f"SELECT COUNT(*) FROM assessment_history {where_clause}"
            cursor.execute(count_sql, params)
            total_count = cursor.fetchone()[0]
            
            # Fetch page
            fetch_sql = f"""
                SELECT id, student_id, timestamp, branch, target_role, cgpa, readiness_tier, 
                       completeness_pct, readiness_score, readiness_band, readiness_category, 
                       placement_prob, scoring_version, model_version, is_demo
                FROM assessment_history
                {where_clause}
                ORDER BY id DESC
                LIMIT ? OFFSET ?
            """
            cursor.execute(fetch_sql, params + [int(limit), int(offset)])
            rows = cursor.fetchall()
            
            results = []
            for r in rows:
                item = dict(r)
                if not item.get("readiness_category"):
                    band = item.get("readiness_band")
                    tier = item.get("readiness_tier")
                    if band in ["Needs attention", "Developing", "Strong preparation"]:
                        item["readiness_category"] = band
                    elif tier == "Foundation":
                        item["readiness_category"] = "Needs attention"
                    elif tier == "Developing":
                        item["readiness_category"] = "Developing"
                    else:
                        item["readiness_category"] = "Strong preparation"
                results.append(item)
                
            return {
                "total_count": total_count,
                "assessments": results,
                "limit": limit,
                "offset": offset,
                "page": (offset // limit) + 1 if limit > 0 else 1
            }
        finally:
            conn.close()

    def delete_test_records(self, dry_run: bool = True) -> dict:
        """Identifies and optionally purges test run artifacts from the database."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            test_patterns = [
                "LPU_UNIT_TEST_%", "STU_VALID_%", "STU_INVALID_%",
                "STU_REPORT_TEST_%", "test_stu_%", "%unit_test%"
            ]
            where_clause = " OR ".join(["student_id LIKE ?"] * len(test_patterns))
            cursor.execute(f"SELECT id, student_id FROM assessment_history WHERE {where_clause}", test_patterns)
            rows = cursor.fetchall()
            matched_ids = [r[0] for r in rows]
            
            if not dry_run and matched_ids:
                with conn:
                    cursor.execute(f"DELETE FROM assessment_history WHERE {where_clause}", test_patterns)
                    
            return {
                "dry_run": dry_run,
                "matched_count": len(matched_ids),
                "matched_ids": matched_ids
            }
        finally:
            conn.close()
