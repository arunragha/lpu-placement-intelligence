"""
LPU Placement Intelligence Platform - Comprehensive Test Suite v2.2
Verifies:
1. Supervised ML predictions across canonical 3-tier readiness categories.
2. Removal of gender from all models and schemas (algorithmic fairness).
3. Data-leakage free metrics and cross-validation metadata.
4. Input validation and error reporting.
5. SQLite assessment history persistence with ISOLATED test database.
6. Skill-gap diagnosis tolerance logic (+/- 3.0 pts) and negative delta handling.
7. Strictly descending feature importance ordering and active backlogs explanation.
8. Guest isolation for demo vs authenticated student records.
"""

import os
import json
import unittest
import numpy as np
import pandas as pd

# Direct database path to test database before app imports
TEST_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "test_assessment_history.db")
os.environ["DATABASE_PATH"] = TEST_DB_PATH

from app import app, get_history_store
from src.ml_engine import PlacementIntelligenceEngine, PRESET_STUDENTS
from src.history_store import AssessmentHistoryStore
from src.readiness_scorer import compute_readiness_score, map_legacy_tier_to_category, categorize_readiness_score
from src.auth_service import AuthService

class TestPlacementIntelligencePipelineV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure test database path is set and clean
        if os.path.exists(TEST_DB_PATH):
            try:
                os.remove(TEST_DB_PATH)
            except Exception:
                pass
        
        hs = get_history_store()
        hs.db_path = TEST_DB_PATH
        hs._init_db()
        
        cls.client = app.test_client()
        cls.engine = PlacementIntelligenceEngine()
        cls.history_store = hs
        cls.auth = AuthService()

    @classmethod
    def tearDownClass(cls):
        # Clean up test database file
        if os.path.exists(TEST_DB_PATH):
            try:
                os.remove(TEST_DB_PATH)
            except Exception:
                pass

    def test_01_dataset_integrity_and_fairness(self):
        """Verifies synthetic dataset has 2,000 records, 4 readiness categories, and excludes gender."""
        data_path = os.path.join(os.path.dirname(__file__), "data", "lpu_placement_dataset.csv")
        self.assertTrue(os.path.exists(data_path), "Dataset file not found")
        df = pd.read_csv(data_path)
        self.assertEqual(len(df), 2000, "Dataset should contain exactly 2,000 synthetic records")
        
        # Verify 4 target categories in raw dataset
        unique_tiers = set(df["readiness_tier"].unique())
        expected_tiers = {"Foundation", "Developing", "Placement Ready", "Advanced"}
        self.assertEqual(unique_tiers, expected_tiers, f"Expected tiers {expected_tiers}, found {unique_tiers}")
        
        # Verify algorithmic fairness: gender completely excluded
        self.assertNotIn("gender", df.columns, "Fairness violation: 'gender' must be excluded from dataset")
        self.assertNotIn("student_id", self.engine.numeric_features, "Student ID must not be a model feature")

    def test_02_model_artifacts_and_v2_metadata(self):
        """Verifies serialization of model artifacts and leakage-free metadata."""
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        expected_files = [
            "preprocessor.pkl",
            "readiness_classifier.pkl",
            "ctc_regressor.pkl",
            "prob_regressor.pkl",
            "role_classifier.pkl",
            "model_metrics.json"
        ]
        for f in expected_files:
            p = os.path.join(models_dir, f)
            self.assertTrue(os.path.exists(p), f"Missing model artifact: {f}")
            
        with open(os.path.join(models_dir, "model_metrics.json"), "r") as f:
            meta = json.load(f)
            
        self.assertIn("classification_models_benchmark", meta)
        self.assertEqual(meta["tier_classes"], ["Foundation", "Developing", "Placement Ready", "Advanced"])
        self.assertNotIn("gender", meta["numeric_features"])
        self.assertNotIn("gender", meta["categorical_features"])
        self.assertIn("Random Forest Classifier", meta["classification_models_benchmark"])
        
        rf_bench = meta["classification_models_benchmark"]["Random Forest Classifier"]
        self.assertGreater(rf_bench["accuracy"], 0.75, "Random Forest test accuracy below threshold")
        self.assertIn("cv_accuracy_mean", rf_bench)

    def test_03_prediction_engine_and_explainability(self):
        """Tests multi-class prediction, Saabas tree attribution, eligibility, and canonical categories."""
        for preset in PRESET_STUDENTS:
            student = preset["data"]
            result = self.engine.predict_student(student)
            
            self.assertIn("prediction", result)
            self.assertIn("completeness", result)
            self.assertIn("eligibility", result)
            self.assertIn("explainability", result)
            self.assertIn("radar_scores", result)
            self.assertIn("gap_analysis", result)
            self.assertIn("roadmap", result)
            
            pred = result["prediction"]
            # Check canonical 3-tier category is present
            self.assertIn("readiness_category", pred)
            self.assertIn(pred["readiness_category"], ["Needs attention", "Developing", "Strong preparation"])
            
            # Check demo model estimate integer
            self.assertIn("demo_model_estimate_pct", pred)
            self.assertIsInstance(pred["demo_model_estimate_pct"], int)
            self.assertIn("demo_model_estimate_notice", pred)
            
            # Check Explainability (Saabas Tree Path Decomposition)
            explain = result["explainability"]
            self.assertIn("primary_drivers", explain)
            self.assertIn("secondary_mitigators", explain)
            self.assertIn("driver_label", explain)
            
            # Check 7-Pillars in radar
            radar = result["radar_scores"]
            self.assertEqual(len(radar["labels"]), 7)
            self.assertEqual(len(radar["student_scores"]), 7)
            
            # Check 3-phase roadmap
            roadmap = result["roadmap"]
            self.assertEqual(len(roadmap), 3)

    def test_04_input_validation_server_side(self):
        """Verifies server rejection of invalid scores, out-of-bounds metrics, and inconsistent inputs."""
        # Test 1: Invalid CGPA (> 10.0)
        invalid_cgpa = dict(PRESET_STUDENTS[0]["data"])
        invalid_cgpa["cgpa"] = 12.5
        res = self.client.post("/api/predict", json=invalid_cgpa)
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertIn("errors", data)
        self.assertTrue(any("CGPA must be between 0.0 and 10.0" in e for e in data["errors"]))

        # Test 2: Inconsistent hackathon counts (won > participated)
        invalid_hack = dict(PRESET_STUDENTS[0]["data"])
        hack_test = {
            "hackathons_participated": 1,
            "hackathons_won": 4
        }
        invalid_hack.update(hack_test)
        res = self.client.post("/api/predict", json=invalid_hack)
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertTrue(any("Inconsistent hackathon counts" in e for e in data["errors"]))

        # Test 3: Inconsistent internship duration (count = 0 but duration > 0)
        invalid_intern = dict(PRESET_STUDENTS[0]["data"])
        invalid_intern["internships_count"] = 0
        invalid_intern["internship_duration_months"] = 6
        res = self.client.post("/api/predict", json=invalid_intern)
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertTrue(any("Inconsistent internship records" in e for e in data["errors"]))

    def test_05_assessment_history_isolated_sqlite_persistence(self):
        """Verifies assessment persistence to isolated SQLite test database and retrieval."""
        test_student_id = "LPU_DEMO_ADV01"  # Uses demo ID so guest queries can find it
        student_data = dict(PRESET_STUDENTS[0]["data"])
        student_data["student_id"] = test_student_id
        
        # Submit prediction
        res = self.client.post("/api/predict", json=student_data)
        self.assertEqual(res.status_code, 200)
        
        # Query history
        res_hist = self.client.get(f"/api/history/{test_student_id}")
        self.assertEqual(res_hist.status_code, 200)
        hist_data = res_hist.get_json()
        self.assertEqual(hist_data["status"], "success")
        self.assertGreaterEqual(hist_data["count"], 1)
        self.assertEqual(hist_data["history"][0]["student_id"], test_student_id)

    def test_06_batch_csv_upload_and_error_reporting(self):
        """Verifies TPO batch processing and reporting of invalid rows."""
        csv_content = """student_id,branch,cgpa,active_backlogs,dsa_problems_solved,dsa_score,coding_rating,web_cloud_score,system_design_score,quantitative_aptitude,logical_reasoning,verbal_ability,mock_interview_score,gd_score,communication_rating,projects_count,github_quality,internships_count,internship_duration_months,certifications_count,hackathons_participated,hackathons_won
STU_VALID_01,Computer Science & Engineering,8.5,0,250,75,1500,70,65,75,78,72,70,68,3.8,3,75,1,3,2,2,0
STU_INVALID_02,Information Technology,14.2,0,100,50,1100,50,45,60,60,60,50,50,3.0,1,50,0,0,1,1,0
STU_VALID_03,AI & Data Engineering,7.9,1,180,68,1400,60,55,70,72,68,65,60,3.2,2,65,0,0,1,1,0
"""
        import io
        data = {
            'file': (io.BytesIO(csv_content.encode('utf-8')), 'test_batch.csv')
        }
        res = self.client.post('/api/batch-predict', data=data, content_type='multipart/form-data')
        self.assertEqual(res.status_code, 200)
        batch_out = res.get_json()
        
        self.assertEqual(batch_out["status"], "success")
        self.assertEqual(batch_out["batch_summary"]["valid_records_count"], 2)
        self.assertEqual(batch_out["batch_summary"]["invalid_rows_count"], 1)
        self.assertEqual(len(batch_out["invalid_rows"]), 1)
        self.assertEqual(batch_out["invalid_rows"][0]["row_index"], 3)
        self.assertTrue(any("CGPA must be between 0.0 and 10.0" in e for e in batch_out["invalid_rows"][0]["errors"]))

    def test_07_transparent_readiness_scoring_logic(self):
        """Verifies transparent readiness scoring rules, weight contributions, caps, and band boundaries."""
        # Test 1: High preparation candidate
        high_candidate = {
            "dsa_score": 95, "dsa_problems_solved": 450, "coding_rating": 2100,
            "web_cloud_score": 90, "system_design_score": 85,
            "quantitative_aptitude": 90, "logical_reasoning": 92, "verbal_ability": 88,
            "mock_interview_score": 90, "gd_score": 88, "communication_rating": 4.5,
            "projects_count": 8, "github_quality": 90,
            "internships_count": 4, "internship_duration_months": 12,
            "certifications_count": 5, "hackathons_won": 2
        }
        res_high = compute_readiness_score(high_candidate)
        self.assertGreaterEqual(res_high["total_score"], 70.0)
        self.assertEqual(res_high["readiness_category"], "Strong preparation")
        self.assertEqual(len(res_high["breakdown"]), 5)
        
        # Test 2: Needs attention candidate
        low_candidate = {
            "dsa_score": 25, "dsa_problems_solved": 15, "coding_rating": 920,
            "web_cloud_score": 20, "system_design_score": 20,
            "quantitative_aptitude": 35, "logical_reasoning": 30, "verbal_ability": 35,
            "mock_interview_score": 30, "gd_score": 30, "communication_rating": 1.5,
            "projects_count": 0, "github_quality": 20,
            "internships_count": 0, "internship_duration_months": 0,
            "certifications_count": 0, "hackathons_won": 0
        }
        res_low = compute_readiness_score(low_candidate)
        self.assertLess(res_low["total_score"], 40.0)
        self.assertEqual(res_low["readiness_category"], "Needs attention")

        # Test 3: Missing inputs handling
        res_empty = compute_readiness_score({})
        self.assertIn("missing_fields", res_empty)
        self.assertIn("unassessed_components", res_empty)
        self.assertTrue(len(res_empty["unassessed_components"]) > 0)

    def test_08_authentication_and_session_tokens(self):
        """Verifies PBKDF2 password hashing, registration, login, and HMAC token validation."""
        import time as _t
        test_user = f"test_stu_{int(_t.time() * 1000)}"
        pwd = "SecureStudentPass!123"
        ok, msg, data = self.auth.register_user(test_user, pwd, "student", "Unit Test Student", f"{test_user}@lpu.in")
        self.assertTrue(ok, f"Registration failed: {msg}")
        self.assertIn("token", data)
        token = data["token"]
        
        # Verify token
        payload = self.auth.verify_token(token)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["username"], test_user)
        self.assertEqual(payload["role"], "student")
        
        # Authenticate with valid password
        auth_ok, _, _ = self.auth.authenticate_user(test_user, pwd)
        self.assertTrue(auth_ok)
        
        # Reject invalid password
        fail_ok, _, _ = self.auth.authenticate_user(test_user, "WrongPassword")
        self.assertFalse(fail_ok)

    def test_09_mentor_assignment_and_feedback(self):
        """Verifies mentor assignment rosters and mentor feedback logging."""
        assigned = self.auth.get_mentor_assigned_students("mentor1")
        self.assertGreater(len(assigned), 0)
        self.assertTrue(self.auth.is_student_assigned_to_mentor("mentor1", "LPU_DEMO_ADV01"))
        
        fid = self.auth.add_mentor_feedback(
            mentor_username="mentor1",
            assessment_id=1,
            student_id="LPU_DEMO_ADV01",
            feedback_notes="Strong algorithmic mastery. Continue LeetCode Medium sets.",
            recommended_focus="Communication & STAR method"
        )
        self.assertGreater(fid, 0)
        
        notes = self.auth.get_feedback_for_student("LPU_DEMO_ADV01")
        self.assertGreater(len(notes), 0)
        self.assertEqual(notes[0]["mentor_username"], "mentor1")

    def test_10_individual_report_retrieval_and_permissions(self):
        """Verifies health endpoint and report retrieval authorization."""
        res_h = self.client.get("/api/health")
        self.assertEqual(res_h.status_code, 200)
        h_data = res_h.get_json()
        self.assertEqual(h_data["status"], "healthy")
        self.assertTrue(h_data["supervised_model_loaded"])
        
        # Authenticate user to obtain token
        _, _, auth_data = self.auth.authenticate_user("student1", "Student@123")
        token = auth_data["token"]
        
        # Submit assessment for test student with user token so ownership is established
        student_data = dict(PRESET_STUDENTS[1]["data"])
        student_data["student_id"] = "student1"
        headers = {"Authorization": f"Bearer {token}"}
        res = self.client.post("/api/predict", json=student_data, headers=headers)
        self.assertEqual(res.status_code, 200)
        assessment_id = res.get_json()["result"]["assessment_history_id"]
        
        # Authorized retrieval
        res_rep = self.client.get(f"/api/report/{assessment_id}", headers=headers)
        self.assertEqual(res_rep.status_code, 200)
        rep = res_rep.get_json()["report"]
        self.assertEqual(rep["id"], assessment_id)
        self.assertEqual(rep["student_id"], "student1")
        self.assertIn("readiness_score", rep)

    def test_11_canonical_3tier_category_thresholds(self):
        """Verifies canonical 3-tier readiness category boundaries and legacy mappings."""
        # 1. Boundary checks
        self.assertEqual(categorize_readiness_score(0.0), "Needs attention")
        self.assertEqual(categorize_readiness_score(39.9), "Needs attention")
        self.assertEqual(categorize_readiness_score(40.0), "Developing")
        self.assertEqual(categorize_readiness_score(69.9), "Developing")
        self.assertEqual(categorize_readiness_score(70.0), "Strong preparation")
        self.assertEqual(categorize_readiness_score(100.0), "Strong preparation")

        # 2. Legacy 4-tier to 3-tier mapping checks
        self.assertEqual(map_legacy_tier_to_category("Foundation"), "Needs attention")
        self.assertEqual(map_legacy_tier_to_category("foundation"), "Needs attention")
        self.assertEqual(map_legacy_tier_to_category("Developing"), "Developing")
        self.assertEqual(map_legacy_tier_to_category("Placement Ready"), "Strong preparation")
        self.assertEqual(map_legacy_tier_to_category("Advanced"), "Strong preparation")

    def test_12_skill_gap_tolerance_and_negative_deltas(self):
        """Verifies that negative gaps are never marked as 'On Target' and respect +/- 3.0 pt tolerance."""
        student = dict(PRESET_STUDENTS[0]["data"])
        # Set DSA score slightly below benchmark
        student["dsa_score"] = 73.0 # target role benchmark is 80.0 -> delta is -7.0 pts
        result = self.engine.predict_student(student)
        
        gaps = result["gap_analysis"]["pillars_breakdown"]
        for row in gaps:
            # Check gap formula: delta = student_score - target_score
            expected_delta = round(row["student_score"] - row["target_score"], 1)
            self.assertEqual(row["delta"], expected_delta)
            
            # Critical requirement: Negative delta must NEVER be 'On Target'
            if row["delta"] < -3.0:
                self.assertNotIn("On Target", row["status"])
                self.assertIn(row["status"], ["Below benchmark", "Critical gap"])
            elif -3.0 <= row["delta"] < 0:
                self.assertEqual(row["status"], "Within benchmark tolerance")

    def test_13_feature_importance_descending_sort(self):
        """Verifies feature importances are sorted in strictly descending order with active backlogs audit."""
        meta = self.engine.model_metadata
        self.assertIn("sorted_feature_importances", meta)
        sorted_feats = meta["sorted_feature_importances"]
        
        # Check strictly non-increasing order
        for i in range(len(sorted_feats) - 1):
            curr_pct = sorted_feats[i]["relative_pct"]
            next_pct = sorted_feats[i + 1]["relative_pct"]
            self.assertGreaterEqual(curr_pct, next_pct, f"Feature importances not descending at index {i}")
            
        # Check active backlogs explanation is present
        self.assertIn("feature_importances_explanation", meta)
        explanation = meta["feature_importances_explanation"].lower()
        self.assertTrue("active_backlogs" in explanation or "active" in explanation or "backlogs" in explanation)
        self.assertIn("gatekeeper", explanation)

    def test_14_guest_isolation_and_demo_filtering(self):
        """Verifies unauthenticated guests only view demo records."""
        # Guest request to /api/history without token
        res = self.client.get("/api/history")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        
        for record in data["recent_assessments"]:
            self.assertEqual(record.get("is_demo", 0), 1, "Guest was served a non-demo student record")

if __name__ == "__main__":
    unittest.main()
