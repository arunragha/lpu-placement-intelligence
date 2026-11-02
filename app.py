"""
LPU Placement Intelligence & Student Readiness Prediction Platform - Flask Backend
Version: 2.1.0
Version: 2.2.0

Provides RESTful APIs for:
- Transparent Student Readiness Assessment (5-Pillars, 0-100 score, transparent rubric)
- Supervised Machine Learning Placement Prediction (Binary outcome & probability)
- Decision Tree Ensemble Path Attribution (Saabas Tree Decomposition)
- Institutional Eligibility Checking (backlogs, CGPA, attendance)
- Role-Based Authentication & Authorization (Student, Mentor, Admin)
- Longitudinal Assessment History & Detailed Report Retrieval
- Mentor Management, Assigned Student Tracking & Feedback
- Batch TPO Cohort Analytics & CSV Processing
"""

import os
import io
import csv
import json
import numpy as np
import pandas as pd
from functools import wraps
from flask import Flask, request, jsonify, send_from_directory, send_file

from src.ml_engine import (
    PlacementIntelligenceEngine, PRESET_STUDENTS,
    FEATURE_DISPLAY_NAMES, ILLUSTRATIVE_ROLE_TARGETS
)
from src.validator import validate_student_assessment
from src.history_store import AssessmentHistoryStore
from src.auth_service import AuthService

app = Flask(__name__, static_folder="static", static_url_path="")

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response

# Global Engine & Store Holders
engine = None
history_store = None
auth_service = None

def get_engine():
    global engine
    if engine is None:
        engine = PlacementIntelligenceEngine()
    return engine

def get_history_store():
    global history_store
    if history_store is None:
        history_store = AssessmentHistoryStore()
    return history_store

def get_auth_service():
    global auth_service
    if auth_service is None:
        auth_service = AuthService()
    return auth_service

def get_current_user():
    """Extracts and verifies the authenticated user from the Authorization header."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1].strip()
        return get_auth_service().verify_token(token)
    return None

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = get_current_user()
        if not user:
            return jsonify({"status": "error", "message": "Authentication required. Please log in."}), 401
        return f(user, *args, **kwargs)
    return decorated_function

def role_required(allowed_roles):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            user = get_current_user()
            if not user:
                return jsonify({"status": "error", "message": "Authentication required. Please log in."}), 401
            if user.get("role") not in allowed_roles:
                return jsonify({"status": "error", "message": f"Forbidden: Requires one of {allowed_roles} role(s)."}), 403
            return f(user, *args, **kwargs)
        return decorated_function
    return decorator


# =============================================================================
# 1. CORE SYSTEM & STATIC ROUTES
# =============================================================================

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/static/<path:filename>")
def serve_static_alias(filename):
    """Fallback handler to ensure static assets resolve with or without /static/ prefix."""
    return send_from_directory("static", filename)

@app.route("/favicon.ico")
def favicon():
    return "", 204

@app.route("/api/health", methods=["GET"])
def health_check():
    eng = get_engine()
    return jsonify({
        "status": "healthy",
        "version": "2.1.0",
        "version": "2.2.0",
        "platform": "LPU Placement Intelligence & Student Readiness Prediction Platform",
        "supervised_model_loaded": eng.placement_clf is not None,
        "readiness_classifier_loaded": eng.readiness_clf is not None,
        "demo_mode": True,
        "disclaimer": "Academic research prototype evaluated on synthetic rubric benchmarks; not affiliated with official LPU recruitment administration."
    })

@app.route("/api/presets", methods=["GET"])
def get_presets():
    return jsonify({
        "status": "success",
        "presets": PRESET_STUDENTS,
        "supported_roles": list(ILLUSTRATIVE_ROLE_TARGETS.keys()),
        "notice": "Demo student archetypes provided for evaluation and demonstration."
    })

@app.route("/api/model-performance", methods=["GET"])
def get_model_performance():
    try:
        eng = get_engine()
        return jsonify({
            "status": "success",
            "metadata": eng.metadata
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# =============================================================================
# 2. AUTHENTICATION & USER PROFILE
# =============================================================================

@app.route("/api/auth/register", methods=["POST"])
def auth_register():
    try:
        data = request.get_json(force=True) or {}
        username = data.get("username", "")
        password = data.get("password", "")
        role = data.get("role", "student")
        full_name = data.get("full_name", "")
        email = data.get("email", "")
        student_id = data.get("student_reference_id", "")
        
        auth = get_auth_service()
        success, msg, res_data = auth.register_user(
            username=username,
            password=password,
            role=role,
            full_name=full_name,
            email=email,
            student_reference_id=student_id
        )
        if not success:
            return jsonify({"status": "error", "message": msg}), 400
        return jsonify({"status": "success", "message": msg, "data": res_data}), 201
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    try:
        data = request.get_json(force=True) or {}
        username = data.get("username", "")
        password = data.get("password", "")
        
        auth = get_auth_service()
        success, msg, res_data = auth.authenticate_user(username, password)
        if not success:
            return jsonify({"status": "error", "message": msg}), 401
        return jsonify({"status": "success", "message": msg, "data": res_data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/auth/me", methods=["GET"])
def auth_me():
    user = get_current_user()
    if not user:
        return jsonify({"status": "guest", "authenticated": False, "role": "guest"})
    return jsonify({
        "status": "success",
        "authenticated": True,
        "user": user
    })


# =============================================================================
# 3. STUDENT ASSESSMENT & PREDICTION
# =============================================================================

@app.route("/api/predict", methods=["POST"])
def predict_single_student():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"status": "error", "message": "No JSON payload provided"}), 400

        # Server-side validation
        is_valid, errors, warnings = validate_student_assessment(data)
        if not is_valid:
            return jsonify({
                "status": "error",
                "message": "Input validation failed. Please correct the highlighted errors.",
                "errors": errors
            }), 400

        current_user = get_current_user()
        owner_username = current_user["username"] if current_user else None

        eng = get_engine()
        result = eng.predict_student(data)

        # Persist to local SQLite assessment history
        try:
            store = get_history_store()
            history_id = store.save_assessment(data, result, owner_username=owner_username)
            result["assessment_history_id"] = history_id
        except Exception as e:
            print(f"Warning: Failed to save assessment history: {e}")

        return jsonify({
            "status": "success",
            "result": result,
            "warnings": warnings
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# =============================================================================
# 4. ASSESSMENT HISTORY & REPORT RETRIEVAL
# =============================================================================

@app.route("/api/history/<student_id>", methods=["GET"])
def get_student_history(student_id):
    try:
        current_user = get_current_user()
        
        # Security check: if authenticated as student, can only view own history
        if current_user and current_user["role"] == "student":
            ref_id = current_user.get("student_reference_id")
            if ref_id and ref_id != student_id and current_user["username"] != student_id:
                return jsonify({"status": "error", "message": "Access denied: Students can only view their own assessment records."}), 403
                
        # If authenticated as mentor, verify student assignment
        if current_user and current_user["role"] == "mentor":
            auth = get_auth_service()
            if not auth.is_student_assigned_to_mentor(current_user["username"], student_id):
                # Allow demo students for inspection
                if not student_id.startswith("LPU_DEMO_") and not student_id.startswith("LPU_UNIT_TEST_"):
                    return jsonify({"status": "error", "message": "Access denied: This student is not assigned to your mentor roster."}), 403

        # If guest (unauthenticated), allow only demo profiles
        if not current_user:
            if not student_id.startswith("LPU_DEMO_") and not student_id.startswith("DEMO_"):
                return jsonify({"status": "error", "message": "Authentication required to view non-demo student histories."}), 401

        store = get_history_store()
        records = store.get_student_history(student_id)
        
        # Format whole percentage display for model estimates
        for r in records:
            prob = r.get("placement_prob")
            prob_val = float(prob) if prob is not None else 0.0
            r["demo_model_estimate_pct"] = int(round(prob_val))
            r["demo_model_estimate_display"] = f"{int(round(prob_val))}%"
            
        return jsonify({
            "status": "success",
            "student_id": student_id,
            "count": len(records),
            "history": records
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/history", methods=["GET"])
def get_recent_history():
    try:
        current_user = get_current_user()
        store = get_history_store()
        
        limit = int(request.args.get("limit", 20))
        page = int(request.args.get("page", 1))
        offset = max(0, (page - 1) * limit)
        category_filter = request.args.get("category", "").strip() or None
        search_kw = request.args.get("q", "").strip() or None
        
        # If student is logged in, return only their history
        if current_user and current_user["role"] == "student":
            ref_id = current_user.get("student_reference_id") or current_user["username"]
            records = store.get_student_history(ref_id, owner_username=current_user["username"])
            # Format and apply in-memory search/filter if needed
            if category_filter and category_filter != "all":
                records = [r for r in records if r.get("readiness_category") == category_filter or r.get("readiness_band") == category_filter]
            if search_kw:
                skw = search_kw.lower()
                records = [r for r in records if skw in r.get("student_id", "").lower() or skw in r.get("target_role", "").lower()]
            total_count = len(records)
            paged_records = records[offset : offset + limit]
            res_obj = {
                "total_count": total_count,
                "assessments": paged_records,
                "limit": limit,
                "page": page
            }
        else:
            is_guest = current_user is None
            res_obj = store.get_recent_assessments(
                limit=limit,
                offset=offset,
                is_guest=is_guest,
                category_filter=category_filter,
                search=search_kw
            )
            
        # Standardize display fields on each record
        for r in res_obj["assessments"]:
            prob = r.get("placement_prob")
            prob_val = float(prob) if prob is not None else 0.0
            r["demo_model_estimate_pct"] = int(round(prob_val))
            r["demo_model_estimate_display"] = f"{int(round(prob_val))}%"
            
        return jsonify({
            "status": "success",
            "count": len(res_obj["assessments"]),
            "total_count": res_obj["total_count"],
            "page": res_obj.get("page", 1),
            "limit": limit,
            "recent_assessments": res_obj["assessments"],
            "is_guest_mode": current_user is None
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/report/<int:assessment_id>", methods=["GET"])
def get_assessment_report(assessment_id):
    try:
        store = get_history_store()
        record = store.get_assessment_by_id(assessment_id)
        if not record:
            return jsonify({"status": "error", "message": f"Assessment record #{assessment_id} not found."}), 404
            
        current_user = get_current_user()
        stu_id = record.get("student_id", "")
        is_demo_rec = bool(record.get("is_demo") or stu_id.startswith("LPU_DEMO_") or stu_id.startswith("DEMO_"))

        # Guest isolation: guests can only inspect demo reports
        if not current_user and not is_demo_rec:
            return jsonify({"status": "error", "message": "Authentication required: Private student assessment reports cannot be viewed anonymously."}), 401

        # Student ownership check
        if current_user and current_user["role"] == "student":
            owner = record.get("owner_username") or stu_id
            if owner != current_user["username"] and owner != current_user.get("student_reference_id") and not is_demo_rec:
                return jsonify({"status": "error", "message": "Access denied: You do not own this assessment report."}), 403

        # Mentor roster check
        if current_user and current_user["role"] == "mentor":
            auth = get_auth_service()
            if not is_demo_rec and not auth.is_student_assigned_to_mentor(current_user["username"], stu_id):
                return jsonify({"status": "error", "message": "Access denied: This student is not assigned to your mentor roster."}), 403

        # Attach any mentor feedback
        auth = get_auth_service()
        feedback = auth.get_feedback_for_student(stu_id)
        record["mentor_feedback"] = feedback

        # Add integer demo model estimate
        prob = record.get("placement_prob", 0.0)
        record["demo_model_estimate_pct"] = int(round(prob))
        record["demo_model_estimate_display"] = f"{int(round(prob))}%"

        return jsonify({
            "status": "success",
            "report": record
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# =============================================================================
# 5. MENTOR & TPO WORKFLOWS
# =============================================================================

@app.route("/api/mentor/students", methods=["GET"])
def get_mentor_students():
    try:
        current_user = get_current_user()
        auth = get_auth_service()
        store = get_history_store()
        eng = get_engine()
        
        mentor_name = current_user["username"] if current_user and current_user["role"] in ["mentor", "admin"] else "mentor1"
        assigned = auth.get_mentor_assigned_students(mentor_name)
        
        # Build student roster with their latest assessment status
        roster = []
        for a in assigned:
            stu_id = a["student_id"]
            hist = store.get_student_history(stu_id)
            if hist:
                latest = hist[-1]
                prob = latest.get("placement_prob", 0.0)
                band = latest.get("readiness_category") or latest.get("readiness_band") or latest.get("readiness_tier")
                if band in ["Needs attention", "Developing", "Strong preparation"]:
                    cat = band
                elif band == "Foundation":
                    cat = "Needs attention"
                elif band == "Developing":
                    cat = "Developing"
                else:
                    cat = "Strong preparation"
                    
                roster.append({
                    "student_id": stu_id,
                    "assigned_at": a["assigned_at"],
                    "latest_assessment_id": latest["id"],
                    "timestamp": latest["timestamp"],
                    "branch": latest["branch"],
                    "target_role": latest["target_role"],
                    "cgpa": latest["cgpa"],
                    "readiness_tier": cat,
                    "readiness_band": cat,
                    "readiness_category": cat,
                    "readiness_score": latest.get("readiness_score", 0.0),
                    "demo_model_estimate_pct": int(round(prob)),
                    "demo_model_estimate": f"{int(round(prob))}%",
                    "placement_prob": prob
                })
            else:
                # Fallback to preset if demo student
                preset = next((p for p in PRESET_STUDENTS if p["data"]["student_id"] == stu_id), None)
                if preset:
                    pred_res = eng.predict_student(preset["data"])
                    cat = pred_res["prediction"]["readiness_category"]
                    prob = pred_res["prediction"].get("simulated_placement_prob", 50.0)
                    roster.append({
                        "student_id": stu_id,
                        "assigned_at": a["assigned_at"],
                        "latest_assessment_id": None,
                        "timestamp": "Demo Profile",
                        "branch": preset["data"]["branch"],
                        "target_role": preset["data"]["target_role"],
                        "cgpa": preset["data"]["cgpa"],
                        "readiness_tier": cat,
                        "readiness_band": cat,
                        "readiness_category": cat,
                        "readiness_score": pred_res.get("readiness_assessment", {}).get("total_score", 65.0),
                        "demo_model_estimate_pct": int(round(prob)),
                        "demo_model_estimate": f"{int(round(prob))}%",
                        "placement_prob": prob
                    })
                else:
                    roster.append({
                        "student_id": stu_id,
                        "assigned_at": a["assigned_at"],
                        "latest_assessment_id": None,
                        "timestamp": "Not yet assessed",
                        "branch": "Computer Science & Engineering",
                        "target_role": "Software Development Engineer (SDE)",
                        "cgpa": 0.0,
                        "readiness_tier": "Needs attention",
                        "readiness_band": "Needs attention",
                        "readiness_category": "Needs attention",
                        "readiness_score": 0.0,
                        "demo_model_estimate_pct": 0,
                        "demo_model_estimate": "0%",
                        "placement_prob": 0.0
                    })
                    
        return jsonify({
            "status": "success",
            "mentor_username": mentor_name,
            "total_assigned_students": len(roster),
            "students": roster
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/mentor/feedback", methods=["POST"])
def submit_mentor_feedback():
    try:
        current_user = get_current_user()
        if not current_user or current_user.get("role") not in ["mentor", "admin"]:
            return jsonify({"status": "error", "message": "Access denied: Only mentors can submit evaluation feedback."}), 403
            
        data = request.get_json(force=True) or {}
        assessment_id = int(data.get("assessment_id") or 0)
        student_id = str(data.get("student_id") or "").strip()
        notes = str(data.get("feedback_notes") or "").strip()
        focus = str(data.get("recommended_focus") or "").strip()
        
        if not student_id or not notes:
            return jsonify({"status": "error", "message": "Fields 'student_id' and 'feedback_notes' are required."}), 400
            
        auth = get_auth_service()
        fid = auth.add_mentor_feedback(
            mentor_username=current_user["username"],
            assessment_id=assessment_id,
            student_id=student_id,
            feedback_notes=notes,
            recommended_focus=focus
        )
        return jsonify({
            "status": "success",
            "message": "Mentor feedback recorded successfully.",
            "feedback_id": fid
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/student/feedback/<student_id>", methods=["GET"])
def get_student_feedback(student_id):
    try:
        auth = get_auth_service()
        notes = auth.get_feedback_for_student(student_id)
        return jsonify({
            "status": "success",
            "student_id": student_id,
            "count": len(notes),
            "feedback": notes
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# =============================================================================
# 6. BATCH COHORT TPO ANALYTICS
# =============================================================================

@app.route("/api/batch-predict", methods=["POST"])
def batch_predict():
    """
    Cohort CSV / JSON batch assessment for Training & Placement Officers (TPO).
    Performs row-by-row validation, reports exact invalid rows, computes readiness distributions,
    completeness rates, and flags students requiring intervention.
    """
    try:
        eng = get_engine()
        store = get_history_store()
        
        if "file" in request.files:
            file = request.files["file"]
            try:
                df = pd.read_csv(file)
            except Exception as e:
                return jsonify({"status": "error", "message": f"Malformed CSV file: {str(e)}"}), 400
        elif request.is_json:
            json_data = request.get_json()
            if isinstance(json_data, list):
                df = pd.DataFrame(json_data)
            elif "students" in json_data:
                df = pd.DataFrame(json_data["students"])
            else:
                return jsonify({"status": "error", "message": "Invalid JSON format for batch"}), 400
        else:
            return jsonify({"status": "error", "message": "Expected file upload or JSON list"}), 400
            
        if len(df) == 0:
            return jsonify({"status": "error", "message": "Empty student batch dataset provided."}), 400

        valid_results = []
        invalid_rows = []
        tier_counts = {"Advanced": 0, "Placement Ready": 0, "Developing": 0, "Foundation": 0}
        band_counts = {"Strong preparation": 0, "Developing": 0, "Needs attention": 0}
        branch_readiness = {}
        all_bottlenecks = {}
        students_needing_support = []
        total_completeness = 0.0
        ctc_values = []
        prob_values = []
        readiness_scores = []

        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            student_id = str(row_dict.get("student_id", f"STU_{idx+1:03d}")).strip()
            
            is_valid, errors, _ = validate_student_assessment(row_dict)
            if not is_valid:
                invalid_rows.append({
                    "row_index": int(idx + 2),
                    "student_id": student_id,
                    "errors": errors
                })
                continue
                
            pred_out = eng.predict_student(row_dict)
            pred = pred_out["prediction"]
            gaps = pred_out["gap_analysis"]
            comp = pred_out["completeness"]
            elig = pred_out["eligibility"]
            readiness = pred_out.get("readiness_assessment", {})
            
            tier = pred["readiness_tier"]
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            
            band = readiness.get("readiness_band", "Developing")
            band_counts[band] = band_counts.get(band, 0) + 1
            
            r_score = readiness.get("total_score", 50.0)
            readiness_scores.append(r_score)
            
            prob = pred["simulated_placement_prob"]
            ctc = pred["simulated_expected_ctc_lpa"]
            prob_values.append(prob)
            ctc_values.append(ctc)
            
            total_completeness += comp["completeness_pct"]
            
            branch = str(row_dict.get("branch", "Computer Science & Engineering"))
            if branch not in branch_readiness:
                branch_readiness[branch] = {"total": 0, "tier_distribution": {}}
            branch_readiness[branch]["total"] += 1
            branch_readiness[branch]["tier_distribution"][tier] = branch_readiness[branch]["tier_distribution"].get(tier, 0) + 1
            
            for g in gaps["pillars_breakdown"]:
                if g["badge"] in ["danger", "critical"]:
                    p_name = g["pillar"]
                    all_bottlenecks[p_name] = all_bottlenecks.get(p_name, 0) + 1
                    
            if tier == "Foundation" or band == "Needs attention" or int(row_dict.get("active_backlogs", 0)) > 0:
                students_needing_support.append({
                    "student_id": student_id,
                    "branch": branch,
                    "readiness_tier": tier,
                    "readiness_band": band,
                    "readiness_score": r_score,
                    "active_backlogs": int(row_dict.get("active_backlogs", 0)),
                    "cgpa": float(row_dict.get("cgpa", 0.0)),
                    "primary_deficit": gaps["top_improvement_priorities"][0] if gaps["top_improvement_priorities"] else "Core Academic Standing"
                })

            valid_results.append({
                "student_id": student_id,
                "branch": branch,
                "target_role": pred["target_role"],
                "cgpa": float(row_dict.get("cgpa", 0.0)),
                "dsa_problems_solved": int(row_dict.get("dsa_problems_solved", 0)),
                "readiness_tier": tier,
                "readiness_band": band,
                "readiness_score": r_score,
                "assessment_completeness_pct": comp["completeness_pct"],
                "eligibility_status": elig["status"],
                "simulated_placement_prob": prob,
                "simulated_expected_ctc_lpa": ctc,
                "critical_gaps_count": gaps["critical_gap_count"]
            })

            try:
                store.save_assessment(row_dict, pred_out)
            except Exception:
                pass

        n_valid = len(valid_results)
        if n_valid == 0:
            return jsonify({
                "status": "error",
                "message": "All uploaded rows contained validation errors. Please fix the CSV format.",
                "invalid_rows": invalid_rows
            }), 400

        cohort_mean_ctc = round(float(np.mean(ctc_values)), 2)
        cohort_median_ctc = round(float(np.median(ctc_values)), 2)
        cohort_mean_prob = round(float(np.mean(prob_values)), 1)
        cohort_mean_readiness = round(float(np.mean(readiness_scores)), 1)
        avg_completeness = round(total_completeness / n_valid, 1)

        bottleneck_summary = [
            {"pillar": k, "affected_students_count": v, "percentage": round((v / n_valid) * 100, 1)}
            for k, v in sorted(all_bottlenecks.items(), key=lambda x: x[1], reverse=True)
        ]

        return jsonify({
            "status": "success",
            "batch_summary": {
                "total_rows_evaluated": len(df),
                "valid_records_count": n_valid,
                "invalid_rows_count": len(invalid_rows),
                "cohort_average_completeness_pct": avg_completeness,
                "cohort_mean_readiness_score": cohort_mean_readiness,
                "cohort_mean_placement_prob": cohort_mean_prob,
                "cohort_mean_ctc_lpa": cohort_mean_ctc,
                "cohort_median_ctc_lpa": cohort_median_ctc,
                "tier_distribution": tier_counts,
                "readiness_band_distribution": band_counts,
                "branch_readiness": branch_readiness,
                "cohort_bottlenecks": bottleneck_summary,
                "priority_intervention_students": students_needing_support[:10]
            },
            "invalid_rows": invalid_rows,
            "students": valid_results
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/sample-csv", methods=["GET"])
def download_sample_csv():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_dir, "data", "lpu_placement_dataset.csv")
        
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            sample_df = df.head(15).copy()
            drop_cols = ["readiness_tier", "placement_probability", "expected_ctc_lpa", "best_fit_role", "gender", "placed_binary"]
            sample_df = sample_df.drop(columns=[c for c in drop_cols if c in sample_df.columns])
            
            sample_df["target_role"] = "Software Development Engineer (SDE)"
            sample_df["weekly_prep_hours"] = "15-20 hrs/week"
            
            output = io.StringIO()
            sample_df.to_csv(output, index=False)
            output.seek(0)
            
            return send_file(
                io.BytesIO(output.getvalue().encode('utf-8')),
                mimetype="text/csv",
                as_attachment=True,
                download_name="lpu_sample_batch_students_v2.csv"
            )
        else:
            return jsonify({"status": "error", "message": "Sample dataset not available"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

def enable_dual_stack_ipv6():
    """Configures Werkzeug BaseWSGIServer socket to support dual-stack IPv6 and IPv4 on Windows."""
    try:
        import socket
        import werkzeug.serving
        orig_bind = werkzeug.serving.BaseWSGIServer.server_bind
        def dual_stack_bind(self):
            if self.address_family == socket.AF_INET6 and hasattr(socket, "IPV6_V6ONLY"):
                try:
                    self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
                except Exception:
                    pass
            orig_bind(self)
        werkzeug.serving.BaseWSGIServer.server_bind = dual_stack_bind
    except Exception:
        pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    enable_dual_stack_ipv6()
    print(f"Starting LPU Placement Intelligence Server v2.1 on:")
    print(f"  --> Localhost:  http://localhost:{port}")
    print(f"  --> IPv4 Loop:  http://127.0.0.1:{port}")
    print(f"  --> All Hosts:  http://0.0.0.0:{port}")
    try:
        # Dual-stack IPv6 bind (with IPV6_V6ONLY=0) accepts ::1, localhost, 127.0.0.1, and all IPv4 interfaces
        app.run(host="::", port=port, debug=False)
    except Exception as e:
        print(f"Notice: IPv6 dual-stack fallback ({e}). Binding to IPv4 0.0.0.0 ...")
        app.run(host="0.0.0.0", port=port, debug=False)
