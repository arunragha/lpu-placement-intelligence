"""
LPU Placement Intelligence Platform - Live Server Verification Suite v2.2
========================================================================
Performs comprehensive live HTTP smoke, contract, and resilience testing against a running Flask backend instance.

Usage:
    python test_live_server.py
    python test_live_server.py http://127.0.0.1:5000
    python test_live_server.py http://localhost:8000
"""

import sys
import os
import json
import time
import urllib.request
import urllib.error

# Force safe encoding for Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Configurable target URL: CLI argument > SERVER_URL environment variable > Default localhost
BASE_URL = (
    sys.argv[1].rstrip("/")
    if len(sys.argv) > 1
    else os.getenv("SERVER_URL", "http://127.0.0.1:5000").rstrip("/")
)

class LiveTestRunner:
    """Lightweight zero-dependency test runner with latency tracking and exit status handling."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.start_time = time.time()

    def check(self, condition: bool, description: str, details: str = "") -> bool:
        if condition:
            self.passed += 1
            print(f"  [PASS] {description}", flush=True)
            if details:
                print(f"         |-- {details}", flush=True)
            return True
        else:
            self.failed += 1
            print(f"  [FAIL] {description}", flush=True)
            if details:
                print(f"         |-- ERROR: {details}", flush=True)
            return False

    def request(self, method: str, endpoint: str, payload=None, extra_headers=None, timeout: float = 10.0):
        url = f"{BASE_URL}{endpoint}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        if extra_headers:
            headers.update(extra_headers)
        req = urllib.request.Request(url, data=data, headers=headers, method=method)

        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as res:
                latency_ms = round((time.time() - t0) * 1000, 1)
                content_type = res.headers.get_content_type()
                raw_bytes = res.read()
                if "application/json" in content_type:
                    body = json.loads(raw_bytes.decode("utf-8"))
                else:
                    body = raw_bytes
                return res.status, body, res.headers, latency_ms
        except urllib.error.HTTPError as e:
            latency_ms = round((time.time() - t0) * 1000, 1)
            raw_bytes = e.read()
            try:
                body = json.loads(raw_bytes.decode("utf-8"))
            except Exception:
                body = raw_bytes.decode("utf-8", errors="replace")
            return e.code, body, e.headers, latency_ms


def log(msg=""):
    print(msg, flush=True)


def main():
    log("=" * 72)
    log(" LPU PLACEMENT INTELLIGENCE - LIVE SERVER VERIFICATION SUITE v2.2")
    log(f" Target Server : {BASE_URL}")
    log("=" * 72)

    runner = LiveTestRunner()

    # Pre-Flight: Check if server is running
    log("\n[Step 0] Verifying Server Connectivity...")
    try:
        with urllib.request.urlopen(BASE_URL, timeout=3.0) as res:
            runner.check(
                res.status == 200,
                f"Server reachable at {BASE_URL}",
                f"HTTP {res.status} OK"
            )
    except urllib.error.URLError as e:
        log(f"\n{'!' * 72}")
        log(f" [FATAL] Cannot connect to server at: {BASE_URL}")
        log(f" Reason: {e.reason}")
        log("\n Remedy:")
        log("   1. Start the Flask server in a separate terminal:")
        log("      python app.py")
        log("   2. Or specify an active custom server address:")
        log("      python test_live_server.py http://127.0.0.1:5000")
        log(f"{'!' * 72}\n")
        sys.exit(1)

    # 1. Static Web Application (Landing Page)
    log("\n[Step 1] Testing GET / (Static Web Application)...")
    status, body, headers, lat = runner.request("GET", "/")
    runner.check(status == 200, "Frontend index served", f"Latency: {lat}ms, Size: {len(body)} bytes")
    runner.check("text/html" in headers.get("Content-Type", ""), "Response is HTML")

    # 1b. Health & System Diagnostic Check
    log("\n[Step 1b] Testing GET /api/health...")
    status, health_data, _, lat = runner.request("GET", "/api/health")
    runner.check(status == 200, "Health endpoint responded 200 OK", f"Latency: {lat}ms")
    runner.check(
        isinstance(health_data, dict) and health_data.get("status") == "healthy",
        "System health status is healthy"
    )
    runner.check(health_data.get("supervised_model_loaded") is True, "Supervised placement model loaded")
    runner.check(health_data.get("readiness_classifier_loaded") is True, "Readiness classifier loaded")

    # 2. Presets Endpoint
    log("\n[Step 2] Testing GET /api/presets...")
    status, presets_data, headers, lat = runner.request("GET", "/api/presets")
    runner.check(status == 200, "Presets endpoint responded", f"Latency: {lat}ms")
    runner.check(
        isinstance(presets_data, dict) and presets_data.get("status") == "success",
        "Presets response status is success"
    )
    presets = presets_data.get("presets", []) if isinstance(presets_data, dict) else []
    runner.check(len(presets) >= 4, f"Returned {len(presets)} presets (Expected >= 4)")
    supported_roles = presets_data.get("supported_roles", []) if isinstance(presets_data, dict) else []
    runner.check(len(supported_roles) > 0, f"Returned {len(supported_roles)} supported career roles")

    # 3. Model Inference Across Presets + Contract Check
    log("\n[Step 3] Testing POST /api/predict Across Presets...")
    for idx, preset in enumerate(presets):
        p_name = preset.get("name", f"Preset {idx}")
        p_data = preset.get("data", {})
        status, pred_data, headers, lat = runner.request("POST", "/api/predict", payload=p_data)
        
        ok = runner.check(
            status == 200 and isinstance(pred_data, dict) and pred_data.get("status") == "success",
            f"Prediction succeeded for '{p_name}'",
            f"Latency: {lat}ms"
        )
        if ok:
            res_obj = pred_data.get("result", {})
            pred = res_obj.get("prediction", {})
            comp = res_obj.get("completeness", {})
            elig = res_obj.get("eligibility", {})
            explain = res_obj.get("explainability", {})

            # Value contract assertions
            tier = pred.get("readiness_tier")
            runner.check(
                tier in ["Foundation", "Developing", "Placement Ready", "Advanced", "Needs attention", "Strong preparation"],
                f"Valid readiness tier: {tier}"
            )
            # Canonical 3-tier check
            cat = pred.get("readiness_category")
            runner.check(
                cat in ["Needs attention", "Developing", "Strong preparation"],
                f"Canonical 3-tier readiness category: {cat}"
            )
            # Whole percentage demo estimate check
            demo_pct = pred.get("demo_model_estimate_pct")
            runner.check(
                isinstance(demo_pct, int) and 0 <= demo_pct <= 100,
                f"Demo model estimate is whole percentage integer: {demo_pct}%"
            )
            prob = pred.get("simulated_placement_prob", -1)
            runner.check(0.0 <= prob <= 100.0, f"Placement probability in range [0, 100]: {prob}%")
            ctc = pred.get("simulated_expected_ctc_lpa", -1)
            runner.check(ctc > 0.0, f"Simulated CTC positive: {ctc} LPA")
            runner.check("completeness_pct" in comp, f"Completeness tracked: {comp.get('completeness_pct')}%")
            runner.check("explanation" in comp, "Input completeness explanation note present")
            runner.check("status" in elig, f"Eligibility evaluated: {elig.get('status')}")
            runner.check(len(explain.get("primary_drivers", [])) > 0, "Explainability drivers present")

    # 4. Assessment History Persistence
    log("\n[Step 4] Testing Assessment History (SQLite Storage)...")
    if presets:
        target_stu_id = presets[0]["data"].get("student_id", "STU_ADV01")
        status, hist_data, _, lat = runner.request("GET", f"/api/history/{target_stu_id}")
        runner.check(status == 200, f"GET /api/history/{target_stu_id} responded", f"Latency: {lat}ms")
        count = hist_data.get("count", 0) if isinstance(hist_data, dict) else 0
        runner.check(count > 0, f"Stored {count} assessment history records for {target_stu_id}")

    # General recent history
    status, recent_data, _, lat = runner.request("GET", "/api/history")
    runner.check(status == 200, "GET /api/history responded", f"Latency: {lat}ms")
    recent_list = recent_data.get("recent_assessments", []) if isinstance(recent_data, dict) else []
    runner.check(isinstance(recent_list, list), f"Found {len(recent_list)} recent cohort assessments")

    # 5. Validation Rejection (Negative Testing)
    log("\n[Step 5] Testing Validation Rejection (Negative Tests)...")
    if presets:
        # 5a. Out-of-bounds CGPA
        bad_cgpa = dict(presets[0]["data"], cgpa=14.5)
        status, err_body, _, _ = runner.request("POST", "/api/predict", payload=bad_cgpa)
        runner.check(
            status == 400 and isinstance(err_body, dict) and "errors" in err_body,
            "Rejects invalid CGPA > 10.0 with HTTP 400",
            f"Reported: {err_body.get('errors', [''])[0] if isinstance(err_body, dict) else err_body}"
        )

        # 5b. Empty payload
        status, err_empty, _, _ = runner.request("POST", "/api/predict", payload={})
        runner.check(
            status == 400,
            "Rejects empty JSON payload with HTTP 400"
        )

    # 6. TPO Batch Cohort Prediction
    log("\n[Step 6] Testing POST /api/batch-predict (Cohort Batch Analytics)...")
    if len(presets) >= 2:
        batch_payload = {"students": [p["data"] for p in presets]}
        status, batch_res, _, lat = runner.request("POST", "/api/batch-predict", payload=batch_payload)
        runner.check(status == 200, "POST /api/batch-predict responded 200 OK", f"Latency: {lat}ms")
        if isinstance(batch_res, dict):
            summary = batch_res.get("batch_summary", {})
            valid_cnt = summary.get("valid_records_count", 0)
            runner.check(valid_cnt == len(presets), f"Successfully processed all {valid_cnt} batch records")
            runner.check("cohort_mean_ctc_lpa" in summary, f"Computed cohort mean CTC: {summary.get('cohort_mean_ctc_lpa')} LPA")
            runner.check("tier_distribution" in summary, "Computed cohort tier distribution")

    # 7. Sample CSV Download
    log("\n[Step 7] Testing GET /api/sample-csv...")
    status, csv_content, headers, lat = runner.request("GET", "/api/sample-csv")
    runner.check(status == 200, "Sample CSV download responded 200 OK", f"Latency: {lat}ms")
    is_csv = isinstance(csv_content, (bytes, str)) and (b"student_id" in csv_content if isinstance(csv_content, bytes) else "student_id" in csv_content)
    runner.check(is_csv, "Sample CSV contains valid student headers")

    # 8. Model Diagnostics & Performance Metrics
    log("\n[Step 8] Testing GET /api/model-performance...")
    status, perf_data, headers, lat = runner.request("GET", "/api/model-performance")
    runner.check(status == 200, "Model performance endpoint responded 200 OK", f"Latency: {lat}ms")
    meta = perf_data.get("metadata", {}) if isinstance(perf_data, dict) else {}
    runner.check("best_classifier_name" in meta, f"Reported Best Classifier: {meta.get('best_classifier_name')}")
    bench = meta.get("classification_models_benchmark", {})
    runner.check(len(bench) > 0, f"Reported benchmarks for {len(bench)} candidate models")

    # 9. CORS Headers Verification
    log("\n[Step 9] Verifying Security & CORS Headers...")
    cors_origin = headers.get("Access-Control-Allow-Origin", "")
    runner.check(cors_origin == "*", f"Access-Control-Allow-Origin is configured ('{cors_origin}')")

    # 10. Authentication Workflow
    log("\n[Step 10] Testing Authentication & RBAC...")
    # 10a. Login as Student
    status, stu_auth, _, lat = runner.request("POST", "/api/auth/login", payload={"username": "student1", "password": "Student@123"})
    runner.check(status == 200, "Student authentication successful", f"Latency: {lat}ms")
    stu_token = stu_auth.get("data", {}).get("token", "") if isinstance(stu_auth, dict) else ""
    runner.check(bool(stu_token), "Received valid session token for student")

    # 10b. Verify current user via /api/auth/me
    status, me_data, _, lat = runner.request("GET", "/api/auth/me", extra_headers={"Authorization": f"Bearer {stu_token}"})
    runner.check(status == 200 and me_data.get("authenticated") is True, "Token verified via /api/auth/me")
    runner.check(me_data.get("user", {}).get("role") == "student", "User identity has student role")

    # 10c. Login as Mentor
    status, mentor_auth, _, lat = runner.request("POST", "/api/auth/login", payload={"username": "mentor1", "password": "Mentor@123"})
    runner.check(status == 200, "Mentor authentication successful", f"Latency: {lat}ms")
    mentor_token = mentor_auth.get("data", {}).get("token", "") if isinstance(mentor_auth, dict) else ""
    runner.check(bool(mentor_token), "Received valid session token for mentor")

    # 11. Mentor Hub & Feedback Operations
    log("\n[Step 11] Testing Mentor Hub & Feedback Endpoints...")
    status, roster_data, _, lat = runner.request(
        "GET", "/api/mentor/students", extra_headers={"Authorization": f"Bearer {mentor_token}"}
    )
    runner.check(status == 200, "GET /api/mentor/students responded 200 OK", f"Latency: {lat}ms")
    roster = roster_data.get("students", []) if isinstance(roster_data, dict) else []
    runner.check(len(roster) > 0, f"Found {len(roster)} assigned students in mentor roster")

    # Submit feedback as mentor
    fb_payload = {
        "assessment_id": 1,
        "student_id": "LPU_DEMO_ADV01",
        "feedback_notes": "Live test automated feedback verification note.",
        "recommended_focus": "System Design & Mock Interviews"
    }
    status, fb_res, _, lat = runner.request(
        "POST", "/api/mentor/feedback", payload=fb_payload, extra_headers={"Authorization": f"Bearer {mentor_token}"}
    )
    runner.check(status == 200, "POST /api/mentor/feedback submitted successfully", f"Latency: {lat}ms")

    # Retrieve feedback for student
    status, fb_list, _, lat = runner.request("GET", "/api/student/feedback/LPU_DEMO_ADV01")
    runner.check(status == 200, "GET /api/student/feedback/LPU_DEMO_ADV01 responded 200 OK", f"Latency: {lat}ms")
    feedbacks = fb_list.get("feedback", []) if isinstance(fb_list, dict) else []
    runner.check(len(feedbacks) > 0, f"Found {len(feedbacks)} mentor feedback entries for student")

    # 12. Assessment Report Generation / Retrieval
    log("\n[Step 12] Testing GET /api/report/<id>...")
    # Fetch report for assessment #1 (or 404 handled gracefully if not yet created)
    status, rep_data, _, lat = runner.request("GET", "/api/report/1")
    runner.check(status in [200, 404], f"GET /api/report/1 responded HTTP {status}", f"Latency: {lat}ms")
    if status == 200 and isinstance(rep_data, dict):
        rep = rep_data.get("report", {})
        runner.check("readiness_tier" in rep, f"Report contains readiness tier: {rep.get('readiness_tier')}")

    # Final Summary Report
    elapsed = round(time.time() - runner.start_time, 2)
    log("\n" + "=" * 72)
    log(f" VERIFICATION RESULTS: {runner.passed} Passed | {runner.failed} Failed | Duration: {elapsed}s")
    log("=" * 72)

    if runner.failed > 0:
        log("\n[FAILED] SOME LIVE TESTS FAILED. Please review the errors above.\n")
        sys.exit(1)
    else:
        log("\n[SUCCESS] ALL LIVE SERVER REST ENDPOINTS VERIFIED AND RESPONDING FLAWLESSLY!\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
