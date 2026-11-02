"""
LPU Placement Intelligence Platform - Machine Learning & Explainability Engine
Version: 2.0.0

Handles model inference, Decision Tree Ensemble Path Attribution (Saabas decomposition),
skill gap benchmarking against illustrative preparation targets, institutional eligibility checking,
and personalized 30/60/90-day actionable improvement sprint roadmaps.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml_algorithms import (
    DataPreprocessor, RandomForestClassifier, RandomForestRegressor,
    GradientBoostingRegressor, MultiClassLogisticRegression, DecisionTree
)
from src.readiness_scorer import compute_readiness_score

FEATURE_DISPLAY_NAMES = {
    "cgpa": "Current CGPA (Scale 10.0)",
    "active_backlogs": "Active Standing Backlogs",
    "dsa_problems_solved": "DSA Coding Problems Solved",
    "dsa_score": "DSA Proficiency Assessment",
    "coding_rating": "Competitive Contest Rating",
    "web_cloud_score": "Web & Cloud Frameworks Proficiency",
    "system_design_score": "System Architecture & Design",
    "quantitative_aptitude": "Quantitative Aptitude Score",
    "logical_reasoning": "Logical Reasoning Score",
    "verbal_ability": "Verbal Ability & Comprehension",
    "mock_interview_score": "Mock Technical & HR Interview",
    "gd_score": "Group Discussion (GD) Score",
    "communication_rating": "Spoken Communication Rubric (1-5)",
    "projects_count": "Major Capstone Projects Count",
    "github_quality": "Project Quality & Engineering Rubric",
    "internships_count": "Completed Internships Count",
    "internship_duration_months": "Total Internship Duration (Months)",
    "certifications_count": "Verified Industry Certifications",
    "hackathons_participated": "Hackathons Participated",
    "hackathons_won": "Hackathons Won / Top 3 Finishes",
    "branch": "Engineering Discipline / Branch",
    "tenth_percentage": "10th Grade Percentage (%)",
    "twelfth_percentage": "12th Grade Percentage (%)",
    "attendance_percentage": "Academic Attendance Percentage (%)",
    "history_backlogs": "Historical Cleared Backlogs"
}

PILLAR_DEFINITIONS = {
    "Academics & Consistency": {
        "features": ["cgpa", "active_backlogs"],
        "description": "Cumulative academic performance and clean standing coursework records.",
        "icon": "academic"
    },
    "Coding & Data Structures": {
        "features": ["dsa_problems_solved", "dsa_score", "coding_rating"],
        "description": "Algorithmic problem-solving, LeetCode/CodeChef mastery, and core data structure implementations.",
        "icon": "code"
    },
    "Software & System Design": {
        "features": ["web_cloud_score", "system_design_score"],
        "description": "Full-stack development, database schema design, and microservices architecture.",
        "icon": "server"
    },
    "Aptitude & Problem Solving": {
        "features": ["quantitative_aptitude", "logical_reasoning", "verbal_ability"],
        "description": "Quantitative math, logical deduction, and verbal reasoning tested in initial screening rounds.",
        "icon": "brain"
    },
    "Communication & Interviewing": {
        "features": ["mock_interview_score", "gd_score", "communication_rating"],
        "description": "Technical communication, behavioral STAR method, GD articulation, and confidence.",
        "icon": "chat"
    },
    "Projects & Quality": {
        "features": ["projects_count", "github_quality"],
        "description": "Practical software engineering, live deployments, code readability, documentation, and testing.",
        "icon": "folder"
    },
    "Practical & Industry Exposure": {
        "features": ["internships_count", "internship_duration_months", "certifications_count", "hackathons_won"],
        "description": "Industry internships, competitive hackathon track record, and verified vendor certifications.",
        "icon": "briefcase"
    }
}

# Illustrative Preparation Targets (Customized by Target Role)
# These represent recommended preparation targets based on recruitment rubrics,
# NOT absolute hard guarantees or single-company criteria.
ILLUSTRATIVE_ROLE_TARGETS = {
    "Software Development Engineer (SDE)": {
        "description": "Target profile for Tier-1 Product & SDE recruitment drives (Algorithms, System Design, Solid Projects).",
        "targets": {
            "Academics & Consistency": 80,
            "Coding & Data Structures": 88,
            "Software & System Design": 80,
            "Aptitude & Problem Solving": 82,
            "Communication & Interviewing": 80,
            "Projects & Quality": 82,
            "Practical & Industry Exposure": 70
        }
    },
    "Full Stack Developer": {
        "description": "Target profile for Product & Fintech Web Engineering drives (Frameworks, Databases, API Integration).",
        "targets": {
            "Academics & Consistency": 75,
            "Coding & Data Structures": 75,
            "Software & System Design": 88,
            "Aptitude & Problem Solving": 76,
            "Communication & Interviewing": 78,
            "Projects & Quality": 88,
            "Practical & Industry Exposure": 72
        }
    },
    "AI / ML & Data Science Engineer": {
        "description": "Target profile for Machine Learning & Data Engineering tracks (Statistics, Python, ML Projects).",
        "targets": {
            "Academics & Consistency": 82,
            "Coding & Data Structures": 80,
            "Software & System Design": 75,
            "Aptitude & Problem Solving": 88,
            "Communication & Interviewing": 78,
            "Projects & Quality": 85,
            "Practical & Industry Exposure": 75
        }
    },
    "Cloud & DevOps Engineer": {
        "description": "Target profile for Infrastructure, Cloud Architecture, and CI/CD Engineering.",
        "targets": {
            "Academics & Consistency": 74,
            "Coding & Data Structures": 70,
            "Software & System Design": 85,
            "Aptitude & Problem Solving": 74,
            "Communication & Interviewing": 76,
            "Projects & Quality": 80,
            "Practical & Industry Exposure": 82
        }
    },
    "Data & Business Analyst": {
        "description": "Target profile for Analytics, Business Intelligence, SQL & Consulting roles.",
        "targets": {
            "Academics & Consistency": 76,
            "Coding & Data Structures": 60,
            "Software & System Design": 65,
            "Aptitude & Problem Solving": 85,
            "Communication & Interviewing": 84,
            "Projects & Quality": 74,
            "Practical & Industry Exposure": 65
        }
    },
    "QA & Automation Engineer": {
        "description": "Target profile for Quality Assurance, Test Automation, and Reliability Engineering.",
        "targets": {
            "Academics & Consistency": 72,
            "Coding & Data Structures": 65,
            "Software & System Design": 68,
            "Aptitude & Problem Solving": 72,
            "Communication & Interviewing": 74,
            "Projects & Quality": 70,
            "Practical & Industry Exposure": 60
        }
    },
    "Associate Technology Consultant": {
        "description": "Target profile for IT Services, Digital Transformation, and Technology Consulting.",
        "targets": {
            "Academics & Consistency": 70,
            "Coding & Data Structures": 58,
            "Software & System Design": 55,
            "Aptitude & Problem Solving": 72,
            "Communication & Interviewing": 78,
            "Projects & Quality": 62,
            "Practical & Industry Exposure": 50
        }
    }
}

# Standard Tier Benchmarks for reference
TIER_BENCHMARKS = {
    "Advanced": {
        "Academics & Consistency": 85,
        "Coding & Data Structures": 90,
        "Software & System Design": 82,
        "Aptitude & Problem Solving": 85,
        "Communication & Interviewing": 84,
        "Projects & Quality": 84,
        "Practical & Industry Exposure": 78
    },
    "Placement Ready": {
        "Academics & Consistency": 75,
        "Coding & Data Structures": 72,
        "Software & System Design": 68,
        "Aptitude & Problem Solving": 74,
        "Communication & Interviewing": 72,
        "Projects & Quality": 68,
        "Practical & Industry Exposure": 58
    },
    "Developing": {
        "Academics & Consistency": 65,
        "Coding & Data Structures": 52,
        "Software & System Design": 48,
        "Aptitude & Problem Solving": 62,
        "Communication & Interviewing": 60,
        "Projects & Quality": 50,
        "Practical & Industry Exposure": 38
    },
    "Foundation": {
        "Academics & Consistency": 50,
        "Coding & Data Structures": 35,
        "Software & System Design": 30,
        "Aptitude & Problem Solving": 45,
        "Communication & Interviewing": 45,
        "Projects & Quality": 30,
        "Practical & Industry Exposure": 20
    }
}

PRESET_STUDENTS = [
    {
        "preset_id": "advanced_candidate",
        "preset_name": "🌟 Advanced Profile (High DSA & Full Stack)",
        "description": "Demonstrates strong DSA (395 solved, 1780 rating), 8.92 CGPA, 2 internships, 88 mock interview score. Target: SDE.",
        "data": {
            "student_id": "LPU_DEMO_ADV01",
            "branch": "Computer Science & Engineering",
            "target_role": "Software Development Engineer (SDE)",
            "weekly_prep_hours": "15-20 hrs/week",
            "cgpa": 8.92,
            "active_backlogs": 0,
            "tenth_percentage": 92.5,
            "twelfth_percentage": 90.0,
            "attendance_percentage": 91.0,
            "history_backlogs": 0,
            "dsa_problems_solved": 395,
            "dsa_score": 92.0,
            "coding_rating": 1780,
            "web_cloud_score": 86.0,
            "system_design_score": 82.0,
            "quantitative_aptitude": 88.0,
            "logical_reasoning": 90.0,
            "verbal_ability": 86.0,
            "mock_interview_score": 88.0,
            "gd_score": 85.0,
            "communication_rating": 4.5,
            "projects_count": 4,
            "github_quality": 88.0,
            "internships_count": 2,
            "internship_duration_months": 5,
            "certifications_count": 3,
            "hackathons_participated": 4,
            "hackathons_won": 1
        }
    },
    {
        "preset_id": "placement_ready_tech",
        "preset_name": "🚀 Placement Ready (Solid Tech, Needs Mock Interview)",
        "description": "Competent coder (310 DSA, 7.85 CGPA), strong web dev, but interview score is 62 and communication needs refinement.",
        "data": {
            "student_id": "LPU_DEMO_RDY02",
            "branch": "Computer Science & Engineering",
            "target_role": "Full Stack Developer",
            "weekly_prep_hours": "10-15 hrs/week",
            "cgpa": 7.85,
            "active_backlogs": 0,
            "tenth_percentage": 82.0,
            "twelfth_percentage": 78.5,
            "attendance_percentage": 82.0,
            "history_backlogs": 0,
            "dsa_problems_solved": 310,
            "dsa_score": 78.0,
            "coding_rating": 1580,
            "web_cloud_score": 80.0,
            "system_design_score": 68.0,
            "quantitative_aptitude": 80.0,
            "logical_reasoning": 82.0,
            "verbal_ability": 68.0,
            "mock_interview_score": 62.0,
            "gd_score": 65.0,
            "communication_rating": 3.4,
            "projects_count": 3,
            "github_quality": 75.0,
            "internships_count": 1,
            "internship_duration_months": 3,
            "certifications_count": 1,
            "hackathons_participated": 2,
            "hackathons_won": 0
        }
    },
    {
        "preset_id": "developing_acad_focused",
        "preset_name": "📈 Developing Profile (High CGPA, Needs Practical DSA & Projects)",
        "description": "9.15 CGPA and good aptitude, but only 45 DSA problems solved, 1 project, and 0 internships.",
        "data": {
            "student_id": "LPU_DEMO_DEV03",
            "branch": "Information Technology",
            "target_role": "AI / ML & Data Science Engineer",
            "weekly_prep_hours": "10-15 hrs/week",
            "cgpa": 9.15,
            "active_backlogs": 0,
            "tenth_percentage": 94.0,
            "twelfth_percentage": 92.0,
            "attendance_percentage": 95.0,
            "history_backlogs": 0,
            "dsa_problems_solved": 45,
            "dsa_score": 42.0,
            "coding_rating": 1050,
            "web_cloud_score": 42.0,
            "system_design_score": 35.0,
            "quantitative_aptitude": 86.0,
            "logical_reasoning": 82.0,
            "verbal_ability": 84.0,
            "mock_interview_score": 62.0,
            "gd_score": 65.0,
            "communication_rating": 3.8,
            "projects_count": 1,
            "github_quality": 40.0,
            "internships_count": 0,
            "internship_duration_months": 0,
            "certifications_count": 1,
            "hackathons_participated": 0,
            "hackathons_won": 0
        }
    },
    {
        "preset_id": "foundation_at_risk",
        "preset_name": "⚠️ Foundation Needed (Active Backlogs & Core Skill Deficits)",
        "description": "6.12 CGPA, 2 active backlogs, 25 DSA solved, low aptitude scores. Needs foundational coursework focus.",
        "data": {
            "student_id": "LPU_DEMO_FND04",
            "branch": "Electronics & Communication",
            "target_role": "Associate Technology Consultant",
            "weekly_prep_hours": "5-10 hrs/week",
            "cgpa": 6.12,
            "active_backlogs": 2,
            "tenth_percentage": 64.0,
            "twelfth_percentage": 61.5,
            "attendance_percentage": 68.0,
            "history_backlogs": 3,
            "dsa_problems_solved": 25,
            "dsa_score": 32.0,
            "coding_rating": 940,
            "web_cloud_score": 34.0,
            "system_design_score": 22.0,
            "quantitative_aptitude": 44.0,
            "logical_reasoning": 48.0,
            "verbal_ability": 42.0,
            "mock_interview_score": 38.0,
            "gd_score": 40.0,
            "communication_rating": 2.0,
            "projects_count": 1,
            "github_quality": 25.0,
            "internships_count": 0,
            "internship_duration_months": 0,
            "certifications_count": 0,
            "hackathons_participated": 0,
            "hackathons_won": 0
        }
    }
]


class PlacementIntelligenceEngine:
    """Core ML inference, tree attribution, eligibility and roadmap engine."""
    
    def __init__(self, models_dir=None):
        if models_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base_dir, "models")
            
        self.models_dir = models_dir
        self.load_models()
        
    def load_models(self):
        with open(os.path.join(self.models_dir, "preprocessor.pkl"), "rb") as f:
            self.preprocessor = pickle.load(f)
            
        with open(os.path.join(self.models_dir, "readiness_classifier.pkl"), "rb") as f:
            self.readiness_clf = pickle.load(f)
            
        placement_clf_path = os.path.join(self.models_dir, "placement_classifier.pkl")
        if os.path.exists(placement_clf_path):
            with open(placement_clf_path, "rb") as f:
                self.placement_clf = pickle.load(f)
        else:
            self.placement_clf = None
            
        with open(os.path.join(self.models_dir, "ctc_regressor.pkl"), "rb") as f:
            self.ctc_reg = pickle.load(f)
            
        with open(os.path.join(self.models_dir, "prob_regressor.pkl"), "rb") as f:
            self.prob_reg = pickle.load(f)
            
        with open(os.path.join(self.models_dir, "role_classifier.pkl"), "rb") as f:
            self.role_clf = pickle.load(f)
            
        with open(os.path.join(self.models_dir, "model_metrics.json"), "r") as f:
            self.metadata = json.load(f)
            
        # Enrich metadata with sorted feature importances and method explanations
        raw_feat_imp = self.metadata.get("feature_importances", {})
        sorted_feats = []
        for feat, imp in sorted(raw_feat_imp.items(), key=lambda x: x[1], reverse=True):
            display_name = FEATURE_DISPLAY_NAMES.get(feat, feat)
            if feat.startswith("branch_"):
                display_name = f"Discipline: {feat.replace('branch_', '')}"
            sorted_feats.append({
                "feature": feat,
                "display_name": display_name,
                "importance_weight": round(float(imp), 4),
                "importance_pct": round(float(imp) * 100.0, 2),
                "relative_pct": round(float(imp) * 100.0, 2)
            })
            
        self.metadata["sorted_feature_importances"] = sorted_feats
        self.model_metadata = self.metadata
        self.metadata["feature_importance_method"] = "Impurity-based Gini Importance (Random Forest Ensemble)"
        self.metadata["feature_importance_axis_units"] = "Relative Importance (%)"
        self.metadata["feature_importances_explanation"] = self.metadata.get("active_backlogs_investigation", "active_backlogs (active standing backlogs) acts as an immediate threshold and gatekeeper.")
        self.metadata["feature_importance_limitations"] = (
            "Gini importance measures the total decrease in node impurity brought by a feature across all trees in the ensemble. "
            "It can favor high-cardinality continuous features, reflects global training associations (not individual student explanations), "
            "and does not imply a positive or negative directional relationship."
        )
        self.metadata["active_backlogs_investigation"] = (
            "Active standing backlogs ranks highest (~12.2% global importance) because it acts as an immediate threshold and severe penalty "
            "in university recruitment drive clearance and synthetic tier labeling. Decision tree algorithms prioritize it as an early, "
            "high-purity split to partition unplaced students near the tree root."
        )

        self.tier_classes = self.metadata["tier_classes"]
        self.role_classes = self.metadata["role_classes"]
        self.numeric_features = self.metadata["numeric_features"]
        self.categorical_features = self.metadata["categorical_features"]
        self.transformed_feature_names = self.metadata["transformed_feature_names"]
        
    def check_placement_eligibility(self, student_dict: dict) -> dict:
        """
        Evaluates standard institutional and recruitment eligibility criteria.
        Kept strictly distinct from the ML skill readiness model.
        """
        cgpa = float(student_dict.get("cgpa", 0.0))
        active_backlogs = int(student_dict.get("active_backlogs", 0))
        attendance = float(student_dict.get("attendance_percentage", 75.0))
        tenth = float(student_dict.get("tenth_percentage", 70.0))
        twelfth = float(student_dict.get("twelfth_percentage", 70.0))
        
        reasons_unmet = []
        tier1_eligible = True
        general_eligible = True
        
        if active_backlogs > 0:
            reasons_unmet.append(f"{active_backlogs} active backlog(s) — most Tier-1 & Product drives require 0 standing backlogs.")
            tier1_eligible = False
            if active_backlogs > 1:
                general_eligible = False
                reasons_unmet.append("More than 1 active backlog disqualifies from standard campus drives.")
                
        if cgpa < 6.5:
            reasons_unmet.append(f"CGPA {cgpa:.2f} is below the 6.50 minimum threshold for standard campus drives.")
            general_eligible = False
            tier1_eligible = False
        elif cgpa < 7.5:
            tier1_eligible = False
            reasons_unmet.append(f"CGPA {cgpa:.2f} is below the 7.50 cutoff typical for Tier-1 / Product drives.")
            
        if attendance < 75.0:
            reasons_unmet.append(f"Attendance ({attendance:.1f}%) is below the mandatory 75% institutional requirement.")
            general_eligible = False
            tier1_eligible = False
            
        if tenth < 60.0 or twelfth < 60.0:
            reasons_unmet.append("10th/12th grade marks below 60% may trigger company-specific eligibility filters.")
            
        status = "Fully Eligible" if tier1_eligible else "Partially Eligible" if general_eligible else "Eligibility Action Needed"
        drive_access_pct = 95 if tier1_eligible else 65 if general_eligible else 25
        
        return {
            "status": status,
            "estimated_drive_clearance_pct": drive_access_pct,
            "tier1_eligible": tier1_eligible,
            "general_eligible": general_eligible,
            "unmet_conditions": reasons_unmet,
            "notes": "Eligibility filters are set by recruiting organizations and university placement cell guidelines, separate from algorithmic skill readiness."
        }

    def compute_assessment_completeness(self, student_dict: dict) -> dict:
        """Checks completeness of student inputs and flags missing evidence."""
        core_fields = self.numeric_features
        present_count = 0
        missing_fields = []
        
        for f in core_fields:
            val = student_dict.get(f)
            if val is not None and val != "":
                present_count += 1
            else:
                missing_fields.append(f)
                
        completeness_pct = round((present_count / len(core_fields)) * 100, 1)
        is_sufficient = completeness_pct >= 50.0
        
        return {
            "completeness_pct": completeness_pct,
            "is_sufficient": is_sufficient,
            "assessed_fields_count": present_count,
            "total_core_fields": len(core_fields),
            "missing_fields": [FEATURE_DISPLAY_NAMES.get(m, m) for m in missing_fields[:5]]
        }

    def predict_student(self, student_dict: dict) -> dict:
        """
        Executes multi-class readiness inference, tree path feature attribution,
        skill gap benchmarking, institutional eligibility check, and personalized roadmap synthesis.
        """
        completeness = self.compute_assessment_completeness(student_dict)
        eligibility = self.check_placement_eligibility(student_dict)
        
        # Prepare DataFrame row matching preprocessor feature expectations
        all_cols = self.numeric_features + self.categorical_features
        row_data = {}
        for col in all_cols:
            if col in student_dict and student_dict[col] is not None and student_dict[col] != "":
                row_data[col] = [student_dict[col]]
            else:
                # Use training median/mode for unassessed features to prevent arbitrary zeros
                if col in self.metadata.get("feature_stats", {}):
                    row_data[col] = [self.metadata["feature_stats"][col]["median"]]
                else:
                    row_data[col] = ["Computer Science & Engineering"]
                    
        df_input = pd.DataFrame(row_data)
        X_trans = self.preprocessor.transform(df_input)
        
        # 1. Supervised Readiness Category Prediction (Primary ML output)
        tier_pred_idx = int(self.readiness_clf.predict(X_trans)[0])
        predicted_tier = self.tier_classes[tier_pred_idx] if tier_pred_idx < len(self.tier_classes) else self.tier_classes[0]
        
        tier_probs_raw = self.readiness_clf.predict_proba(X_trans)[0]
        tier_probs = {}
        for idx, cls_name in enumerate(self.tier_classes):
            prob_val = tier_probs_raw[idx] if idx < len(tier_probs_raw) else 0.0
            tier_probs[cls_name] = round(float(prob_val) * 100, 1)
            
        # Category meaning and plain-language explanation
        tier_descriptions = {
            "Advanced": "Exemplary technical, algorithmic, and communicative profile. Prepared for competitive Tier-1 and high-bracket software engineering hiring drives.",
            "Placement Ready": "Solid competence across core CS fundamentals, problem solving, projects, and interview readiness. Well-positioned for standard product, fintech, and IT consulting placement drives.",
            "Developing": "Demonstrates basic technical literacy and academic progress, but exhibits notable gaps in algorithmic problem solving, practical project depth, or interview articulation. Structured upskilling recommended.",
            "Foundation": "Significant foundational gaps identified in core programming, technical coursework, or academic backlogs. Immediate priority is academic recovery, clearing backlogs, and mastering basic CS fundamentals."
        }
        
        # 2. Transparent Multi-Pillar Student Readiness Scoring Service
        readiness_assessment = compute_readiness_score(student_dict)

        # 3. Supervised Binary Placement Prediction (Placed / Unplaced)
        if self.placement_clf is not None:
            placed_binary_pred = int(self.placement_clf.predict(X_trans)[0])
            if hasattr(self.placement_clf, "predict_proba"):
                probs = self.placement_clf.predict_proba(X_trans)[0]
                supervised_prob = round(float(probs[1]) * 100.0, 1) if len(probs) > 1 else (100.0 if placed_binary_pred == 1 else 0.0)
            else:
                supervised_prob = 100.0 if placed_binary_pred == 1 else 0.0
            supervised_status = "Available (Supervised Model v2.1)"
            placement_outcome_text = "Likely Placed" if placed_binary_pred == 1 else "Placement Support Needed"
        else:
            placed_binary_pred = None
            supervised_prob = None
            supervised_status = "Unavailable — Supervised placement model not loaded"
            placement_outcome_text = "Unavailable"

        # 4. Experimental Simulation Outputs (Clearly labeled as synthetic simulation)
        placement_prob = float(self.prob_reg.predict(X_trans)[0])
        placement_prob = round(float(np.clip(placement_prob, 5.0, 99.5)), 1)
        
        expected_ctc = float(self.ctc_reg.predict(X_trans)[0])
        expected_ctc = round(float(np.clip(expected_ctc, 2.8, 42.0)), 2)
        ctc_min = round(max(2.5, expected_ctc * 0.85), 1)
        ctc_max = round(expected_ctc * 1.20, 1)
        
        # 5. Target Role Match & Recommendations
        target_role = student_dict.get("target_role") or "Software Development Engineer (SDE)"
        role_pred_idx = int(self.role_clf.predict(X_trans)[0])
        recommended_role = self.role_classes[role_pred_idx] if role_pred_idx < len(self.role_classes) else self.role_classes[0]
        
        role_probs_raw = self.role_clf.predict_proba(X_trans)[0]
        top_roles = []
        for idx, cls_name in enumerate(self.role_classes):
            fit_val = role_probs_raw[idx] if idx < len(role_probs_raw) else 0.0
            top_roles.append({
                "role": cls_name,
                "fit_score": round(float(fit_val) * 100, 1),
                "is_student_target": (cls_name == target_role)
            })
        top_roles.sort(key=lambda x: x["fit_score"], reverse=True)
        
        # 6. Decision Tree Ensemble Path Attribution (Relative to predicted class)
        attribution_analysis = self._compute_tree_path_attributions(X_trans, tier_pred_idx, student_dict, predicted_tier)
        
        # 7. Competency Radar & Gap Analysis against student's chosen Target Role
        radar_scores, gap_analysis = self._compute_skill_gaps(student_dict, target_role, predicted_tier)
        
        # 8. Actionable 30/60/90-Day Improvement Sprint Roadmap
        weekly_hours = student_dict.get("weekly_prep_hours") or "10-15 hrs/week"
        roadmap = self._generate_prescriptive_roadmap(student_dict, gap_analysis, predicted_tier, target_role, weekly_hours)
        
        # Canonical 3-tier readiness category
        canonical_category = readiness_assessment.get("readiness_category") or readiness_assessment.get("readiness_band")
        if not canonical_category or canonical_category not in ["Needs attention", "Developing", "Strong preparation"]:
            if predicted_tier == "Foundation":
                canonical_category = "Needs attention"
            elif predicted_tier == "Developing":
                canonical_category = "Developing"
            else:
                canonical_category = "Strong preparation"

        # Honest Demo Model Estimate calculation (integer display precision, full precision preserved)
        raw_model_prob = supervised_prob if supervised_prob is not None else placement_prob
        demo_model_estimate_pct = int(round(raw_model_prob))

        completeness_info = {
            "supplied_fields_count": completeness.get("supplied_fields_count", 0),
            "total_fields_count": completeness.get("total_fields_count", 0),
            "completeness_pct": round(float(completeness.get("completeness_pct", 100.0)), 1),
            "input_completeness_pct": int(round(float(completeness.get("completeness_pct", 100.0)))),
            "input_completeness_display": f"{int(round(float(completeness.get('completeness_pct', 100.0))))}%",
            "explanation": "Percentage of required assessment fields supplied. This is not model confidence or placement likelihood.",
            "missing_optional_fields": completeness.get("missing_optional_fields", [])
        }

        return {
            "prediction": {
                "readiness_category": canonical_category,
                "readiness_band": canonical_category,
                "readiness_tier": canonical_category,
                "legacy_readiness_tier": predicted_tier,
                "tier_description": tier_descriptions.get(predicted_tier, ""),
                "tier_probabilities": tier_probs,
                "target_role": target_role,
                "recommended_role": recommended_role,
                "top_roles": top_roles[:4],
                # Honest Demo Model Estimate
                "demo_model_estimate_pct": demo_model_estimate_pct,
                "demo_model_estimate_display": f"{demo_model_estimate_pct}%",
                "demo_model_label": "Demo model estimate",
                "demo_model_disclaimer": "Generated by a model trained on synthetic profiles. Not validated for real student placement outcomes.",
                "demo_model_estimate_notice": "Generated by a model trained on synthetic profiles. Not validated for real student placement outcomes.",
                # Supervised Placement Classification
                "supervised_placement_outcome": placement_outcome_text,
                "supervised_placement_pred": placed_binary_pred,
                "supervised_placement_prob": supervised_prob,
                "model_availability_status": supervised_status,
                # Backwards compatible aliases
                "simulated_placement_prob": placement_prob,
                "raw_placement_prob": raw_model_prob,
                # Secondary experimental simulation section
                "experimental_simulated_ctc": {
                    "label": "Experimental Simulated Compensation",
                    "expected_ctc_lpa": expected_ctc,
                    "ctc_bracket": f"{ctc_min} - {ctc_max} LPA",
                    "status": "Experimental — Synthetic demonstration only",
                    "disclaimer": "Salary projection derived from synthetic demonstration model; purely illustrative."
                },
                "simulated_expected_ctc_lpa": expected_ctc,
                "simulated_ctc_bracket": f"{ctc_min} - {ctc_max} LPA",
                "simulation_disclaimer": "Salary bracket and demo estimates are generated by a model trained on synthetic profiles, not validated for real student placement outcomes."
            },
            "readiness_assessment": readiness_assessment,
            "completeness": completeness_info,
            "eligibility": eligibility,
            "explainability": attribution_analysis,
            "radar_scores": radar_scores,
            "gap_analysis": gap_analysis,
            "roadmap": roadmap,
            "model_version": "2.1.0",
            "scoring_version": "2.2.0_3tier"
        }

    def _compute_tree_path_attributions(self, X_trans, tier_pred_idx, student_dict, predicted_tier) -> dict:
        """
        Computes Decision Tree Ensemble Path Attributions (Saabas Decomposition)
        relative to the predicted readiness class.
        
        If predicted class is Foundation / Developing:
          - Features with positive contribution towards Foundation are Deficit Drivers / Risk Factors.
        If predicted class is Placement Ready / Advanced:
          - Features with positive contribution towards readiness are Key Driving Catalysts.
        """
        x_row = X_trans[0]
        attributions = []
        
        try:
            # Tree Path Feature Attribution
            raw_tree_attr = self.readiness_clf.compute_tree_path_attribution(x_row)
            if tier_pred_idx < raw_tree_attr.shape[0]:
                class_attr = raw_tree_attr[tier_pred_idx]
            else:
                class_attr = raw_tree_attr[0]
                
            for idx, feat_name in enumerate(self.transformed_feature_names):
                val = float(class_attr[idx])
                raw_val = student_dict.get(feat_name, None)
                
                display_name = FEATURE_DISPLAY_NAMES.get(feat_name, feat_name)
                if feat_name.startswith("branch_"):
                    display_name = f"Discipline: {feat_name.replace('branch_', '')}"
                    
                val = round(val, 4)
                
                attributions.append({
                    "feature": feat_name,
                    "display_name": display_name,
                    "raw_value": raw_val if raw_val is not None else "Transformed",
                    "path_contribution": val
                })
        except Exception as e:
            print(f"Attribution error: {e}")
            
        # Sort by absolute impact
        attributions.sort(key=lambda x: abs(x["path_contribution"]), reverse=True)
        
        is_at_risk_tier = predicted_tier in ["Foundation", "Developing"]
        
        # Categorize appropriately relative to the predicted class
        if is_at_risk_tier:
            # Pushing towards Foundation/Developing = Deficit Factors
            primary_drivers = [a for a in attributions if a["path_contribution"] > 0][:5]
            secondary_mitigators = [a for a in attributions if a["path_contribution"] < 0][:4]
            driver_label = f"Key Factors Contributing to '{predicted_tier}' Classification (Risk Drivers / Skill Deficits):"
            mitigator_label = f"Compensating Strengths Reducing '{predicted_tier}' Probability:"
        else:
            # Pushing towards Placement Ready/Advanced = Positive Catalysts
            primary_drivers = [a for a in attributions if a["path_contribution"] > 0][:5]
            secondary_mitigators = [a for a in attributions if a["path_contribution"] < 0][:4]
            driver_label = f"Key Strengths Elevating Profile into '{predicted_tier}':"
            mitigator_label = f"Constraining Gaps Restricting Further Tier Progression:"
            
        return {
            "method": "Decision Tree Ensemble Path Attribution (Saabas Tree Path Decomposition)",
            "predicted_class": predicted_tier,
            "driver_label": driver_label,
            "mitigator_label": mitigator_label,
            "is_at_risk_tier": is_at_risk_tier,
            "primary_drivers": primary_drivers,
            "secondary_mitigators": secondary_mitigators,
            "method_explanation": "Attributions quantify the net change in class probability introduced by decision node splits on each feature along the instance's path across all trees in the ensemble. Associations represent statistical model behavior and do not constitute absolute guarantees of placement."
        }

    def _compute_skill_gaps(self, student_dict: dict, target_role: str, predicted_tier: str) -> tuple:
        """
        Calculates normalized competency scores (0-100) across the 7 pillars
        and computes deltas against the student's chosen Target Role illustrative targets.
        """
        # Selected Illustrative Targets based on chosen role
        role_target_spec = ILLUSTRATIVE_ROLE_TARGETS.get(target_role, ILLUSTRATIVE_ROLE_TARGETS["Software Development Engineer (SDE)"])
        benchmark_targets = role_target_spec["targets"]
        
        # 1. Compute student's scores on 0-100 scale
        cgpa = float(student_dict.get("cgpa", 7.0))
        active_bl = int(student_dict.get("active_backlogs", 0))
        acad_score = np.clip(((cgpa - 5.0) / 5.0) * 100 - (active_bl * 25), 0, 100)
        
        dsa_solved = int(student_dict.get("dsa_problems_solved", 100))
        dsa_test = float(student_dict.get("dsa_score", 50.0))
        rating = float(student_dict.get("coding_rating", 1200))
        dsa_score = np.clip(
            (min(dsa_solved, 450) / 450.0) * 35 +
            (dsa_test / 100.0) * 45 +
            (max(0, rating - 1000) / 1000.0) * 20,
            0, 100
        )
        
        wc = float(student_dict.get("web_cloud_score", 50.0))
        sd = float(student_dict.get("system_design_score", 45.0))
        system_score = np.clip(wc * 0.55 + sd * 0.45, 0, 100)
        
        q = float(student_dict.get("quantitative_aptitude", 60.0))
        l = float(student_dict.get("logical_reasoning", 60.0))
        v = float(student_dict.get("verbal_ability", 60.0))
        apt_score = np.clip((q + l + v) / 3.0, 0, 100)
        
        mock = float(student_dict.get("mock_interview_score", 55.0))
        gd = float(student_dict.get("gd_score", 55.0))
        comm_r = float(student_dict.get("communication_rating", 3.0))
        soft_score = np.clip(mock * 0.50 + gd * 0.30 + (comm_r / 5.0) * 20.0, 0, 100)
        
        proj_cnt = int(student_dict.get("projects_count", 2))
        proj_qual = float(student_dict.get("github_quality", 50.0))
        proj_score = np.clip((min(proj_cnt, 4) / 4.0) * 40 + (proj_qual / 100.0) * 60, 0, 100)
        
        intern_cnt = int(student_dict.get("internships_count", 0))
        intern_mo = int(student_dict.get("internship_duration_months", 0))
        certs = int(student_dict.get("certifications_count", 1))
        hack_won = int(student_dict.get("hackathons_won", 0))
        exp_score = np.clip(
            (min(intern_cnt, 2) / 2.0) * 40 +
            (min(intern_mo, 6) / 6.0) * 25 +
            (min(certs, 3) / 3.0) * 20 +
            (min(hack_won, 1) / 1.0) * 15,
            0, 100
        )
        
        student_pillar_scores = {
            "Academics & Consistency": round(float(acad_score), 1),
            "Coding & Data Structures": round(float(dsa_score), 1),
            "Software & System Design": round(float(system_score), 1),
            "Aptitude & Problem Solving": round(float(apt_score), 1),
            "Communication & Interviewing": round(float(soft_score), 1),
            "Projects & Quality": round(float(proj_score), 1),
            "Practical & Industry Exposure": round(float(exp_score), 1)
        }
        
        # Calculate gaps: Gap = student score - benchmark
        breakdown = []
        critical_gaps = []
        demonstrated_strengths = []
        
        for pillar, s_score in student_pillar_scores.items():
            t_score = benchmark_targets.get(pillar, 70)
            delta = round(s_score - t_score, 1)
            
            # Rigorous tolerance logic (Documented benchmark margin: ±3.0 points)
            if delta < -15.0:
                status = "Below benchmark"
                status_detail = "Critical gap (< -15 pts)"
                badge = "danger"
                critical_gaps.append(pillar)
            elif delta < -3.0:
                status = "Below benchmark"
                status_detail = "Moderate deficit (-3 to -15 pts)"
                badge = "warning"
                critical_gaps.append(pillar)
            elif delta < 0.0:
                status = "Within benchmark tolerance"
                status_detail = "Within ±3 pt tolerance margin"
                badge = "info"
            elif delta == 0.0:
                status = "Meets benchmark"
                status_detail = "Exactly meets role benchmark"
                badge = "success"
                demonstrated_strengths.append(pillar)
            elif delta <= 8.0:
                status = "Meets benchmark"
                status_detail = "Meets role benchmark (+1 to +8 pts)"
                badge = "success"
                demonstrated_strengths.append(pillar)
            else:
                status = "Above benchmark"
                status_detail = "Exceeds role benchmark (> +8 pts)"
                badge = "champion"
                demonstrated_strengths.append(pillar)
                
            breakdown.append({
                "competency": pillar,
                "your_score": s_score,
                "role_benchmark": t_score,
                "gap": delta,
                "status": status,
                "status_detail": status_detail,
                "badge": badge,
                # Backwards compatible aliases
                "pillar": pillar,
                "student_score": s_score,
                "target_score": t_score,
                "delta": delta
            })
            
        radar_data = {
            "labels": list(student_pillar_scores.keys()),
            "student_scores": list(student_pillar_scores.values()),
            "target_role_name": target_role,
            "target_role_scores": [benchmark_targets[k] for k in student_pillar_scores.keys()],
            "target_role_description": role_target_spec["description"],
            "benchmark_label": "Illustrative Role Benchmark"
        }
        
        gap_data = {
            "target_role": target_role,
            "table_headers": ["Competency", "Your score", "Role benchmark", "Gap", "Status"],
            "tolerance_rule": "Gaps within -3.0 to 0.0 are within benchmark tolerance. Gaps below -3.0 are below benchmark.",
            "benchmark_nature": "Illustrative Role Benchmark (Educational reference based on standard recruitment profiles; not an institutional mandate).",
            "critical_gap_count": len(critical_gaps),
            "pillars_breakdown": breakdown,
            "top_strengths": demonstrated_strengths[:3],
            "top_improvement_priorities": critical_gaps
        }
        
        return radar_data, gap_data

    def _generate_prescriptive_roadmap(self, student_dict: dict, gap_analysis: dict, predicted_tier: str, target_role: str, weekly_hours: str) -> list:
        """
        Synthesizes a structured 30-, 60-, 90-day actionable improvement plan.
        Pacing and milestones adjust based on student's target role, primary gap pillars,
        and weekly time commitment.
        """
        breakdown = gap_analysis["pillars_breakdown"]
        # Find weakest pillars
        sorted_gaps = sorted(breakdown, key=lambda x: x["delta"])
        weakest_1 = sorted_gaps[0]["pillar"]
        weakest_2 = sorted_gaps[1]["pillar"] if len(sorted_gaps) > 1 else weakest_1
        
        # Pacing context
        hours_note = f"Paced for {weekly_hours} commitment."
        
        # Phase 1: Days 1-30 (Core Foundation & High-Priority Gap Diagnosis)
        phase1_tasks = []
        if "Coding & Data Structures" in [weakest_1, weakest_2]:
            phase1_tasks.extend([
                "Review time and space complexity analysis (Big-O notation) and foundational data structures (Arrays, Strings, Hash Maps, Two-Pointers).",
                "Solve 25 targeted LeetCode Easy/Medium problems focusing on sliding window and prefix sums."
            ])
        elif "Software & System Design" in [weakest_1, weakest_2]:
            phase1_tasks.extend([
                f"Solidify fundamentals for {target_role}: revise relational database normalization and SQL queries (JOINs, aggregation).",
                "Set up local development workflow with Git, clean commit conventions, and environment variables."
            ])
        else:
            phase1_tasks.extend([
                "Consolidate core subject notes: Operating Systems (processes, threads, deadlocks) and DBMS transactions (ACID properties).",
                "Complete weekly quantitative aptitude timed drill (20 questions: Percentages, Profit/Loss, Time & Work)."
            ])
            
        phase1_tasks.append("Conduct an initial baseline mock technical test and record self-introduction video for speech clarity review.")
        
        # Phase 2: Days 31-60 (Applied Mastery & Portfolio Development)
        phase2_tasks = []
        if "Projects & Quality" in [weakest_1, weakest_2] or "Software & System Design" in [weakest_1, weakest_2]:
            phase2_tasks.extend([
                f"Build and deploy a full-stack capstone project aligned with {target_role} (RESTful API, database persistence, authentication, Docker containerization).",
                "Implement unit and integration tests covering at least 60% of critical business logic; write a comprehensive README with architecture diagrams."
            ])
        else:
            phase2_tasks.extend([
                "Advance to Tree, Graph (BFS/DFS), and Dynamic Programming patterns; solve 30 medium-tier algorithmic challenges.",
                "Participate in weekly timed coding contests (LeetCode Biweekly or CodeChef) to build contest composure."
            ])
            
        phase2_tasks.append("Practice 2 peer mock interviews utilizing the STAR technique for behavioral questions.")
        
        # Phase 3: Days 61-90 (Recruitment Simulation & Polish)
        phase3_tasks = [
            f"Tailor resume to {target_role}: highlight measurable project outcomes, technologies used, and deployment URLs.",
            "Complete 3 full-length timed placement assessment simulations (Aptitude + Technical MCQs + 2 Coding Problems).",
            "Engage in mentor-reviewed mock technical and HR interview rounds; refine vocal modulation and structured answers.",
            "Review company-specific recruitment archives and high-frequency problem sets for upcoming campus drives."
        ]
        
        roadmap = [
            {
                "phase": "Phase 1: Days 1 to 30",
                "title": "Foundational Remediation & Core Skill Consolidation",
                "focus": f"Primary Focus: {weakest_1} ({hours_note})",
                "milestones": phase1_tasks,
                "target_deliverable": "Verified mastery of core problem patterns and documented baseline assessment."
            },
            {
                "phase": "Phase 2: Days 31 to 60",
                "title": "Applied Project Engineering & Algorithmic Depth",
                "focus": f"Primary Focus: {weakest_2} & Capstone Implementation",
                "milestones": phase2_tasks,
                "target_deliverable": "Functional, well-documented capstone project on GitHub with unit tests and live deployment."
            },
            {
                "phase": "Phase 3: Days 61 to 90",
                "title": "Recruitment Simulation, STAR Interviewing & Final Polish",
                "focus": "Full Campus Selection Simulation & Behavioral Communication",
                "milestones": phase3_tasks,
                "target_deliverable": "Polished resume, portfolio deployment, and 3 completed mentor mock interviews."
            }
        ]
        
        return roadmap
