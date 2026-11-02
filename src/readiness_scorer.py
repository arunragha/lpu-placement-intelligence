"""
LPU Placement Intelligence Platform - Transparent Student Readiness Scoring Service
Version: 1.0.0

Calculates a deterministic, rule-based readiness score on a 0-100 scale,
strictly separated from supervised placement outcome prediction.

Standard Configurable Weights (Initial University Rubric Baselines):
- Technical Assessment: 30%
- Aptitude & Logical Reasoning: 25%
- Communication & Soft Skills: 20%
- Project Experience & Quality: 15%
- Internship & Practical Exposure: 10%

Band Boundaries:
- Below 40: Needs attention
- 40 to below 70: Developing
- 70 to 100: Strong preparation
"""

import numpy as np

SCORING_VERSION = "2.2.0_3tier"

DEFAULT_WEIGHTS = {
    "technical": 0.30,
    "aptitude": 0.25,
    "communication": 0.20,
    "projects": 0.15,
    "internships": 0.10
}

BAND_BOUNDARIES = {
    "needs_attention": (0.0, 40.0, "Needs attention"),
    "developing": (40.0, 70.0, "Developing"),
    "strong_preparation": (70.0, 100.0, "Strong preparation")
}

def categorize_readiness_score(score: float) -> str:
    """
    Assigns canonical 3-tier readiness category based on transparent 0-100 score:
    - score < 40.0: Needs attention
    - 40.0 <= score < 70.0: Developing
    - 70.0 <= score <= 100.0: Strong preparation
    """
    s = float(score)
    if s < 40.0:
        return "Needs attention"
    elif s < 70.0:
        return "Developing"
    else:
        return "Strong preparation"

def map_legacy_tier_to_category(legacy_tier: str) -> str:
    """
    Maps legacy 4-tier model categories (Foundation, Developing, Placement Ready, Advanced)
    to the standardized canonical 3-tier readiness categories.
    """
    if not legacy_tier:
        return "Developing"
    tier_clean = str(legacy_tier).strip().lower()
    mapping = {
        "foundation": "Needs attention",
        "developing": "Developing",
        "placement ready": "Strong preparation",
        "advanced": "Strong preparation",
        # Pass-through for canonical 3-tier names
        "needs attention": "Needs attention",
        "strong preparation": "Strong preparation"
    }
    return mapping.get(tier_clean, "Developing")

def compute_readiness_score(data: dict, weights: dict = None) -> dict:
    """
    Computes a transparent, multi-pillar readiness score (0-100) from student inputs.
    
    Rules:
    - Does NOT treat missing values as zero silently; tracks missing inputs and notes them.
    - Caps project count at 4 and internship duration at 6 months to prevent infinite gaming.
    - Normalizes scores within well-defined, documented ranges.
    - Explains exact component contributions towards the total score.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()
        
    missing_fields = []
    unassessed_components = []
    notes = []
    
    # -------------------------------------------------------------
    # 1. Technical Assessment (30% weight)
    # -------------------------------------------------------------
    tech_inputs = []
    
    if data.get("dsa_score") is not None and data.get("dsa_score") != "":
        tech_inputs.append(float(data["dsa_score"]))
    else:
        missing_fields.append("dsa_score")
        
    if data.get("dsa_problems_solved") is not None and data.get("dsa_problems_solved") != "":
        # 0 to 400 problems normalized to 0-100
        dsa_count = min(float(data["dsa_problems_solved"]), 400.0)
        tech_inputs.append((dsa_count / 400.0) * 100.0)
    else:
        missing_fields.append("dsa_problems_solved")
        
    if data.get("coding_rating") is not None and data.get("coding_rating") != "":
        # Rating 900 to 2100 normalized to 0-100
        rating = float(data["coding_rating"])
        norm_rating = np.clip((rating - 900.0) / (2100.0 - 900.0) * 100.0, 0.0, 100.0)
        tech_inputs.append(norm_rating)
    else:
        missing_fields.append("coding_rating")
        
    if data.get("web_cloud_score") is not None and data.get("web_cloud_score") != "":
        tech_inputs.append(float(data["web_cloud_score"]))
        
    if data.get("system_design_score") is not None and data.get("system_design_score") != "":
        tech_inputs.append(float(data["system_design_score"]))
        
    if tech_inputs:
        tech_subscore = float(np.mean(tech_inputs))
    else:
        tech_subscore = 45.0  # Neutral baseline when unassessed
        unassessed_components.append("Technical Assessment")
        notes.append("Technical assessment inputs missing; evaluated on neutral baseline.")
        
    tech_subscore = float(np.clip(tech_subscore, 0.0, 100.0))
    tech_contrib = round(tech_subscore * weights["technical"], 2)

    # -------------------------------------------------------------
    # 2. Aptitude & Problem Solving (25% weight)
    # -------------------------------------------------------------
    apt_inputs = []
    for f in ["quantitative_aptitude", "logical_reasoning", "verbal_ability"]:
        if data.get(f) is not None and data.get(f) != "":
            apt_inputs.append(float(data[f]))
        else:
            missing_fields.append(f)
            
    if apt_inputs:
        apt_subscore = float(np.mean(apt_inputs))
    else:
        apt_subscore = 50.0
        unassessed_components.append("Aptitude")
        notes.append("Aptitude inputs missing; evaluated on neutral baseline.")
        
    apt_subscore = float(np.clip(apt_subscore, 0.0, 100.0))
    apt_contrib = round(apt_subscore * weights["aptitude"], 2)

    # -------------------------------------------------------------
    # 3. Communication & Soft Skills (20% weight)
    # -------------------------------------------------------------
    comm_inputs = []
    if data.get("mock_interview_score") is not None and data.get("mock_interview_score") != "":
        comm_inputs.append(float(data["mock_interview_score"]))
    else:
        missing_fields.append("mock_interview_score")
        
    if data.get("gd_score") is not None and data.get("gd_score") != "":
        comm_inputs.append(float(data["gd_score"]))
    else:
        missing_fields.append("gd_score")
        
    if data.get("communication_rating") is not None and data.get("communication_rating") != "":
        # 1.0 - 5.0 scaled to 0-100
        cr = float(data["communication_rating"])
        comm_inputs.append(np.clip((cr - 1.0) / 4.0 * 100.0, 0.0, 100.0))
        
    if comm_inputs:
        comm_subscore = float(np.mean(comm_inputs))
    else:
        comm_subscore = 50.0
        unassessed_components.append("Communication")
        notes.append("Communication assessment missing; evaluated on neutral baseline.")
        
    comm_subscore = float(np.clip(comm_subscore, 0.0, 100.0))
    comm_contrib = round(comm_subscore * weights["communication"], 2)

    # -------------------------------------------------------------
    # 4. Project Experience (15% weight)
    # -------------------------------------------------------------
    # Evaluates project volume (capped at 4) and project quality rubric (0-100)
    proj_cnt = float(data.get("projects_count") or 0.0)
    proj_cnt_capped = min(proj_cnt, 4.0)
    
    if data.get("github_quality") is not None and data.get("github_quality") != "":
        proj_qual = float(data["github_quality"])
    else:
        proj_qual = 50.0 if proj_cnt > 0 else 25.0
        missing_fields.append("github_quality")
        
    proj_subscore = (proj_cnt_capped / 4.0) * 40.0 + (proj_qual / 100.0) * 60.0
    proj_subscore = float(np.clip(proj_subscore, 0.0, 100.0))
    proj_contrib = round(proj_subscore * weights["projects"], 2)

    # -------------------------------------------------------------
    # 5. Internship & Practical Exposure (10% weight)
    # -------------------------------------------------------------
    # Evaluates internships (capped at 2), duration (capped at 6 mos),
    # verified certifications (capped at 3), and hackathons won (capped at 1).
    intern_cnt = min(float(data.get("internships_count") or 0.0), 2.0)
    intern_mo = min(float(data.get("internship_duration_months") or 0.0), 6.0)
    certs = min(float(data.get("certifications_count") or 0.0), 3.0)
    hack_won = min(float(data.get("hackathons_won") or 0.0), 1.0)
    
    intern_subscore = (
        (intern_cnt / 2.0) * 40.0 +
        (intern_mo / 6.0) * 30.0 +
        (certs / 3.0) * 15.0 +
        (hack_won / 1.0) * 15.0
    )
    intern_subscore = float(np.clip(intern_subscore, 0.0, 100.0))
    intern_contrib = round(intern_subscore * weights["internships"], 2)

    # -------------------------------------------------------------
    # Overall Composite Readiness Score
    # -------------------------------------------------------------
    total_score = round(tech_contrib + apt_contrib + comm_contrib + proj_contrib + intern_contrib, 1)
    total_score = float(np.clip(total_score, 0.0, 100.0))
    
    # Determine Readiness Band
    if total_score < 40.0:
        band_name = "Needs attention"
        band_code = "needs_attention"
        band_badge = "danger"
        band_desc = "Significant skill gaps identified across core technical or academic areas. Focused recovery and foundational coursework recommended."
    elif total_score < 70.0:
        band_name = "Developing"
        band_code = "developing"
        band_badge = "warning"
        band_desc = "Demonstrates competent baseline aptitude and coursework knowledge, but needs deeper problem-solving practice and portfolio project polish."
    else:
        band_name = "Strong preparation"
        band_code = "strong_preparation"
        band_badge = "success"
        band_desc = "Comprehensive preparation across algorithms, software engineering, and interview communication. Prepared for standard and competitive placement drives."

    breakdown = {
        "technical": {
            "name": "Technical Assessment",
            "weight_pct": int(weights["technical"] * 100),
            "subscore": round(tech_subscore, 1),
            "contribution": tech_contrib,
            "description": "DSA problem volume, assessment score, competitive rating, and system architecture."
        },
        "aptitude": {
            "name": "Aptitude & Problem Solving",
            "weight_pct": int(weights["aptitude"] * 100),
            "subscore": round(apt_subscore, 1),
            "contribution": apt_contrib,
            "description": "Quantitative, logical, and verbal aptitude evaluated in preliminary screening."
        },
        "communication": {
            "name": "Communication & Interviewing",
            "weight_pct": int(weights["communication"] * 100),
            "subscore": round(comm_subscore, 1),
            "contribution": comm_contrib,
            "description": "Mock technical/HR interview performance, group discussion, and spoken articulation."
        },
        "projects": {
            "name": "Project Experience & Quality",
            "weight_pct": int(weights["projects"] * 100),
            "subscore": round(proj_subscore, 1),
            "contribution": proj_contrib,
            "description": "Deployed software applications, engineering quality rubrics, and clean version control."
        },
        "internships": {
            "name": "Internships & Practical Exposure",
            "weight_pct": int(weights["internships"] * 100),
            "subscore": round(intern_subscore, 1),
            "contribution": intern_contrib,
            "description": "Industry internships, practical project duration, competitive hackathons, and certifications."
        }
    }

    return {
        "total_score": total_score,
        "readiness_category": band_name,
        "readiness_band": band_name,
        "band_code": band_code,
        "band_badge": band_badge,
        "band_description": band_desc,
        "scoring_version": SCORING_VERSION,
        "weights": {k: int(v * 100) for k, v in weights.items()},
        "breakdown": breakdown,
        "missing_fields": list(set(missing_fields)),
        "unassessed_components": unassessed_components,
        "notes": notes,
        "disclaimer": "Readiness scores are calculated transparently using university rubric weights and student self-assessment inputs, distinct from the machine learning placement outcome model."
    }

