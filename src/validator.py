"""
LPU Placement Intelligence Platform - Assessment & Batch Input Validator
Enforces server-side validation, score scales, consistency checks,
and distinguishes missing information from valid zeros.
"""

def validate_student_assessment(data: dict) -> tuple:
    """
    Validates a student profile dictionary.
    Returns: (is_valid: bool, errors: list, warnings: list)
    """
    errors = []
    warnings = []
    
    if not isinstance(data, dict):
        return False, ["Payload must be a JSON object."], []
        
    # 1. Range Validations
    numeric_ranges = {
        "cgpa": (0.0, 10.0, "CGPA must be between 0.0 and 10.0."),
        "tenth_percentage": (0.0, 100.0, "10th percentage must be between 0.0% and 100.0%."),
        "twelfth_percentage": (0.0, 100.0, "12th percentage must be between 0.0% and 100.0%."),
        "attendance_percentage": (0.0, 100.0, "Attendance percentage must be between 0.0% and 100.0%."),
        "dsa_problems_solved": (0, 3000, "DSA problems solved must be a non-negative integer <= 3000."),
        "dsa_score": (0.0, 100.0, "DSA assessment score must be between 0 and 100."),
        "coding_rating": (0, 3500, "Coding rating must be between 0 and 3500."),
        "web_cloud_score": (0.0, 100.0, "Web & Cloud score must be between 0 and 100."),
        "system_design_score": (0.0, 100.0, "System Design score must be between 0 and 100."),
        "quantitative_aptitude": (0.0, 100.0, "Quantitative Aptitude score must be between 0 and 100."),
        "logical_reasoning": (0.0, 100.0, "Logical Reasoning score must be between 0 and 100."),
        "verbal_ability": (0.0, 100.0, "Verbal Ability score must be between 0 and 100."),
        "mock_interview_score": (0.0, 100.0, "Mock Interview score must be between 0 and 100."),
        "gd_score": (0.0, 100.0, "Group Discussion score must be between 0 and 100."),
        "communication_rating": (1.0, 5.0, "Communication rating must be on the 1.0 to 5.0 rubric scale."),
        "github_quality": (0.0, 100.0, "Project quality score must be between 0 and 100."),
        "projects_count": (0, 20, "Projects count must be a non-negative integer <= 20."),
        "active_backlogs": (0, 15, "Active backlogs count must be a non-negative integer."),
        "history_backlogs": (0, 25, "History backlogs count must be a non-negative integer."),
        "internships_count": (0, 10, "Internships count must be a non-negative integer."),
        "internship_duration_months": (0, 36, "Internship duration months must be a non-negative integer."),
        "certifications_count": (0, 20, "Certifications count must be a non-negative integer."),
        "hackathons_participated": (0, 25, "Hackathons participated must be a non-negative integer."),
        "hackathons_won": (0, 25, "Hackathons won must be a non-negative integer.")
    }
    
    for key, (min_val, max_val, msg) in numeric_ranges.items():
        if key in data and data[key] is not None and data[key] != "":
            try:
                val = float(data[key])
                if val < min_val or val > max_val:
                    errors.append(f"Field '{key}': {msg} Received: {val}")
            except (ValueError, TypeError):
                errors.append(f"Field '{key}' must be numeric. Received: {data[key]}")

    # 2. Logical Consistency Checks
    try:
        hack_part = int(data.get("hackathons_participated") or 0)
        hack_won = int(data.get("hackathons_won") or 0)
        if hack_won > hack_part:
            errors.append(f"Inconsistent hackathon counts: 'hackathons_won' ({hack_won}) cannot exceed 'hackathons_participated' ({hack_part}).")
    except (ValueError, TypeError):
        pass

    try:
        intern_cnt = int(data.get("internships_count") or 0)
        intern_dur = int(data.get("internship_duration_months") or 0)
        if intern_cnt == 0 and intern_dur > 0:
            errors.append(f"Inconsistent internship records: 'internships_count' is 0, but 'internship_duration_months' is {intern_dur}. Set duration to 0 if no internships.")
    except (ValueError, TypeError):
        pass

    try:
        active_bl = int(data.get("active_backlogs") or 0)
        hist_bl = int(data.get("history_backlogs") or 0)
        if active_bl > hist_bl:
            warnings.append(f"Active backlogs ({active_bl}) exceed recorded historical backlogs ({hist_bl}). Auto-adjusting history.")
    except (ValueError, TypeError):
        pass

    # 3. Assessment Completeness / Evidence Sufficiency
    core_predictive = [
        "cgpa", "dsa_score", "dsa_problems_solved", "web_cloud_score",
        "system_design_score", "quantitative_aptitude", "mock_interview_score",
        "communication_rating", "projects_count", "github_quality"
    ]
    provided_count = sum(1 for k in core_predictive if k in data and data[k] is not None and data[k] != "")
    if provided_count < 4:
        warnings.append("Insufficient assessment information: fewer than 4 core competency fields provided. Prediction will rely on baseline population medians.")

    is_valid = len(errors) == 0
    return is_valid, errors, warnings
