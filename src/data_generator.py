"""
LPU Placement Intelligence Platform - Synthetic Dataset Generator
Generates realistic university engineering student profiles with correlated academic,
technical, aptitude, soft-skills, and experiential attributes.

IMPORTANT NOTICE - DEMO SYNTHETIC DATA:
This dataset is synthesized using statistical distributions, correlations, and heuristic formulas
strictly for demonstrating supervised machine learning pipelines, model comparison, explainability,
and mentor analytics. Evaluation metrics on this data reflect agreement with the synthetic rubric,
NOT real placement probabilities or actual recruitment outcomes of LPU students.
"""

import os
import argparse
import numpy as np
import pandas as pd

# Supported Target Classes
READINESS_CATEGORIES = [
    "Foundation",
    "Developing",
    "Placement Ready",
    "Advanced"
]

TARGET_ROLES = [
    "Software Development Engineer (SDE)",
    "Full Stack Developer",
    "AI / ML & Data Science Engineer",
    "Cloud & DevOps Engineer",
    "Data & Business Analyst",
    "QA & Automation Engineer",
    "Associate Technology Consultant"
]

SUPPORTED_BRANCHES = [
    "Computer Science & Engineering",
    "Information Technology",
    "AI & Data Engineering",
    "Electronics & Communication",
    "Mechanical & Mechatronics"
]

def generate_placement_dataset(n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """
    Generates synthetic engineering student dataset with correlated academic, technical,
    aptitude, and practical experience features.
    
    NOTE ON FAIRNESS & PRIVACY:
    Gender and personal demographic identifiers are strictly excluded from predictive features
    in adherence to algorithmic fairness and institutional recruitment ethics.
    """
    np.random.seed(random_state)
    
    # 1. Academic Program & Department (Initially CSE, IT and related engineering disciplines)
    branch_weights = [0.45, 0.22, 0.15, 0.12, 0.06]
    branch = np.random.choice(SUPPORTED_BRANCHES, size=n_samples, p=branch_weights)
    
    # Latent factors (multivariate correlation foundation)
    latent_academic = np.random.normal(0, 1, size=n_samples)
    latent_tech = np.random.normal(0, 1, size=n_samples) + 0.40 * latent_academic
    latent_soft = np.random.normal(0, 1, size=n_samples) + 0.25 * latent_academic
    latent_initiative = np.random.normal(0, 1, size=n_samples) + 0.35 * latent_tech
    
    # 2. Academic Performance & Consistency (Pre-placement)
    cgpa = np.clip(7.55 + 0.85 * latent_academic + np.random.normal(0, 0.35, n_samples), 5.0, 9.95).round(2)
    tenth_percentage = np.clip(78.0 + 7.5 * latent_academic + np.random.normal(0, 4.0, n_samples), 55.0, 99.0).round(1)
    twelfth_percentage = np.clip(76.0 + 8.0 * latent_academic + np.random.normal(0, 4.5, n_samples), 52.0, 98.5).round(1)
    attendance = np.clip(78.5 + 6.0 * latent_academic + np.random.normal(0, 5.0, n_samples), 60.0, 99.5).round(1)
    
    # Backlogs
    backlog_prob = np.clip(0.18 - 0.12 * latent_academic, 0.02, 0.65)
    history_backlogs = np.random.binomial(n=4, p=backlog_prob)
    active_backlogs = np.array([np.random.binomial(n=h, p=0.35) if h > 0 else 0 for h in history_backlogs])
    
    # 3. Technical & Coding Skills (DSA, Development, Architecture)
    dsa_base = np.exp(np.clip(4.8 + 0.65 * latent_tech, 2.5, 6.6))
    dsa_solved = np.clip(dsa_base + np.random.normal(0, 30, n_samples), 0, 750).astype(int)
    dsa_score = np.clip(55.0 + 14.0 * latent_tech + 0.04 * dsa_solved + np.random.normal(0, 5.0, n_samples), 20.0, 100.0).round(1)
    coding_rating = np.clip(1200 + 220 * latent_tech + 1.2 * dsa_solved + np.random.normal(0, 60, n_samples), 900, 2350).astype(int)
    
    web_cloud_score = np.clip(50.0 + 15.0 * latent_tech + 10.0 * latent_initiative + np.random.normal(0, 6.0, n_samples), 20.0, 100.0).round(1)
    system_design_score = np.clip(45.0 + 13.0 * latent_tech + 0.03 * dsa_solved + np.random.normal(0, 7.0, n_samples), 15.0, 100.0).round(1)
    
    # 4. Aptitude & Logical Reasoning
    quant_score = np.clip(60.0 + 11.0 * latent_academic + 5.0 * latent_tech + np.random.normal(0, 6.0, n_samples), 30.0, 100.0).round(1)
    logical_score = np.clip(62.0 + 10.0 * latent_academic + 6.0 * latent_tech + np.random.normal(0, 6.0, n_samples), 32.0, 100.0).round(1)
    verbal_score = np.clip(60.0 + 12.0 * latent_soft + np.random.normal(0, 6.0, n_samples), 30.0, 100.0).round(1)
    aptitude_avg = ((quant_score + logical_score + verbal_score) / 3.0).round(1)
    
    # 5. Communication & Interview Evaluation
    mock_interview_score = np.clip(58.0 + 14.0 * latent_soft + 4.0 * latent_tech + np.random.normal(0, 5.5, n_samples), 25.0, 100.0).round(1)
    gd_score = np.clip(56.0 + 15.0 * latent_soft + np.random.normal(0, 6.0, n_samples), 25.0, 100.0).round(1)
    # Communication rubric 1.0 - 5.0
    communication_rating = np.clip(2.9 + 0.65 * latent_soft + np.random.normal(0, 0.3, n_samples), 1.0, 5.0).round(1)
    
    # 6. Projects & Experiential Exposure
    projects_count = np.clip(np.random.poisson(lam=np.clip(2.6 + 0.9 * latent_initiative, 0.5, 6.0)), 0, 8)
    # Project quality rubric score 15 - 100 (based on functionality, depth, contribution, testing)
    project_quality = np.clip(45.0 + 13.0 * latent_initiative + 3.5 * projects_count + np.random.normal(0, 6.0, n_samples), 15.0, 100.0).round(1)
    
    # Internships
    internship_prob = np.clip(0.35 + 0.22 * latent_initiative + 0.10 * latent_tech, 0.05, 0.90)
    internships_count = np.array([np.random.binomial(n=3, p=p) for p in internship_prob])
    internship_duration = np.array([
        0 if c == 0 else int(np.clip(c * np.random.choice([2, 3, 6]) + np.random.randint(0, 2), 1, 14))
        for c in internships_count
    ])
    
    # Certifications & Hackathons
    certifications_count = np.clip(np.random.poisson(lam=np.clip(1.5 + 0.6 * latent_initiative, 0.2, 5.0)), 0, 7)
    hackathons_participated = np.clip(np.random.poisson(lam=np.clip(1.2 + 0.7 * latent_initiative, 0.1, 5.0)), 0, 8)
    # Logical constraint: won cannot exceed participated
    hackathons_won = np.array([
        np.random.binomial(n=p, p=0.25) if p > 0 else 0 
        for p in hackathons_participated
    ])
    
    # 7. Composite Rubric Score for Supervised Label Generation
    norm_cgpa = np.clip((cgpa - 5.0) / 5.0, 0.0, 1.0)
    norm_dsa = np.clip(dsa_solved / 450.0, 0.0, 1.2) * 0.5 + (dsa_score / 100.0) * 0.5
    norm_tech = (web_cloud_score * 0.5 + system_design_score * 0.5) / 100.0
    norm_aptitude = aptitude_avg / 100.0
    norm_soft = (mock_interview_score * 0.5 + gd_score * 0.3 + (communication_rating / 5.0) * 20.0) / 100.0
    norm_projects = (np.clip(projects_count / 4.0, 0.0, 1.2) * 0.4 + (project_quality / 100.0) * 0.6)
    norm_practical = (np.clip(internships_count / 2.0, 0.0, 1.2) * 0.6 + np.clip(certifications_count / 3.0, 0.0, 1.0) * 0.2 + np.clip(hackathons_won / 1.0, 0.0, 1.0) * 0.2)
    
    raw_composite = (
        0.14 * norm_cgpa +
        0.26 * norm_dsa +
        0.12 * norm_tech +
        0.16 * norm_aptitude +
        0.18 * norm_soft +
        0.08 * norm_projects +
        0.06 * norm_practical
    )
    
    backlog_penalty = active_backlogs * 0.15 + history_backlogs * 0.03
    composite_readiness = np.clip(raw_composite - backlog_penalty + np.random.normal(0, 0.03, n_samples), 0.05, 0.98)
    
    # 8. Supervised Readiness Category Labelling (Foundation, Developing, Placement Ready, Advanced)
    readiness_tier = []
    for i in range(n_samples):
        score = composite_readiness[i]
        ab = active_backlogs[i]
        dsa = dsa_solved[i]
        comm = mock_interview_score[i]
        
        if score >= 0.72 and ab == 0 and dsa >= 250 and comm >= 65:
            readiness_tier.append("Advanced")
        elif score >= 0.54 and ab <= 1 and dsa >= 120:
            readiness_tier.append("Placement Ready")
        elif score >= 0.38 and ab <= 2:
            readiness_tier.append("Developing")
        else:
            readiness_tier.append("Foundation")
            
    # 9. Simulated Experimental Targets (Clearly labeled as experimental baseline)
    placement_prob = np.clip(
        (1.0 / (1.0 + np.exp(-10.0 * (composite_readiness - 0.45)))) * 100 + np.random.normal(0, 2.5, n_samples),
        5.0, 99.5
    ).round(1)
    placement_prob = np.where(active_backlogs >= 2, np.minimum(placement_prob, 45.0), placement_prob)
    
    base_ctc = 3.5 + 32.0 * np.power(composite_readiness, 2.6)
    ctc_boost = (hackathons_won * 1.5) + (internships_count * 0.8) + (dsa_solved > 350) * 2.0
    expected_ctc = np.clip(base_ctc + ctc_boost + np.random.normal(0, 0.6, n_samples), 2.8, 44.0).round(2)
    expected_ctc = np.where(active_backlogs > 0, np.minimum(expected_ctc, 7.5), expected_ctc)
    
    # 10. Best-Fit Target Role
    best_fit_role = []
    for i in range(n_samples):
        b = branch[i]
        d_cnt = dsa_solved[i]
        wc = web_cloud_score[i]
        sd = system_design_score[i]
        q = quant_score[i]
        
        if d_cnt >= 280 and sd >= 65:
            best_fit_role.append("Software Development Engineer (SDE)")
        elif (b == "AI & Data Engineering" or q >= 75) and (cgpa[i] >= 7.8 and projects_count[i] >= 2):
            best_fit_role.append("AI / ML & Data Science Engineer")
        elif wc >= 68 and projects_count[i] >= 2:
            best_fit_role.append("Full Stack Developer")
        elif (wc >= 60 or sd >= 60) and certifications_count[i] >= 1:
            best_fit_role.append("Cloud & DevOps Engineer")
        elif q >= 65 and logical_score[i] >= 65 and mock_interview_score[i] >= 60:
            best_fit_role.append("Data & Business Analyst")
        elif dsa_score[i] >= 45:
            best_fit_role.append("QA & Automation Engineer")
        else:
            best_fit_role.append("Associate Technology Consultant")
            
    student_ids = [f"LPU2026_{1000 + i}" for i in range(n_samples)]
    
    # Note: Column names maintained for backward compatibility, with 'project_quality' replacing unexplained rating
    df = pd.DataFrame({
        "student_id": student_ids,
        "branch": branch,
        "cgpa": cgpa,
        "tenth_percentage": tenth_percentage,
        "twelfth_percentage": twelfth_percentage,
        "attendance_percentage": attendance,
        "active_backlogs": active_backlogs,
        "history_backlogs": history_backlogs,
        "dsa_problems_solved": dsa_solved,
        "dsa_score": dsa_score,
        "coding_rating": coding_rating,
        "web_cloud_score": web_cloud_score,
        "system_design_score": system_design_score,
        "quantitative_aptitude": quant_score,
        "logical_reasoning": logical_score,
        "verbal_ability": verbal_score,
        "mock_interview_score": mock_interview_score,
        "gd_score": gd_score,
        "communication_rating": communication_rating,
        "projects_count": projects_count,
        "github_quality": project_quality,  # Project quality rubric score (functionality, depth, testing)
        "internships_count": internships_count,
        "internship_duration_months": internship_duration,
        "certifications_count": certifications_count,
        "hackathons_participated": hackathons_participated,
        "hackathons_won": hackathons_won,
        # Supervised target
        "readiness_tier": readiness_tier,
        # Experimental simulation targets
        "placement_probability": placement_prob,
        "expected_ctc_lpa": expected_ctc,
        "best_fit_role": best_fit_role
    })
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic placement dataset for LPU Placement Intelligence Platform")
    parser.add_argument("--samples", type=int, default=2000, help="Number of student records to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "data")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "lpu_placement_dataset.csv")
    
    print(f"Generating synthetic placement dataset with {args.samples} student records (seed={args.seed})...", flush=True)
    df = generate_placement_dataset(n_samples=args.samples, random_state=args.seed)
    df.to_csv(out_path, index=False)
    print(f"Dataset successfully created at: {out_path}", flush=True)
    print(f"Shape: {df.shape} (Gender removed: {('gender' in df.columns)})", flush=True)
    print("\nReadiness Category Distribution:", flush=True)
    print(df["readiness_tier"].value_counts(normalize=True).round(3), flush=True)
    print("\nBest Fit Role Distribution:", flush=True)
    print(df["best_fit_role"].value_counts(normalize=True).round(3), flush=True)
