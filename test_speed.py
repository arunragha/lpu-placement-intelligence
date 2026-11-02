import time
import numpy as np
import pandas as pd
from src.ml_algorithms import DecisionTree, RandomForestClassifier, DataPreprocessor

df = pd.read_csv("data/lpu_placement_dataset.csv")
num_cols = ["cgpa", "tenth_percentage", "twelfth_percentage", "attendance_percentage",
            "active_backlogs", "history_backlogs", "dsa_problems_solved", "dsa_score",
            "coding_rating", "web_cloud_score", "system_design_score",
            "quantitative_aptitude", "logical_reasoning", "verbal_ability",
            "mock_interview_score", "gd_score", "communication_rating",
            "projects_count", "github_quality", "internships_count",
            "internship_duration_months", "certifications_count",
            "hackathons_participated", "hackathons_won"]
cat_cols = ["branch", "gender"]
prep = DataPreprocessor(num_cols, cat_cols)
X = prep.fit_transform(df)
tier_map = {"At-Risk / Needs Foundation":0, "IT Services & Mass Ready":1, "Core & Product Ready":2, "Dream & Super-Dream Ready":3}
y = np.array([tier_map[t] for t in df["readiness_tier"]])

t0 = time.time()
dt = DecisionTree(max_depth=5, min_samples_split=10)
dt.fit(X[:1000], y[:1000])
t1 = time.time()
print(f"DecisionTree 1000 samples time: {t1-t0:.3f}s", flush=True)
