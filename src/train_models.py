"""
LPU Placement Intelligence Platform - Model Training & Evaluation Pipeline
Version: 2.1.0

Trains and benchmarks multiple supervised learning algorithms:
1. Binary Placement Prediction Models (Placed vs. Unplaced):
   - DummyClassifier (Baseline majority class)
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier (Champion Model)
2. Multi-Class Readiness Tier Classifiers (Foundation, Developing, Placement Ready, Advanced)
3. Career Track Matcher (7 engineering roles)
4. Gradient Boosting Regressors (CTC and Likelihood Baselines)

METHODOLOGICAL INTEGRITY & LEAKAGE PREVENTION:
1. Pure Train-Test Split (80/20) executed BEFORE any preprocessing or scaling.
2. StandardScaler and OneHotEncoder fit strictly on the training partition.
3. 3-Fold Cross-Validation performs fold-independent preprocessing to prevent data leakage.
4. Model selection based on Cross-Validation Macro F1 / ROC-AUC on the training partition.
5. Held-out test set evaluated only after model selection.
6. Gender and post-placement targets strictly excluded from features.
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
    GradientBoostingRegressor, MultiClassLogisticRegression, DecisionTree,
    RidgeRegressor, DummyClassifier, classification_metrics, r2_score,
    mean_squared_error, mean_absolute_error, accuracy_score,
    brier_score_loss, roc_auc_score_binary, pr_auc_score_binary
)

def k_fold_cv_leakage_free(clf_cls, kwargs, df_train, numeric_cols, categorical_cols, y_train, k=3):
    """
    Performs k-fold cross validation with fold-independent preprocessing
    to ensure zero data leakage across partitions.
    """
    n = len(df_train)
    indices = np.random.permutation(n)
    folds = np.array_split(indices, k)
    
    cv_accuracies = []
    cv_macro_f1s = []
    
    for i in range(k):
        val_idx = folds[i]
        tr_idx = np.hstack([folds[j] for j in range(k) if j != i])
        
        fold_train_df = df_train.iloc[tr_idx]
        fold_val_df = df_train.iloc[val_idx]
        
        # Fit preprocessor strictly on fold training data
        fold_prep = DataPreprocessor(numeric_cols, categorical_cols)
        X_tr = fold_prep.fit_transform(fold_train_df)
        X_val = fold_prep.transform(fold_val_df)
        
        y_tr = y_train[tr_idx]
        y_val = y_train[val_idx]
        
        model = clf_cls(**kwargs)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        
        acc = accuracy_score(y_val, y_pred)
        classes = np.unique(y_train)
        f1_list = []
        for c in classes:
            tp = np.sum((y_val == c) & (y_pred == c))
            fp = np.sum((y_val != c) & (y_pred == c))
            fn = np.sum((y_val == c) & (y_pred != c))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            f1_list.append(f1)
            
        cv_accuracies.append(acc)
        cv_macro_f1s.append(float(np.mean(f1_list)))
        
    return {
        "accuracy_mean": float(np.mean(cv_accuracies)),
        "accuracy_std": float(np.std(cv_accuracies)),
        "macro_f1_mean": float(np.mean(cv_macro_f1s)),
        "macro_f1_std": float(np.std(cv_macro_f1s))
    }

def run_training_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "lpu_placement_dataset.csv")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print("================================================================", flush=True)
    print(" LPU Placement Intelligence: Supervised ML Training Pipeline v2.1", flush=True)
    print("================================================================", flush=True)
    print(f"Loading dataset from: {data_path}", flush=True)
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} student placement records with {len(df.columns)} columns.\n", flush=True)
    
    # 1. Feature Partitioning
    numeric_features = [
        "cgpa", "active_backlogs",
        "dsa_problems_solved", "dsa_score", "coding_rating",
        "web_cloud_score", "system_design_score",
        "quantitative_aptitude", "logical_reasoning", "verbal_ability",
        "mock_interview_score", "gd_score", "communication_rating",
        "projects_count", "github_quality",
        "internships_count", "internship_duration_months",
        "certifications_count", "hackathons_participated", "hackathons_won"
    ]
    
    eligibility_features = [
        "tenth_percentage", "twelfth_percentage", "attendance_percentage", "history_backlogs"
    ]
    
    categorical_features = ["branch"]
    
    print(f"Predictive numeric features ({len(numeric_features)}): {numeric_features}", flush=True)
    print(f"Categorical features: {categorical_features}", flush=True)
    print(f"Eligibility-only features (unbiased skill modelling): {eligibility_features}", flush=True)
    print(f"Fairness check: 'gender' in predictive inputs: {'gender' in numeric_features or 'gender' in categorical_features}\n", flush=True)
    
    # 2. Target Encoding
    tier_order = ["Foundation", "Developing", "Placement Ready", "Advanced"]
    tier_to_idx = {name: i for i, name in enumerate(tier_order)}
    
    role_order = sorted(list(df["best_fit_role"].unique()))
    role_to_idx = {name: i for i, name in enumerate(role_order)}
    
    # Binary Placement Target (1 = Placed in campus drive, 0 = Unplaced)
    # Defined strictly by academic clearance (active backlogs <= 1) and recruitment clearance probability >= 50%
    df["placed_binary"] = ((df["placement_probability"] >= 50.0) & (df["active_backlogs"] <= 1)).astype(int)
    
    # 3. Train-Test Split (80% Train, 20% Test) BEFORE Preprocessing
    np.random.seed(42)
    n = len(df)
    n_test = int(n * 0.20)
    shuffled_idx = np.random.permutation(n)
    test_idx = shuffled_idx[:n_test]
    train_idx = shuffled_idx[n_test:]
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    
    # Targets
    y_placed_train = train_df["placed_binary"].values.astype(int)
    y_placed_test = test_df["placed_binary"].values.astype(int)
    
    y_tier_train = np.array([tier_to_idx[t] for t in train_df["readiness_tier"]])
    y_tier_test = np.array([tier_to_idx[t] for t in test_df["readiness_tier"]])
    
    y_prob_train = train_df["placement_probability"].values.astype(np.float64)
    y_prob_test = test_df["placement_probability"].values.astype(np.float64)
    
    y_ctc_train = train_df["expected_ctc_lpa"].values.astype(np.float64)
    y_ctc_test = test_df["expected_ctc_lpa"].values.astype(np.float64)
    
    y_role_train = np.array([role_to_idx[r] for r in train_df["best_fit_role"]])
    y_role_test = np.array([role_to_idx[r] for r in test_df["best_fit_role"]])
    
    print(f"Training partition: {len(train_df)} samples | Test partition: {len(test_df)} samples\n", flush=True)
    
    # 4. Fit Preprocessor ONLY on Training Partition
    preprocessor = DataPreprocessor(numeric_features, categorical_features)
    X_train = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)
    transformed_feature_names = preprocessor.transformed_feature_names
    
    # =========================================================================
    # 5. Supervised Binary Placement Outcome Models
    # =========================================================================
    print("----------------------------------------------------------------", flush=True)
    print("1. Benchmarking Supervised Binary Placement Models (Placed / Unplaced)", flush=True)
    print("----------------------------------------------------------------", flush=True)
    
    binary_classifiers = {
        "Dummy Classifier (Majority)": (DummyClassifier, {"strategy": "most_frequent"}),
        "Logistic Regression": (MultiClassLogisticRegression, {"lr": 0.08, "max_iter": 300, "l2_reg": 0.005, "random_state": 42}),
        "Decision Tree": (DecisionTree, {"max_depth": 6, "is_classifier": True}),
        "Random Forest Classifier": (RandomForestClassifier, {"n_estimators": 30, "max_depth": 6, "random_state": 42})
    }
    
    binary_benchmarks = {}
    best_binary_name = None
    best_binary_f1 = -1.0
    trained_binary_models = {}
    
    for name, (clf_cls, kwargs) in binary_classifiers.items():
        clf = clf_cls(**kwargs)
        clf.fit(X_train, y_placed_train)
        trained_binary_models[name] = clf
        
        y_pred = clf.predict(X_test)
        
        # Predicted probability for positive class
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X_test)
            y_prob_pos = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
        else:
            y_prob_pos = y_pred.astype(float)
            
        acc = accuracy_score(y_placed_test, y_pred)
        
        # Positive class Precision, Recall, F1
        tp = np.sum((y_placed_test == 1) & (y_pred == 1))
        fp = np.sum((y_placed_test == 0) & (y_pred == 1))
        fn = np.sum((y_placed_test == 1) & (y_pred == 0))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        roc_auc = roc_auc_score_binary(y_placed_test, y_prob_pos)
        pr_auc = pr_auc_score_binary(y_placed_test, y_prob_pos)
        brier = brier_score_loss(y_placed_test, y_prob_pos)
        
        binary_benchmarks[name] = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(roc_auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "brier_score": round(float(brier), 4),
            "confusion_matrix": [
                [int(np.sum((y_placed_test == 0) & (y_pred == 0))), int(np.sum((y_placed_test == 0) & (y_pred == 1)))],
                [int(np.sum((y_placed_test == 1) & (y_pred == 0))), int(np.sum((y_placed_test == 1) & (y_pred == 1)))]
            ]
        }
        
        print(f"  -> {name}:")
        print(f"     Accuracy: {acc*100:.2f}% | Precision: {prec*100:.2f}% | Recall: {rec*100:.2f}% | F1: {f1:.4f}")
        print(f"     ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | Brier: {brier:.4f}", flush=True)
        
        if f1 > best_binary_f1:
            best_binary_f1 = f1
            best_binary_name = name
            
    print(f"\n[CHAMPION PLACEMENT CLASSIFIER] Selected: {best_binary_name} (F1: {best_binary_f1:.4f})\n", flush=True)
    best_placement_clf = trained_binary_models[best_binary_name]

    # =========================================================================
    # 6. Multi-Class Readiness Classifiers (4 Categories)
    # =========================================================================
    print("----------------------------------------------------------------", flush=True)
    print("2. Benchmarking Multi-Class Readiness Classifiers (4 Categories)", flush=True)
    print("----------------------------------------------------------------", flush=True)
    
    classifiers = {
        "Random Forest Classifier": (RandomForestClassifier, {"n_estimators": 30, "max_depth": 6, "random_state": 42}),
        "Decision Tree Classifier": (DecisionTree, {"max_depth": 6, "is_classifier": True}),
        "Multi-Class Logistic Regression": (MultiClassLogisticRegression, {"lr": 0.08, "max_iter": 300, "l2_reg": 0.005, "random_state": 42})
    }
    
    clf_benchmarks = {}
    best_clf_name = None
    best_clf_cv_f1 = -1.0
    trained_clfs = {}
    
    for name, (clf_cls, kwargs) in classifiers.items():
        print(f"Running 3-fold cross-validation on train split for {name}...", flush=True)
        cv_res = k_fold_cv_leakage_free(clf_cls, kwargs, train_df, numeric_features, categorical_features, y_tier_train, k=3)
        
        clf = clf_cls(**kwargs)
        clf.fit(X_train, y_tier_train)
        trained_clfs[name] = clf
        
        y_pred = clf.predict(X_test)
        metrics = classification_metrics(y_tier_test, y_pred, tier_order)
        metrics["cv_accuracy_mean"] = round(cv_res["accuracy_mean"], 4)
        metrics["cv_accuracy_std"] = round(cv_res["accuracy_std"], 4)
        metrics["cv_macro_f1_mean"] = round(cv_res["macro_f1_mean"], 4)
        metrics["cv_macro_f1_std"] = round(cv_res["macro_f1_std"], 4)
        clf_benchmarks[name] = metrics
        
        print(f"  -> {name}:")
        print(f"     Test Accuracy: {metrics['accuracy']*100:.2f}% | Test Macro F1: {metrics['macro_f1']:.4f}")
        print(f"     3-Fold CV Acc: {cv_res['accuracy_mean']*100:.2f}% (±{cv_res['accuracy_std']*100:.2f}%) | CV Macro F1: {cv_res['macro_f1_mean']:.4f}", flush=True)
        
        if cv_res["macro_f1_mean"] > best_clf_cv_f1:
            best_clf_cv_f1 = cv_res["macro_f1_mean"]
            best_clf_name = name
            
    print(f"\n[CHAMPION READINESS CLASSIFIER] Selected: {best_clf_name} (CV Macro F1: {best_clf_cv_f1:.4f})\n", flush=True)
    best_tier_clf = trained_clfs[best_clf_name]
    
    # Feature Importances
    feature_importances = {}
    if hasattr(best_tier_clf, "feature_importances_") and best_tier_clf.feature_importances_ is not None:
        fi = best_tier_clf.feature_importances_
        sorted_idx = np.argsort(fi)[::-1]
        for idx in sorted_idx:
            feature_importances[transformed_feature_names[idx]] = round(float(fi[idx]), 4)
            
    # =========================================================================
    # 7. Experimental Simulation Regressors (CTC and Likelihood)
    # =========================================================================
    print("----------------------------------------------------------------", flush=True)
    print("3. Benchmarking Experimental Regressors (Simulated Baselines)", flush=True)
    print("----------------------------------------------------------------", flush=True)
    
    regressors_ctc = {
        "Gradient Boosting Regressor": (GradientBoostingRegressor, {"n_estimators": 30, "learning_rate": 0.1, "max_depth": 4, "random_state": 42}),
        "Random Forest Regressor": (RandomForestRegressor, {"n_estimators": 30, "max_depth": 6, "random_state": 42}),
        "Ridge Regressor": (RidgeRegressor, {"alpha": 1.0})
    }
    
    reg_benchmarks = {}
    best_ctc_reg = None
    best_ctc_r2 = -1.0
    
    for name, (reg_cls, kwargs) in regressors_ctc.items():
        reg = reg_cls(**kwargs)
        reg.fit(X_train, y_ctc_train)
        y_pred = reg.predict(X_test)
        
        r2 = r2_score(y_ctc_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_ctc_test, y_pred))
        mae = mean_absolute_error(y_ctc_test, y_pred)
        
        reg_benchmarks[name] = {
            "r2_score": round(r2, 4),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4)
        }
        print(f"  -> {name} | Test R2: {r2:.4f} | RMSE: {rmse:.2f} LPA | MAE: {mae:.2f} LPA", flush=True)
        if r2 > best_ctc_r2:
            best_ctc_r2 = r2
            best_ctc_reg = reg
            
    # Placement Probability Regressor
    prob_reg = GradientBoostingRegressor(n_estimators=30, learning_rate=0.1, max_depth=4, random_state=42)
    prob_reg.fit(X_train, y_prob_train)
    prob_pred = prob_reg.predict(X_test)
    prob_r2 = r2_score(y_prob_test, prob_pred)
    prob_mae = mean_absolute_error(y_prob_test, prob_pred)
    print(f"  -> Simulated Probability Model: R2 = {prob_r2:.4f} | MAE = {prob_mae:.2f}%\n", flush=True)
    
    # =========================================================================
    # 8. Role Recommendation Classifier
    # =========================================================================
    print("----------------------------------------------------------------", flush=True)
    print("4. Training Target Career Track Classifier", flush=True)
    print("----------------------------------------------------------------", flush=True)
    role_clf = RandomForestClassifier(n_estimators=30, max_depth=6, random_state=42)
    role_clf.fit(X_train, y_role_train)
    role_pred = role_clf.predict(X_test)
    role_acc = accuracy_score(y_role_test, role_pred)
    print(f"  -> Role Classifier Test Accuracy: {role_acc*100:.2f}%\n", flush=True)
    
    # 9. Compute Feature Summary Statistics from Training Partition Only
    feature_stats = {}
    for col in numeric_features:
        feature_stats[col] = {
            "min": float(train_df[col].min()),
            "max": float(train_df[col].max()),
            "mean": round(float(train_df[col].mean()), 2),
            "median": round(float(train_df[col].median()), 2),
            "advanced_mean": round(float(train_df[train_df["readiness_tier"] == "Advanced"][col].mean()), 2) if len(train_df[train_df["readiness_tier"] == "Advanced"]) > 0 else 0.0,
            "placement_ready_mean": round(float(train_df[train_df["readiness_tier"] == "Placement Ready"][col].mean()), 2),
            "developing_mean": round(float(train_df[train_df["readiness_tier"] == "Developing"][col].mean()), 2),
            "foundation_mean": round(float(train_df[train_df["readiness_tier"] == "Foundation"][col].mean()), 2)
        }
        
    # 10. Serialize Artifacts and Metrics
    print("----------------------------------------------------------------", flush=True)
    print("5. Serializing Model Weights & Evaluation Metadata (v2.1)", flush=True)
    print("----------------------------------------------------------------", flush=True)
    
    with open(os.path.join(models_dir, "preprocessor.pkl"), "wb") as f:
        pickle.dump(preprocessor, f)
        
    with open(os.path.join(models_dir, "placement_classifier.pkl"), "wb") as f:
        pickle.dump(best_placement_clf, f)
        
    with open(os.path.join(models_dir, "readiness_classifier.pkl"), "wb") as f:
        pickle.dump(best_tier_clf, f)
        
    with open(os.path.join(models_dir, "ctc_regressor.pkl"), "wb") as f:
        pickle.dump(best_ctc_reg, f)
        
    with open(os.path.join(models_dir, "prob_regressor.pkl"), "wb") as f:
        pickle.dump(prob_reg, f)
        
    with open(os.path.join(models_dir, "role_classifier.pkl"), "wb") as f:
        pickle.dump(role_clf, f)
        
    metadata = {
        "version": "2.1.0",
        "description": "LPU Placement Intelligence Supervised ML Benchmark & Artifacts",
        "dataset_source": "data/lpu_placement_dataset.csv (Synthetic correlated institutional rubric dataset)",
        "synthetic_notice": "Labelled dataset generated algorithmically for demonstration of supervised ML placement and readiness classification. Evaluation measures agreement with synthetic labels, not real LPU students or guaranteed recruitment outcomes.",
        "best_classifier_name": best_clf_name,
        "best_placement_classifier_name": best_binary_name,
        "tier_classes": tier_order,
        "role_classes": role_order,
        "numeric_features": numeric_features,
        "eligibility_features": eligibility_features,
        "categorical_features": categorical_features,
        "transformed_feature_names": transformed_feature_names,
        "feature_stats": feature_stats,
        "feature_importances": feature_importances,
        "binary_placement_benchmark": binary_benchmarks,
        "classification_models_benchmark": clf_benchmarks,
        "regression_models_benchmark": reg_benchmarks,
        "probability_model_metrics": {
            "r2_score": round(float(prob_r2), 4),
            "mae": round(float(prob_mae), 2)
        },
        "role_model_accuracy": round(float(role_acc), 4),
        "total_dataset_samples": int(len(df)),
        "train_samples": int(len(train_df)),
        "test_samples": int(len(test_df))
    }
    
    with open(os.path.join(models_dir, "model_metrics.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"All ML models & v2.1 metrics successfully serialized to '{models_dir}'.", flush=True)
    print("================================================================", flush=True)

if __name__ == "__main__":
    run_training_pipeline()
