"""
LPU Placement Intelligence Platform - Supervised Machine Learning Algorithms Suite
Pure NumPy implementation of Decision Trees, Random Forests,
Gradient Boosted Decision Trees, Logistic Regression, Ridge Regression, Preprocessing,
and Decision Tree Ensemble Path Feature Attribution (Saabas Decomposition).
"""

import numpy as np
import json
import pickle

# ==========================================
# 1. PREPROCESSING & SCALING
# ==========================================

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class OneHotEncoder:
    def __init__(self):
        self.categories_ = []
        self.feature_names_out_ = []

    def fit(self, X, feature_names=None):
        X = np.asarray(X)
        n_features = X.shape[1]
        self.categories_ = []
        self.feature_names_out_ = []
        
        for i in range(n_features):
            cats = sorted(list(set(X[:, i])))
            self.categories_.append(cats)
            feat_name = feature_names[i] if feature_names else f"feat_{i}"
            for c in cats:
                self.feature_names_out_.append(f"{feat_name}_{c}")
        return self

    def transform(self, X):
        X = np.asarray(X)
        n_samples, n_features = X.shape
        encoded_cols = []
        
        for i in range(n_features):
            cats = self.categories_[i]
            col_data = X[:, i]
            for c in cats:
                encoded_cols.append((col_data == c).astype(np.float64))
                
        return np.column_stack(encoded_cols)

    def fit_transform(self, X, feature_names=None):
        return self.fit(X, feature_names).transform(X)


class DataPreprocessor:
    def __init__(self, numeric_cols, categorical_cols):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder()
        self.transformed_feature_names = []

    def fit(self, df):
        X_num = df[self.numeric_cols].values.astype(np.float64)
        self.scaler.fit(X_num)
        
        if self.categorical_cols:
            X_cat = df[self.categorical_cols].values
            self.encoder.fit(X_cat, self.categorical_cols)
            self.transformed_feature_names = list(self.numeric_cols) + self.encoder.feature_names_out_
        else:
            self.transformed_feature_names = list(self.numeric_cols)
        return self

    def transform(self, df):
        X_num = df[self.numeric_cols].values.astype(np.float64)
        X_num_trans = self.scaler.transform(X_num)
        
        if self.categorical_cols:
            X_cat = df[self.categorical_cols].values
            X_cat_trans = self.encoder.transform(X_cat)
            return np.hstack([X_num_trans, X_cat_trans])
        return X_num_trans

    def fit_transform(self, df):
        return self.fit(df).transform(df)


# ==========================================
# 2. DECISION TREES & ENSEMBLES
# ==========================================

class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, gain=0.0):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Class probability distribution vector or regression mean
        self.gain = gain

    @property
    def is_leaf(self):
        return self.left is None and self.right is None


class DecisionTree:
    def __init__(self, max_depth=6, min_samples_split=6, is_classifier=True, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.is_classifier = is_classifier
        self.max_features = max_features
        self.root = None
        self.n_classes_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if self.is_classifier:
            self.n_classes_ = int(np.max(y) + 1)
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        
        if self.is_classifier:
            counts = np.bincount(y, minlength=self.n_classes_)
            leaf_val = counts / n_samples
            if depth >= self.max_depth or n_samples < self.min_samples_split or len(np.unique(y)) == 1:
                return TreeNode(value=leaf_val)
            current_impurity = 1.0 - np.sum(leaf_val ** 2)
        else:
            leaf_val = float(np.mean(y))
            if depth >= self.max_depth or n_samples < self.min_samples_split or np.var(y) < 1e-7:
                return TreeNode(value=leaf_val)
            current_impurity = float(np.var(y))

        feat_indices = np.arange(n_features)
        if self.max_features and self.max_features < n_features:
            feat_indices = np.random.choice(n_features, size=self.max_features, replace=False)

        best_gain = -1.0
        best_feat = None
        best_thresh = None

        for feat in feat_indices:
            col_vals = X[:, feat]
            thresholds = np.percentile(col_vals, [20, 35, 50, 65, 80])
            thresholds = np.unique(thresholds)

            for thresh in thresholds:
                left_mask = col_vals <= thresh
                right_mask = ~left_mask

                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)

                if n_left < 2 or n_right < 2:
                    continue

                if self.is_classifier:
                    p_left = np.bincount(y[left_mask], minlength=self.n_classes_) / n_left
                    p_right = np.bincount(y[right_mask], minlength=self.n_classes_) / n_right
                    imp_left = 1.0 - np.sum(p_left ** 2)
                    imp_right = 1.0 - np.sum(p_right ** 2)
                else:
                    imp_left = float(np.var(y[left_mask]))
                    imp_right = float(np.var(y[right_mask]))

                gain = current_impurity - ((n_left / n_samples) * imp_left + (n_right / n_samples) * imp_right)

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh

        if best_gain <= 1e-5 or best_feat is None:
            return TreeNode(value=leaf_val)

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return TreeNode(
            feature=best_feat,
            threshold=best_thresh,
            left=left_child,
            right=right_child,
            value=leaf_val,
            gain=best_gain
        )

    def _predict_row(self, node, x):
        if node.is_leaf:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_row(node.left, x)
        return self._predict_row(node.right, x)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._predict_row(self.root, row) for row in X])

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.is_classifier:
            probs = self.predict_proba(X)
            return np.argmax(probs, axis=1)
        else:
            return np.array([self._predict_row(self.root, row) for row in X])


class RandomForestClassifier:
    def __init__(self, n_estimators=30, max_depth=6, min_samples_split=6, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []
        self.classes_ = None
        self.feature_importances_ = None

    def fit(self, X, y):
        np.random.seed(self.random_state)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        max_feat = max(2, int(np.sqrt(n_features)))
        
        self.trees = []
        feat_gains = np.zeros(n_features)

        for _ in range(self.n_estimators):
            boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                is_classifier=True,
                max_features=max_feat
            )
            tree.fit(X[boot_idx], y[boot_idx])
            self.trees.append(tree)
            self._accumulate_gains(tree.root, feat_gains)

        total_gain = np.sum(feat_gains)
        self.feature_importances_ = (feat_gains / total_gain) if total_gain > 0 else np.ones(n_features) / n_features
        return self

    def _accumulate_gains(self, node, gains):
        if node is None or node.is_leaf:
            return
        gains[node.feature] += node.gain
        self._accumulate_gains(node.left, gains)
        self._accumulate_gains(node.right, gains)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        tree_probs = [tree.predict_proba(X) for tree in self.trees]
        return np.mean(tree_probs, axis=0)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def compute_tree_path_attribution(self, x_row):
        """
        Computes Decision Tree Ensemble Path Attributions (Saabas Decomposition)
        for a single student instance. Quantifies how each decision node's split along
        the traversal path shifted the class probability distribution from the root prior.
        """
        n_features = len(x_row)
        n_classes = len(self.classes_)
        attributions = np.zeros((n_classes, n_features))
        
        weight = 1.0 / len(self.trees)
        for tree in self.trees:
            self._explain_tree_path(tree.root, x_row, attributions, weight)
        return attributions

    def _explain_tree_path(self, node, x, attributions, weight):
        if node is None or node.is_leaf:
            return
        feat = node.feature
        child = node.left if x[feat] <= node.threshold else node.right
        if child is not None and child.value is not None and node.value is not None:
            delta = child.value - node.value
            for c in range(min(len(self.classes_), len(delta))):
                attributions[c, feat] += weight * delta[c]
        self._explain_tree_path(child, x, attributions, weight)


class RandomForestRegressor:
    def __init__(self, n_estimators=30, max_depth=6, min_samples_split=5, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []
        self.feature_importances_ = None

    def fit(self, X, y):
        np.random.seed(self.random_state)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        max_feat = max(2, int(n_features / 3))
        
        self.trees = []
        feat_gains = np.zeros(n_features)

        for _ in range(self.n_estimators):
            boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                is_classifier=False,
                max_features=max_feat
            )
            tree.fit(X[boot_idx], y[boot_idx])
            self.trees.append(tree)
            self._accumulate_gains(tree.root, feat_gains)

        total_gain = np.sum(feat_gains)
        self.feature_importances_ = (feat_gains / total_gain) if total_gain > 0 else np.ones(n_features) / n_features
        return self

    def _accumulate_gains(self, node, gains):
        if node is None or node.is_leaf:
            return
        gains[node.feature] += node.gain
        self._accumulate_gains(node.left, gains)
        self._accumulate_gains(node.right, gains)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        preds = [tree.predict(X) for tree in self.trees]
        return np.mean(preds, axis=0)


class GradientBoostingRegressor:
    def __init__(self, n_estimators=30, learning_rate=0.1, max_depth=4, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []
        self.init_mean = 0.0

    def fit(self, X, y):
        np.random.seed(self.random_state)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        self.init_mean = float(np.mean(y))
        y_pred = np.full(len(y), self.init_mean)
        self.trees = []

        for _ in range(self.n_estimators):
            residuals = y - y_pred
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=6, is_classifier=False)
            tree.fit(X, residuals)
            update = tree.predict(X)
            y_pred += self.learning_rate * update
            self.trees.append(tree)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        y_pred = np.full(len(X), self.init_mean)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred


class MultiClassLogisticRegression:
    def __init__(self, lr=0.08, max_iter=300, l2_reg=0.005, random_state=42):
        self.lr = lr
        self.max_iter = max_iter
        self.l2_reg = l2_reg
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.classes_ = None

    def fit(self, X, y):
        np.random.seed(self.random_state)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        self.weights = np.random.normal(0, 0.01, (n_features, n_classes))
        self.bias = np.zeros(n_classes)

        Y_oh = np.zeros((n_samples, n_classes))
        for i, c in enumerate(y):
            Y_oh[i, c] = 1.0

        for _ in range(self.max_iter):
            scores = np.dot(X, self.weights) + self.bias
            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            error = probs - Y_oh
            grad_w = (np.dot(X.T, error) / n_samples) + self.l2_reg * self.weights
            grad_b = np.sum(error, axis=0) / n_samples

            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        scores = np.dot(X, self.weights) + self.bias
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


class RidgeRegressor:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = None
        self.intercept = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        
        self.intercept = float(np.mean(y))
        y_centered = y - self.intercept
        
        I = np.eye(n_features)
        self.weights = np.linalg.solve(X.T @ X + self.alpha * I, X.T @ y_centered)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return X @ self.weights + self.intercept


# ==========================================
# 3. METRICS & EVALUATION FUNCTIONS
# ==========================================

def accuracy_score(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))

def confusion_matrix(y_true, y_pred, n_classes=None):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if n_classes is None:
        n_classes = max(np.max(y_true), np.max(y_pred)) + 1
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def classification_metrics(y_true, y_pred, class_names):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n_classes = len(class_names)
    
    cm = confusion_matrix(y_true, y_pred, n_classes)
    
    precisions = []
    recalls = []
    f1s = []
    supports = []
    
    for c in range(n_classes):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp
        support = np.sum(cm[c, :])
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(int(support))
        
    total_samples = len(y_true)
    macro_f1 = float(np.mean(f1s))
    weighted_f1 = float(np.sum(np.array(f1s) * (np.array(supports) / total_samples)))
    weighted_prec = float(np.sum(np.array(precisions) * (np.array(supports) / total_samples)))
    weighted_rec = float(np.sum(np.array(recalls) * (np.array(supports) / total_samples)))
    accuracy = accuracy_score(y_true, y_pred)
    
    report = {}
    for i, name in enumerate(class_names):
        report[name] = {
            "precision": round(precisions[i], 4),
            "recall": round(recalls[i], 4),
            "f1-score": round(f1s[i], 4),
            "support": supports[i]
        }
    report["accuracy"] = round(accuracy, 4)
    report["macro avg"] = {
        "precision": round(float(np.mean(precisions)), 4),
        "recall": round(float(np.mean(recalls)), 4),
        "f1-score": round(macro_f1, 4),
        "support": total_samples
    }
    report["weighted avg"] = {
        "precision": round(weighted_prec, 4),
        "recall": round(weighted_rec, 4),
        "f1-score": round(weighted_f1, 4),
        "support": total_samples
    }
    
    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "weighted_precision": round(weighted_prec, 4),
        "weighted_recall": round(weighted_rec, 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

def mean_squared_error(y_true, y_pred):
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))

def mean_absolute_error(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

def k_fold_cross_validation(model_class, model_kwargs, X, y, k=3, is_classifier=True):
    n = len(X)
    indices = np.random.permutation(n)
    folds = np.array_split(indices, k)
    scores = []
    
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(k) if j != i])
        
        m = model_class(**model_kwargs)
        m.fit(X[train_idx], y[train_idx])
        preds = m.predict(X[val_idx])
        
        if is_classifier:
            score = accuracy_score(y[val_idx], preds)
        else:
            score = r2_score(y[val_idx], preds)
        scores.append(score)
        
    return float(np.mean(scores)), float(np.std(scores))


# ==========================================
# 4. BASELINE DUMMY CLASSIFIER & ADVANCED METRICS
# ==========================================

class DummyClassifier:
    """Baseline heuristic classifier (majority class / prior distribution)."""
    def __init__(self, strategy="most_frequent", random_state=42):
        self.strategy = strategy
        self.random_state = random_state
        self.classes_ = None
        self.majority_class_ = None
        self.class_priors_ = None

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_, counts = np.unique(y, return_counts=True)
        self.majority_class_ = self.classes_[np.argmax(counts)]
        self.class_priors_ = counts / len(y)
        return self

    def predict(self, X):
        n_samples = len(X)
        if self.strategy == "most_frequent":
            return np.full(n_samples, self.majority_class_)
        elif self.strategy == "prior":
            np.random.seed(self.random_state)
            return np.random.choice(self.classes_, size=n_samples, p=self.class_priors_)
        return np.full(n_samples, self.majority_class_)

    def predict_proba(self, X):
        n_samples = len(X)
        return np.tile(self.class_priors_, (n_samples, 1))


def brier_score_loss(y_true, y_prob):
    """
    Computes Brier score for probability calibration:
    Mean squared difference between predicted probability and actual binary indicator.
    Lower is better (0.0 = perfect calibration).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    return float(np.mean((y_prob - y_true) ** 2))


def roc_auc_score_binary(y_true, y_prob):
    """Computes the Area Under the Receiver Operating Characteristic Curve (ROC-AUC) for binary targets."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return 0.5  # Undefined single-class fallback
        
    order = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true[order]
    
    tp = 0
    fp = 0
    tp_prev = 0
    fp_prev = 0
    auc = 0.0
    
    for val in y_true_sorted:
        if val == 1:
            tp += 1
        else:
            fp += 1
            # Trapezoid rule area increment
            auc += (tp + tp_prev) / 2.0
            tp_prev = tp
            fp_prev = fp
            
    return float(auc / (n_pos * n_neg)) if (n_pos * n_neg) > 0 else 0.5


def pr_auc_score_binary(y_true, y_prob):
    """Computes Precision-Recall Area Under the Curve (PR-AUC) via numerical integration."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    
    n_pos = np.sum(y_true == 1)
    if n_pos == 0:
        return 0.0
        
    thresholds = np.linspace(0.0, 1.0, 50)
    precisions = []
    recalls = []
    
    for t in sorted(thresholds, reverse=True):
        y_pred = (y_prob >= t).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        
    # Sort by recall for trapezoidal area
    rec_arr = np.array(recalls)
    prec_arr = np.array(precisions)
    idx = np.argsort(rec_arr)
    sorted_rec = rec_arr[idx]
    sorted_prec = prec_arr[idx]
    # Pure NumPy trapz formula compatible with NumPy 1.x and 2.x+
    area = np.sum((sorted_prec[:-1] + sorted_prec[1:]) * np.diff(sorted_rec)) / 2.0
    return float(np.clip(area, 0.0, 1.0))

