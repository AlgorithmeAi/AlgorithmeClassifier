import numpy as np
import pandas as pd
from time import time
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from joblib import Parallel, delayed

# --- FONCTIONS STATIQUES POUR PARALLÉLISATION ---
def _evaluate_clause_batch_static(X, clause):
    """Évaluation vectorisée d'une clause sur un bloc de données."""
    mask = np.zeros(X.shape[0], dtype=bool)
    for f_idx, val, is_le in clause:
        if is_le: mask |= (X[:, f_idx] <= val)
        else: mask |= (X[:, f_idx] > val)
    return mask

def _fit_single_layer(population, target, unique_targets):
    """Génère une couche de règles SAT (Théorème de Dana)."""
    layer_models = []
    for t in unique_targets:
        f_idx_list = np.where(target == t)[0]
        ts_idx_list = np.where(target != t)[0]
        temp_f = list(f_idx_list)
        
        while len(temp_f) > 0:
            f_idx = np.random.choice(temp_f)
            T_idx = np.random.choice(ts_idx_list)
            T_row, F_row = population[T_idx], population[f_idx]
            diffs = np.where(T_row != F_row)[0]
            
            if len(diffs) == 0: 
                clause = [[0, 0, True]]
            else:
                feat = np.random.choice(diffs)
                clause = [[feat, (T_row[feat] + F_row[feat]) / 2.0, T_row[feat] < F_row[feat]]]
            
            # Extension de la clause pour garantir la séparation
            while not np.all(_evaluate_clause_batch_static(population[ts_idx_list], clause)):
                mask_ts = _evaluate_clause_batch_static(population[ts_idx_list], clause)
                ts_rem = ts_idx_list[~mask_ts]
                if len(ts_rem) == 0: break
                T_row_rem = population[np.random.choice(ts_rem)]
                diffs = np.where(T_row_rem != F_row)[0]
                if len(diffs) > 0:
                    feat = np.random.choice(diffs)
                    clause.append([feat, (T_row_rem[feat] + F_row[feat]) / 2.0, T_row_rem[feat] < F_row[feat]])

            # Calcul des lookalikes (nombre d'échantillons F satisfaisant la concordance)
            mask_f = _evaluate_clause_batch_static(population[f_idx_list], clause)
            covered_indices = f_idx_list[~mask_f]
            layer_models.append((t, clause, len(covered_indices)))
            temp_f = [idx for idx in temp_f if idx not in covered_indices]
    return layer_models

class AlgorithmeClassifier():
    def __init__(self, n_layers=100, n_jobs=-1):
        self.layers = n_layers
        self.n_jobs = n_jobs
        self.sat_models = []
        self.classes_ = None

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int).ravel()
        self.classes_ = np.unique(y)
        # Distribution du calcul sur les cœurs CPU
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_single_layer)(X, y, self.classes_) for _ in range(self.layers)
        )
        self.sat_models = [m for layer in results for m in layer]

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        scores = np.zeros((X.shape[0], len(self.classes_)))
        target_to_idx = {t: i for i, t in enumerate(self.classes_)}
        
        # Scoring par accumulation de concordance
        for t, clause, weight in self.sat_models:
            mask = ~_evaluate_clause_batch_static(X, clause)
            scores[mask, target_to_idx[t]] += weight
        
        row_sums = scores.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        probs = scores / row_sums
        # Gestion des points hors domaine
        if np.any(row_sums == 1.0): 
            probs[row_sums.ravel() == 1.0] = 1.0 / len(self.classes_)
        return probs

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

# --- MODULE DE VALIDATION ---
if __name__ == "__main__":
    print("🔬 ANALYSE COMPARATIVE : DATASET DIGITS (64 features, 10 classes)")
    print("-" * 70)
    
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.3, random_state=42
    )

    models = {
        "Random Forest (100)": RandomForestClassifier(n_estimators=100, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=300),
        "Algorithme (Dana SAT)": AlgorithmeClassifier(n_layers=100, n_jobs=-1)
    }

    results = []
    for name, clf in models.items():
        t0 = time()
        clf.fit(X_train, y_train)
        t_fit = time() - t0
        
        t1 = time()
        y_proba = clf.predict_proba(X_test)
        t_pred = time() - t1
        
        y_pred = np.argmax(y_proba, axis=1)
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
        results.append([name, f"{t_fit:.3f}s", f"{t_pred:.3f}s", f"{acc:.4f}", f"{auc:.4f}"])

    df = pd.DataFrame(results, columns=["Modèle", "Fit Time", "Pred Time", "Accuracy", "AUROC"])
    print(df.to_string(index=False))
    print("-" * 70)
