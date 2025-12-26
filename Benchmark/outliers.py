import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from algorithmeclassifier import FastAlgorithmeClassifier

def test_outlier_robustness():
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, random_state=42)
    
    contamination_levels = [0, 0.05, 0.10, 0.15, 0.20] # % de labels corrompus
    results = { "Snake": [], "RF": [], "GB": [], "LogReg": [] }

    for level in contamination_levels:
        y_noisy = y.copy()
        n_corrupt = int(level * len(y))
        idx_to_corrupt = np.random.choice(len(y), n_corrupt, replace=False)
        # On inverse les labels (0->1, 1->0) pour simuler une erreur
        y_noisy[idx_to_corrupt] = 1 - y_noisy[idx_to_corrupt]
        
        models = {
            "Snake": FastAlgorithmeClassifier(n_layers=100),
            "RF": RandomForestClassifier(),
            "GB": HistGradientBoostingClassifier(),
            "LogReg": LogisticRegression()
        }
        
        for name, model in models.items():
            model.fit(X, y_noisy)
            # On évalue sur le y d'origine (le "vrai")
            score = model.score(X, y)
            results[name].append(score)
            print(f"Bruit label: {level*100:>2}% | {name}: {score:.4f}")

    plt.figure(figsize=(10, 6))
    for name, scores in results.items():
        plt.plot(contamination_levels, scores, 'o-', label=name)
    
    plt.title("Résistance à la corruption des données (Outliers)")
    plt.xlabel("Proportion de labels erronés injectés")
    plt.ylabel("Précision sur les vrais labels")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    test_outlier_robustness()
