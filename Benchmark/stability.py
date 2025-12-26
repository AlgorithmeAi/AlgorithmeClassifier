import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from algorithmeclassifier import FastAlgorithmeClassifier

def stability_test():
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    
    models = {
        "Snake": FastAlgorithmeClassifier(n_layers=100),
        "RF": RandomForestClassifier(),
        "GradBoost": GradientBoostingClassifier(),
        "LogReg": LogisticRegression(),
        "Linear": RidgeClassifier()
    }

    print("📊 TEST DE STABILITÉ (Cross-Validation - 5 folds)")
    print("-" * 45)
    
    for name, model in models.items():
        # On calcule la moyenne et l'écart-type sur 5 lancements
        scores = cross_val_score(model, X, y, cv=5)
        print(f"{name:<10} | Acc: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

if __name__ == "__main__":
    stability_test()
