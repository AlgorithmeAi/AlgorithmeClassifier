import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from algorithmeclassifier import FastAlgorithmeClassifier

def test_extreme_generalization():
    # Dataset complexe : 20 features, mais on ne donne que 20 lignes pour apprendre !
    X, y = make_classification(n_samples=10020, n_features=20, n_informative=15, random_state=42)
    
    # Séparation extrême
    X_train, y_train = X[:20], y[:20]  # 20 points pour l'entraînement
    X_test, y_test = X[20:], y[20:]    # 10 000 points pour le test
    
    models = {
        "🐍 Snake": FastAlgorithmeClassifier(n_layers=100),
        "🌲 RF": RandomForestClassifier(n_estimators=100),
        "🔥 GB": HistGradientBoostingClassifier()
    }

    print("🧪 TEST: GÉNÉRALISATION EXTRÊME (20 samples training / 10k test)")
    print("-" * 60)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"{name:<10} Accuracy sur 10 000 points: {score:.4f}")

if __name__ == "__main__":
    test_extreme_generalization()
