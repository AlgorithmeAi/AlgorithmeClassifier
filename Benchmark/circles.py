import numpy as np
from sklearn.datasets import make_circles
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from algorithmeclassifier import FastAlgorithmeClassifier

def test_the_winner():
    # Un problème de géométrie (cercles)
    X, y = make_circles(n_samples=5000, noise=0.1, factor=0.3, random_state=42)
    
    # Entraînement sur un échantillon RIDICULE (15 points seulement)
    X_train, y_train = X[:15], y[:15]
    X_test, y_test = X[15:], y[15:]
    
    models = {
        "🐍 Snake": FastAlgorithmeClassifier(n_layers=100),
        "🌲 RF": RandomForestClassifier(),
        "🔥 GB": HistGradientBoostingClassifier()
    }

    print("🧪 TEST: LE SAINT GRAAL (Small Data + Geometry)")
    print("-" * 50)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"{name:<10} Accuracy: {model.score(X_test, y_test):.4f}")

if __name__ == "__main__":
    test_the_winner()
