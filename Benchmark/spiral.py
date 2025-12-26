import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from algorithmeclassifier import FastAlgorithmeClassifier

def make_spiral(n_samples=1000, noise=0.1):
    n = np.sqrt(np.random.rand(n_samples, 1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.randn(n_samples, 1) * noise
    d1y = np.sin(n)*n + np.random.randn(n_samples, 1) * noise
    return np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))), \
           np.hstack((np.zeros(n_samples), np.ones(n_samples)))

def test_spiral_intelligence():
    # Génération de la spirale
    X, y = make_spiral(n_samples=500, noise=0.2)
    
    # On ne donne que 40 points pour apprendre une forme si complexe !
    indices = np.random.permutation(len(X))
    X_train, y_train = X[indices[:40]], y[indices[:40]]
    X_test, y_test = X[indices[40:]], y[indices[40:]]
    
    models = {
        "🐍 Snake": FastAlgorithmeClassifier(n_layers=150),
        "🌲 RF": RandomForestClassifier(),
        "🔥 GB": HistGradientBoostingClassifier()
    }

    print("🧪 TEST: LA SPIRALE DE LANG (40 samples training)")
    print("-" * 50)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"{name:<10} Accuracy: {score:.4f}")

if __name__ == "__main__":
    test_spiral_intelligence()
