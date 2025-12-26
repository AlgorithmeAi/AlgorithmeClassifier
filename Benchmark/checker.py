import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from algorithmeclassifier import FastAlgorithmeClassifier

def make_checkerboard(n_samples=1000):
    X = np.random.uniform(-1, 1, (n_samples, 2))
    # Classe 1 si les signes de x et y sont différents (cases 2 et 4 de l'échiquier)
    y = (np.sign(X[:, 0]) != np.sign(X[:, 1])).astype(int)
    return X, y

def test_checkerboard_intelligence():
    X, y = make_checkerboard(2000)
    
    # On ne donne que 12 points ! (C'est presque rien pour 4 zones)
    X_train, y_train = X[:12], y[:12]
    X_test, y_test = X[12:], y[12:]
    
    models = {
        "🐍 Snake": FastAlgorithmeClassifier(n_layers=100),
        "🌲 RF": RandomForestClassifier(n_estimators=100),
        "🔥 GB": HistGradientBoostingClassifier()
    }

    print("🧪 TEST: L'ÉCHIQUIER (12 samples training)")
    print("-" * 50)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"{name:<10} Accuracy: {model.score(X_test, y_test):.4f}")

if __name__ == "__main__":
    test_checkerboard_intelligence()
