import numpy as np
from sklearn.ensemble import RandomForestClassifier
from algorithmeclassifier import FastAlgorithmeClassifier

def test_scale_sensitivity():
    # 1000 points, 2 features informatives
    X = np.random.randn(1000, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # On torture les données : la feature 2 est multipliée par 1 000 000
    X_scaled = X.copy()
    X_scaled[:, 1] = X_scaled[:, 1] * 1_000_000
    
    models = {
        "🐍 Snake": FastAlgorithmeClassifier(n_layers=100),
        "🌲 RF": RandomForestClassifier()
    }

    print("🧪 TEST: CHOC DES ÉCHELLES (x1,000,000 factor)")
    print("-" * 50)
    
    for name, model in models.items():
        model.fit(X_scaled, y)
        print(f"{name:<10} Accuracy: {model.score(X_scaled, y):.4f}")

if __name__ == "__main__":
    test_scale_sensitivity()
