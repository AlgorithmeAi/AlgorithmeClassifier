import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from algorithmeclassifier import FastAlgorithmeClassifier

def test_noisy_small_data():
    np.random.seed(42)
    # 1000 points de test, 20 points d'entraînement
    X = np.random.randn(1020, 2)
    # La règle d'or : une diagonale simple (x + y > 0)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # On isole les 20 points d'entraînement et on injecte 30% de bruit
    X_train, y_train = X[:20], y[:20]
    X_test, y_test = X[20:], y[20:]
    
    # Inversion de 6 labels sur 20 (30% de bruit)
    noise_indices = np.random.choice(20, 6, replace=False)
    y_train[noise_indices] = 1 - y_train[noise_indices]
    
    models = {
        "🐍 Snake": FastAlgorithmeClassifier(n_layers=100),
        "🌲 RF": RandomForestClassifier(n_estimators=100),
        "🔥 GB": HistGradientBoostingClassifier()
    }

    print("🧪 TEST: RARE ET SALE (20 samples / 30% Noise)")
    print("-" * 55)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"{name:<10} Accuracy sur test propre: {model.score(X_test, y_test):.4f}")

if __name__ == "__main__":
    test_noisy_small_data()
