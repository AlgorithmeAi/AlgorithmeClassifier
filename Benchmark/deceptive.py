import numpy as np
from sklearn.ensemble import RandomForestClassifier
from algorithmeclassifier import FastAlgorithmeClassifier

def test_deception_resilience():
    np.random.seed(42)
    # 1. On crée 2000 points (100 train / 1900 test)
    X = np.random.randn(2000, 2)
    # La vraie règle immuable (X0 + X1 > 0)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # 2. On crée la variable "PIÈGE"
    # Elle copie 'y' mais on introduit 15% d'erreurs aléatoires
    trap = y.copy()
    noise_indices = np.random.choice(2000, int(2000 * 0.15), replace=False)
    trap[noise_indices] = 1 - trap[noise_indices]
    
    # On ajoute cette colonne aux données (X devient donc : [X0, X1, Trap])
    X_with_trap = np.column_stack((X, trap))
    
    # Séparation 100 points pour l'entraînement (le piège est tentant !)
    X_train, y_train = X_with_trap[:100], y[:100]
    X_test, y_test = X_with_trap[100:], y[100:]
    
    models = {
        "🐍 Snake": FastAlgorithmeClassifier(n_layers=100),
        "🌲 RF": RandomForestClassifier(n_estimators=100)
    }

    print("🧪 TEST: RÉSILIENCE AU PIÈGE (15% Deception)")
    print("-" * 50)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"{name:<10} Accuracy sur le futur: {score:.4f}")

if __name__ == "__main__":
    test_deception_resilience()
