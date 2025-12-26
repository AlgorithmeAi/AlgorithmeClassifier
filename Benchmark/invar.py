import numpy as np
from sklearn.datasets import make_moons
from algorithmeclassifier import FastAlgorithmeClassifier

def test_spatial_invariance():
    # 1. On crée des lunes
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
    X_train, y_train = X[:100], y[:100] # Petit entraînement
    
    # 2. On crée un set de test qui a subi une ROTATION et une TRANSLATION
    X_test_orig, y_test = X[100:], y[100:]
    
    theta = np.radians(45)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    
    # Rotation + Décalage de +5.0 sur les deux axes
    X_test_transformed = X_test_orig.dot(R) + 5.0
    
    model = FastAlgorithmeClassifier(n_layers=100)
    model.fit(X_train, y_train)
    
    # On teste sur le set d'origine vs le set transformé
    score_orig = model.score(X_test_orig, y_test)
    
    print("🧪 TEST: INVARIANCE SPATIALE")
    print("-" * 45)
    print(f"Accuracy sur données normales : {score_orig:.4f}")
    
    try:
        score_trans = model.score(X_test_transformed, y_test)
        print(f"Accuracy après Rotation/Translation : {score_trans:.4f}")
        drop = (score_orig - score_trans) * 100
        print(f"📉 Chute de performance : {drop:.2f}%")
    except:
        print("❌ Le modèle n'a pas supporté le changement de repère.")

if __name__ == "__main__":
    test_spatial_invariance()
