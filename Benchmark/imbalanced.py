import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score
from algorithmeclassifier import FastAlgorithmeClassifier
from sklearn.ensemble import RandomForestClassifier

def test_imbalance_resilience():
    # 1000 échantillons, mais seulement 5% de la classe 1 (Minoritaire)
    X, y = make_classification(n_samples=1000, n_features=20, 
                           weights=[0.95, 0.05], random_state=42)
    
    models = {
        "🐍 Snake": FastAlgorithmeClassifier(n_layers=100),
        "🌲 RF": RandomForestClassifier()
    }

    print("🧪 TEST: RÉSILIENCE AU DÉSÉQUILIBRE (5% vs 95%)")
    print("-" * 45)
    
    for name, model in models.items():
        model.fit(X, y)
        y_pred = model.predict(X)
        # On utilise le F1-Score car l'Accuracy est trompeuse ici
        score = f1_score(y, y_pred)
        print(f"{name} F1-Score: {score:.4f}")

if __name__ == "__main__":
    test_imbalance_resilience()
