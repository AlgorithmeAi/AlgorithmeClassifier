import numpy as np
from sklearn.datasets import make_classification
from algorithmeclassifier import FastAlgorithmeClassifier
from sklearn.ensemble import RandomForestClassifier

def test_high_dimensionality():
    # 1000 lignes, 500 colonnes, mais seulement 2 sont importantes !
    X, y = make_classification(n_samples=1000, n_features=500, n_informative=2, random_state=42)
    
    models = {
        "🐍 Snake": FastAlgorithmeClassifier(n_layers=100),
        "🌲 RF": RandomForestClassifier()
    }

    print("🧪 TEST: AIGUILLE DANS UNE BOTTE DE FOIN (500 features)")
    print("-" * 45)
    
    for name, model in models.items():
        model.fit(X, y)
        score = model.score(X, y)
        print(f"{name} Acc: {score:.4f}")

if __name__ == "__main__":
    test_high_dimensionality()
