import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from algorithmeclassifier import FastAlgorithmeClassifier
from sklearn.linear_model import LogisticRegression

def test_geometry_intelligence():
    # Un cercle dans un cercle : impossible à séparer avec une ligne droite !
    X, y = make_circles(n_samples=500, noise=0.05, factor=0.5, random_state=42)
    
    models = {
        "🐍 Snake": FastAlgorithmeClassifier(n_layers=100),
        "📈 LogReg": LogisticRegression()
    }

    print("🧪 TEST: INTELLIGENCE GÉOMÉTRIQUE (Circles)")
    print("-" * 45)
    
    for name, model in models.items():
        model.fit(X, y)
        score = model.score(X, y)
        print(f"{name} Acc: {score:.4f}")

if __name__ == "__main__":
    test_geometry_intelligence()
