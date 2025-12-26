import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from algorithmeclassifier import FastAlgorithmeClassifier

def test_feature_interaction():
    # Création d'un problème XOR : l'interaction pure
    n_samples = 1000
    X = np.random.randn(n_samples, 10)
    # La cible dépend UNIQUEMENT de si X0 et X1 sont de même signe ou pas
    y = np.where(X[:, 0] * X[:, 1] > 0, 1, 0)
    
    models = {
        "🐍 Snake": FastAlgorithmeClassifier(n_layers=100),
        "🌲 RF": RandomForestClassifier(),
        "📈 LogReg": LogisticRegression()
    }

    print("🧪 TEST: INTERACTION LOGIQUE (XOR)")
    print("-" * 45)
    
    for name, model in models.items():
        model.fit(X, y)
        score = model.score(X, y)
        print(f"{name} Acc: {score:.4f}")

if __name__ == "__main__":
    test_feature_interaction()
