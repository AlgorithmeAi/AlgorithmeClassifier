import numpy as np
from sklearn.datasets import make_classification
from algorithmeclassifier import FastAlgorithmeClassifier

def test_redundancy_resilience():
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=42)
    
    # On ajoute 10 colonnes qui sont juste des copies des 10 premières !
    X_redundant = np.hstack((X, X)) 
    
    model = FastAlgorithmeClassifier(n_layers=100)
    
    print("🧪 TEST: RÉSILIENCE À LA REDONDANCE (20 features dont 10 clones)")
    print("-" * 55)
    
    model.fit(X, y)
    score_normal = model.score(X, y)
    
    model.fit(X_redundant, y)
    score_redundant = model.score(X_redundant, y)
    
    print(f"Accuracy (Données normales) : {score_normal:.4f}")
    print(f"Accuracy (Données redondantes) : {score_redundant:.4f}")
    print(f"Différence : {abs(score_normal - score_redundant):.4f}")

if __name__ == "__main__":
    test_redundancy_resilience()
