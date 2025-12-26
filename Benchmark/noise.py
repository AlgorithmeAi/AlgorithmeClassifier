import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from algorithmeclassifier import FastAlgorithmeClassifier

def test_noise_robustness():
    n_samples = 1000
    n_useful = 5
    noise_levels = [0, 10, 20, 30, 40, 50] # Nombre de colonnes de bruit ajoutées
    
    results = { "Snake": [], "RF": [], "GB": [] }

    for n_noise in noise_levels:
        # Création du dataset : 5 colonnes utiles + n_noise colonnes de bruit
        X, y = make_classification(
            n_samples=n_samples, 
            n_features=n_useful + n_noise, 
            n_informative=n_useful, 
            n_redundant=0, 
            random_state=42
        )
        
        models = {
            "Snake": FastAlgorithmeClassifier(n_layers=100),
            "RF": RandomForestClassifier(n_estimators=100),
            "GB": HistGradientBoostingClassifier()
        }
        
        for name, model in models.items():
            score = cross_val_score(model, X, y, cv=3).mean()
            results[name].append(score)
            print(f"Bruit: {n_noise} features | {name}: {score:.4f}")

    # Visualisation
    plt.figure(figsize=(10, 6))
    for name, scores in results.items():
        plt.plot(noise_levels, scores, 'o-', label=name)
    
    plt.title("Résistance au Bruit : Snake vs GB vs RF")
    plt.xlabel("Nombre de variables 'bruit' ajoutées")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("noise_robustness.png")
    plt.show()

if __name__ == "__main__":
    test_noise_robustness()
