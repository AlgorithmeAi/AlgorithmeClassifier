import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from algorithmeclassifier import FastAlgorithmeClassifier

def test_few_shot():
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # On teste de 5 échantillons par classe à 50
    samples_per_class = [2, 5, 10, 20, 50]
    results = { "Snake": [], "RF": [], "GB": [] }

    for n in samples_per_class:
        # On crée un petit train set équilibré
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=n*10, stratify=y, random_state=42
        )
        
        models = {
            "Snake": FastAlgorithmeClassifier(n_layers=100),
            "RF": RandomForestClassifier(n_estimators=100),
            "GB": HistGradientBoostingClassifier()
        }
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            results[name].append(score)
            print(f"Echantillons/classe: {n} | {name}: {score:.4f}")

    # Graphique
    plt.figure(figsize=(10, 6))
    for name, scores in results.items():
        plt.plot(samples_per_class, scores, 'o-', label=name)
    
    plt.title("Performance en Few-Shot Learning (Digits dataset)")
    plt.xlabel("Nombre d'échantillons par classe")
    plt.ylabel("Accuracy sur Test Set (fixe)")
    plt.legend()
    plt.grid(True)
    plt.savefig("few_shot_test.png")
    plt.show()

if __name__ == "__main__":
    test_few_shot()
