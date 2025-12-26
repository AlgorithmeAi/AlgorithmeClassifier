import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from algorithmeclassifier import FastAlgorithmeClassifier

def test_inference_speed():
    X, y = make_classification(n_samples=5000, n_features=20, random_state=42)
    
    models = {
        "Snake": FastAlgorithmeClassifier(n_layers=100),
        "RF": RandomForestClassifier(n_estimators=100),
        "GB": HistGradientBoostingClassifier()
    }

    # Entraînement préalable
    for name, model in models.items():
        model.fit(X, y)

    batch_sizes = [1, 10, 100, 1000, 5000]
    results = {name: [] for name in models}

    print("📊 MESURE DE LA LATENCE D'INFÉRENCE")
    print("-" * 45)

    for size in batch_sizes:
        X_test = X[:size]
        for name, model in models.items():
            start = time.perf_counter()
            _ = model.predict(X_test)
            end = time.perf_counter()
            
            latency = (end - start) / size # Temps par échantillon
            results[name].append(latency)
        print(f"Batch Size: {size:<5} | Snake: {results['Snake'][-1]:.6f}s/obs")

    # Graphique
    plt.figure(figsize=(10, 6))
    for name, latencies in results.items():
        plt.plot(batch_sizes, latencies, 'o-', label=name)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Latence de prédiction par échantillon (Log-Log scale)")
    plt.xlabel("Taille du Batch (nombre d'échantillons)")
    plt.ylabel("Temps par prédiction (secondes)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig("inference_speed.png")
    plt.show()

if __name__ == "__main__":
    test_inference_speed()
