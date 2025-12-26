import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
from algorithmeclassifier import FastAlgorithmeClassifier

def plot_learning_comparison():
    # Génération d'un dataset assez grand pour voir une évolution
    X, y = make_classification(n_samples=5000, n_features=25, n_informative=20, random_state=42)
    
    models = {
        "🐍 Snake": FastAlgorithmeClassifier(n_layers=100),
        "🔥 Gradient Boosting": HistGradientBoostingClassifier(),
        "📏 Linear Classifier": RidgeClassifier()
    }

    plt.figure(figsize=(12, 7))
    
    # Tailles d'entraînement : de 10% à 100% du dataset
    train_sizes = np.linspace(0.1, 1.0, 10)

    for name, model in models.items():
        print(f"Calcul pour {name}...")
        
        # On calcule les scores (moyenne sur 3-fold pour la rapidité)
        train_sizes_abs, train_scores, test_scores = learning_curve(
            model, X, y, train_sizes=train_sizes, cv=3, n_jobs=-1, scoring='accuracy'
        )
        
        # Calcul des moyennes et écarts-types
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # Tracé de la ligne et de la zone d'ombre pour la variance
        plt.plot(train_sizes_abs, test_mean, 'o-', label=name)
        plt.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, alpha=0.1)

    plt.title("Efficacité Data : Snake vs GB vs Linear", fontsize=14)
    plt.xlabel("Nombre d'échantillons d'entraînement", fontsize=12)
    plt.ylabel("Accuracy sur Test Set", fontsize=12)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Sauvegarde pour ton GitHub
    plt.savefig("learning_curve.png")
    print("\n✅ Graphique sauvegardé sous 'learning_curve.png'")
    plt.show()

if __name__ == "__main__":
    plot_learning_comparison()
