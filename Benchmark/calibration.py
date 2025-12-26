import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from algorithmeclassifier import FastAlgorithmeClassifier

def plot_calibration_curve():
    # Dataset binaire pour la calibration
    X, y = make_classification(n_samples=5000, n_features=20, n_informative=10, random_state=42)
    
    models = {
        "Snake": FastAlgorithmeClassifier(n_layers=100),
        "LogReg": LogisticRegression(),
        "GradBoosting": HistGradientBoostingClassifier()
    }

    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], "k:", label="Parfaitement calibré")

    for name, model in models.items():
        model.fit(X, y)
        # On récupère les probas de la classe 1
        prob_pos = model.predict_proba(X)[:, 1]
        
        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y, prob_pos, n_bins=10)

        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=name)

    plt.ylabel("Fraction réelle de positifs")
    plt.xlabel("Probabilité moyenne prédite")
    plt.title("Courbe de Calibration : Snake vs Others")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("calibration_snake.png")
    plt.show()

if __name__ == "__main__":
    plot_calibration_curve()
