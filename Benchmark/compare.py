import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from algorithmeclassifier import FastAlgorithmeClassifier

def compare_boundaries():
    X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
    
    models = {
        "🐍 Snake": FastAlgorithmeClassifier(n_layers=100),
        "🌲 Random Forest": RandomForestClassifier(n_estimators=100),
        "🔥 Gradient Boosting": HistGradientBoostingClassifier()
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    for ax, (name, model) in zip(axes, models.items()):
        model.fit(X, y)
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', s=40, cmap='RdYlBu')
        ax.set_title(f"{name}\nAcc: {model.score(X, y):.3f}")
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("battle_of_the_models.png")
    plt.show()

if __name__ == "__main__":
    compare_boundaries()
