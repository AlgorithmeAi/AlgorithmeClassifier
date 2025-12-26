import numpy as np
import pandas as pd
from time import time
from sklearn.datasets import load_breast_cancer, load_wine, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# Import des modèles
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
# Ton algorithme
try:
    from algorithmeclassifier import FastAlgorithmeClassifier
except ImportError:
    print("❌ Erreur : Assure-toi que 'algorithmeclassifier.py' est dans le même dossier.")

def run_comparison():
    # 1. Préparation des Datasets
    datasets = {
        "Breast Cancer": load_breast_cancer(return_X_y=True),
        "Wine": load_wine(return_X_y=True),
        "Synthetic (10k)": make_classification(n_samples=10000, n_features=20, n_informative=15, random_state=42)
    }

    # 2. Définition des modèles
    models = {
        "🐍 Snake": FastAlgorithmeClassifier(n_layers=100),
        "🌲 Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "🔥 Grad. Boosting": HistGradientBoostingClassifier(random_state=42),
        "📈 Log. Regression": LogisticRegression(max_iter=1000),
        "📏 Lin. Classifier": RidgeClassifier() # Equivalent LinReg pour classification
    }

    results = []

    print(f"{'Dataset':<20} | {'Model':<18} | {'Acc':<8} | {'F1':<8} | {'Time':<8}")
    print("-" * 75)

    for ds_name, (X, y) in datasets.items():
        # Prétraitement
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        for name, model in models.items():
            start_time = time()
            
            # Entraînement et prédiction
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            duration = time() - start_time
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='macro')

            print(f"{ds_name:<20} | {name:<18} | {acc:.4f} | {f1:.4f} | {duration:.2f}s")
            
            results.append({
                "Dataset": ds_name,
                "Model": name,
                "Accuracy": acc,
                "F1": f1,
                "Time": duration
            })

    # Affichage du gagnant par dataset
    df = pd.DataFrame(results)
    print("\n🏆 RÉSUMÉ DES PERFORMANCES :")
    summary = df.groupby('Model')[['Accuracy', 'F1', 'Time']].mean().sort_values(by='Accuracy', ascending=False)
    print(summary)

if __name__ == "__main__":
    run_comparison()
