"""
Benchmark Rigoureux : Forest Cover Type
Compare AlgorithmeClassifier vs RF vs Hist-GBM

Dataset: Forest CoverType (échantillonné)
- 1000 train samples (exigence utilisateur)
- 54 features (géologie + hydrologie)
- 7 classes forestières
"""
import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, log_loss
import warnings

warnings.filterwarnings('ignore')

# Import de ton modèle
try:
    from algorithmeclassifier import AlgorithmeClassifier
    ALGO_AVAILABLE = True
except ImportError:
    print("⚠️ AlgorithmeClassifier non trouvé.")
    ALGO_AVAILABLE = False

def run_forest_benchmark():
    print("📥 Chargement du dataset Forest CoverType (échantillonnage)...")
    # On récupère le dataset (très lourd, fetch_covtype gère le cache localement)
    data = fetch_covtype()
    X, y = data.data, data.target
    
    # Pour la classification scikit-learn, on préfère souvent les labels commençant à 0
    y = y - 1 

    # Split spécifique : 1000 pour le train, 4000 pour le test (pour avoir un test solide)
    # On utilise stratify pour garder la distribution des classes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=1000, test_size=4000, random_state=42, stratify=y
    )

    # Standardisation : Cruciale ici car les features vont de 0 (binary) à 7000 (altitude)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"📊 Données prêtes : Train={X_train.shape[0]}, Test={X_test.shape[0]}, Features={X_train.shape[1]}")

    models = [
        ("RF (100 trees)", RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)),
        ("Hist-GBM", HistGradientBoostingClassifier(max_iter=100, random_state=42))
    ]

    if ALGO_AVAILABLE:
        # Configuration suggérée pour 54 features
        models.append(("AlgorithmeClassifier", AlgorithmeClassifier(n_layers=100)))

    results = []

    print("\n" + "═"*75)
    print(f"{'MODÈLE':<22} | {'ACC':<8} | {'F1-MACRO':<10} | {'LOGLOSS':<10} | {'TEMPS'}")
    print("─"*75)

    for name, model in models:
        # Training
        t_start = time.perf_counter()
        model.fit(X_train, y_train)
        t_train = time.perf_counter() - t_start
        
        # Inférence
        preds = model.predict(X_test)
        probas = model.predict_proba(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='macro')
        loss = log_loss(y_test, probas)
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "F1-Macro": f1,
            "LogLoss": loss,
            "Time": t_train
        })
        
        print(f"{name:<22} | {acc:.4f} | {f1:.4f}   | {loss:.4f}   | {t_train:.3f}s")

    # Conclusion
    df_res = pd.DataFrame(results).sort_values(by="F1-Macro", ascending=False)
    print("\n🏆 BILAN DES PERFORMANCES :")
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    run_forest_benchmark()
