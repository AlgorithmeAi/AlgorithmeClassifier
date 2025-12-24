"""
Benchmark Final : Satimage (Statlog)
Cible : Robustesse aux données multispectrales
"""
import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, cohen_kappa_score
import warnings

warnings.filterwarnings('ignore')

try:
    from algorithmeclassifier import AlgorithmeClassifier
    ALGO_AVAILABLE = True
except ImportError:
    ALGO_AVAILABLE = False

def run_satellite_benchmark():
    print("🛰️  Récupération des données Satellite (Statlog)...")
    # Satellite dataset (ID 182 sur OpenML)
    data = fetch_openml(data_id=182, as_frame=True, parser='auto')
    X, y = data.data, data.target
    
    # Encodage propre des labels
    from sklearn.preprocessing import LabelEncoder
    y = LabelEncoder().fit_transform(y)

    # Split : 1000 Train / Reste en Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=1000, random_state=42, stratify=y
    )

    # Utilisation de MinMaxScaler (souvent meilleur pour les données spectrales)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"✅ Prêt : Train=1000 | Test={len(X_test)} | Features={X.shape[1]}")

    models = [
        ("RF (200 trees)", RandomForestClassifier(n_estimators=200, random_state=42)),
        ("Hist-GBM", HistGradientBoostingClassifier(max_iter=150, random_state=42))
    ]

    if ALGO_AVAILABLE:
        models.append(("AlgorithmeClassifier", AlgorithmeClassifier(n_layers=100)))

    results = []

    print("\n" + "═"*80)
    print(f"{'MODÈLE':<22} | {'ACC':<8} | {'KAPPA':<8} | {'LOGLOSS':<10} | {'TEMPS'}")
    print("─"*80)

    for name, model in models:
        t_0 = time.perf_counter()
        model.fit(X_train, y_train)
        t_train = time.perf_counter() - t_0
        
        preds = model.predict(X_test)
        probas = model.predict_proba(X_test)
        
        acc = accuracy_score(y_test, preds)
        kappa = cohen_kappa_score(y_test, preds) # Robustesse statistique
        loss = log_loss(y_test, probas)
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Kappa": kappa,
            "LogLoss": loss,
            "Time": t_train
        })
        
        print(f"{name:<22} | {acc:.4f} | {kappa:.4f} | {loss:.4f}   | {t_train:.3f}s")

    print("\n🏆 CLASSEMENT PAR SCORE KAPPA (Fiabilité) :")
    df = pd.DataFrame(results).sort_values(by="Kappa", ascending=False)
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_satellite_benchmark()
