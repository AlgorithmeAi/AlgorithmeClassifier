"""
ULTIMES BENCHMARKS (9 & 10) : Mice Protein & OptDigits
Objectif : Haute dimensionnalité et reconnaissance de patterns spatiaux.
"""
import time
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, f1_score
import warnings

warnings.filterwarnings('ignore')

try:
    from algorithmeclassifier import AlgorithmeClassifier
    ALGO_AVAILABLE = True
except ImportError:
    ALGO_AVAILABLE = False

def run_benchmark(data_id, name, train_size=1000):
    print(f"\n{name} (OpenML ID: {data_id})...")
    data = fetch_openml(data_id=data_id, as_frame=True, parser='auto')
    X, y = data.data, data.target
    
    # Nettoyage des NaNs (spécifique au dataset Mice Protein)
    if X.isnull().values.any():
        X = X.fillna(X.mean())
    
    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"✅ Ready: {X_train.shape[1]} Features | {len(np.unique(y))} Classes")

    models = [
        ("RF (200 trees)", RandomForestClassifier(n_estimators=200, random_state=42)),
        ("Hist-GBM", HistGradientBoostingClassifier(max_iter=100, random_state=42))
    ]
    if ALGO_AVAILABLE:
        models.append(("AlgorithmeClassifier", AlgorithmeClassifier(n_layers=100)))

    print("─"*85)
    print(f"{'MODÈLE':<22} | {'ACC':<8} | {'F1-MACRO':<10} | {'LOGLOSS':<10} | {'TEMPS'}")
    print("─"*85)

    for m_name, model in models:
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        t_exec = time.perf_counter() - t0
        
        preds = model.predict(X_test)
        probas = model.predict_proba(X_test)
        
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='macro')
        loss = log_loss(y_test, probas)
        
        print(f"{m_name:<22} | {acc:.4f} | {f1:.4f}   | {loss:.4f}   | {t_exec:.3f}s")

if __name__ == "__main__":
    # BENCHMARK 9 : Mice Protein Expression (Bio-médical)
    #run_benchmark(data_id=40966, name="🧬 BENCHMARK 9: Mice Protein")

    # BENCHMARK 10 : Optical Recognition of Handwritten Digits
    run_benchmark(data_id=28, name="🔢 BENCHMARK 10: OptDigits", train_size=1500)
