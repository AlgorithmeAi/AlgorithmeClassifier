"""
Benchmark : Steel Plates Faults (Version Corrigée)
Cible : Classification multi-classe (7 types de défauts)
"""
import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
import warnings

warnings.filterwarnings('ignore')

try:
    from algorithmeclassifier import AlgorithmeClassifier
    ALGO_AVAILABLE = True
except ImportError:
    ALGO_AVAILABLE = False

def run_steel_benchmark():
    print("🏗️  Chargement du Steel Plates Faults Dataset...")
    # OpenML ID 1504
    data = fetch_openml(data_id=1504, as_frame=True, parser='auto')
    X = data.data
    y = data.target
    
    # --- CORRECTION : Encodage explicite en entiers ---
    le = LabelEncoder()
    y = le.fit_transform(y) 
    # --------------------------------------------------
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=1200, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = [
        ("RF (Gini)", RandomForestClassifier(n_estimators=150, random_state=42)),
        ("Hist-GBM", HistGradientBoostingClassifier(max_iter=100, random_state=42))
    ]
    if ALGO_AVAILABLE:
        models.append(("AlgorithmeClassifier", AlgorithmeClassifier(n_layers=100)))

    print(f"\n🚀 Test sur {len(X_train)} train samples / {len(X_test)} test samples")
    print("─"*85)
    print(f"{'MODÈLE':<22} | {'ACC':<8} | {'F1-MACRO':<10} | {'LOGLOSS':<10} | {'TEMPS'}")
    print("─"*85)

    for name, model in models:
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        t_exec = time.perf_counter() - t0
        
        preds = model.predict(X_test)
        probas = model.predict_proba(X_test)
        
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='macro')
        loss = log_loss(y_test, probas)
        
        print(f"{name:<22} | {acc:.4f} | {f1:.4f}   | {loss:.4f}   | {t_exec:.3f}s")

if __name__ == "__main__":
    run_steel_benchmark()
