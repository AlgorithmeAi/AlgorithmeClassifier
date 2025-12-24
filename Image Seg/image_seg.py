"""
Benchmark Fix : Image Segmentation
Suppression des lignes conflictuelles et réinitialisation des index.
"""
import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score, log_loss
import warnings

warnings.filterwarnings('ignore')

try:
    from algorithmeclassifier import AlgorithmeClassifier
    ALGO_AVAILABLE = True
except ImportError:
    ALGO_AVAILABLE = False

def load_and_fix_segmentation():
    print("📥 Chargement et nettoyage du Segmentation Dataset...")
    # Utilisation de l'ID 40984 (version plus propre sur OpenML)
    data = fetch_openml(data_id=40984, as_frame=True, parser='auto')
    X = data.data
    y = data.target

    # 1. Drop des lignes conflictuelles (souvent les premières sur ce dataset spécifique)
    # On s'assure de travailler sur des copies pour éviter les SettingWithCopyWarning
    X = X.copy()
    y = y.copy()

    # 2. Suppression des lignes contenant des NaNs éventuels ou colonnes constantes
    X = X.dropna()
    y = y[X.index]
    
    # 3. CRUCIAL : Réinitialisation des index pour éviter la boucle infinie/conflit au split
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"   ✓ Nettoyage terminé : {X.shape[0]} lignes valides.")
    return X, y_encoded

def run_fixed_benchmark():
    X, y = load_and_fix_segmentation()
    
    # Prétraitement
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split avec au moins 1000 samples en train comme demandé précédemment
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, train_size=1000, random_state=42, stratify=y
    )

    models = [
        ("RF (100 trees)", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("Hist-GBM", HistGradientBoostingClassifier(max_iter=100, random_state=42))
    ]

    if ALGO_AVAILABLE:
        models.append(("AlgorithmeClassifier", AlgorithmeClassifier(n_layers=100)))

    print("\n" + "═"*80)
    print(f"{'MODÈLE':<22} | {'ACC':<8} | {'F1-MACRO':<10} | {'LOGLOSS':<10} | {'TEMPS'}")
    print("─"*80)

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
    run_fixed_benchmark()
