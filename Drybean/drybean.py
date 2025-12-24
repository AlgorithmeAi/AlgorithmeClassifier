"""
ULTIME BENCHMARK : Dry Bean Dataset (OpenML)
Le crash-test final pour la classification multiclasse complexe.
"""
import time
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

try:
    from algorithmeclassifier import AlgorithmeClassifier
    ALGO_AVAILABLE = True
except ImportError:
    ALGO_AVAILABLE = False

def run_dry_bean_final():
    print("🌱 Chargement du Dry Bean Dataset via OpenML (ID: 42477)...")
    # Chargement robuste
    data = fetch_openml(data_id=42477, as_frame=True, parser='auto')
    X, y = data.data, data.target
    
    le = LabelEncoder()
    y = le.fit_transform(y)

    # On monte à 2000 train samples pour donner de la matière
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=2000, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"✅ Setup : {X_train.shape[0]} Train / {len(X_test)} Test / 7 Classes")

    models = [
        ("RF (300 trees)", RandomForestClassifier(n_estimators=300, random_state=42)),
        ("Hist-GBM", HistGradientBoostingClassifier(max_iter=200, random_state=42))
    ]
    if ALGO_AVAILABLE:
        models.append(("AlgorithmeClassifier", AlgorithmeClassifier(n_layers=100)))

    results = []
    print("\n" + "═"*85)
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
        
        results.append({"Model": name, "Acc": acc, "F1": f1, "Loss": loss, "Time": t_exec})
        print(f"{name:<22} | {acc:.4f} | {f1:.4f}   | {loss:.4f}   | {t_exec:.3f}s")

    # Petit bonus : vérifier la confusion sur les classes difficiles
    print("\n🧐 Analyse : Les haricots sont beaucoup plus durs que l'acier !")

if __name__ == "__main__":
    run_dry_bean_final()
