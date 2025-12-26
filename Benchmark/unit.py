import numpy as np
import pandas as pd
from time import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import (fetch_covtype, fetch_openml, load_breast_cancer, 
                              load_wine, make_classification)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, f1_score, log_loss, 
                             classification_report, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import sys
import os

# Rediriger stderr temporairement
original_stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

sys.stderr.close()
sys.stderr = original_stderr

from algorithmeclassifier import FastAlgorithmeClassifier

def print_header(text, char="═"):
    print(f"\n{char * 80}")
    print(f"{text.center(80)}")
    print(f"{char * 80}")

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds//60:.0f}m {seconds%60:.0f}s"
    else:
        return f"{seconds//3600:.0f}h {(seconds%3600)//60:.0f}m"

def apply_pca(X_train, X_test, n_components=20):
    """Applique PCA pour réduire à n_components features"""
    if X_train.shape[1] <= n_components:
        return X_train, X_test
    
    print(f"   → PCA: {X_train.shape[1]} features → {n_components} composantes", end="")
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(f" (variance expliquée: {pca.explained_variance_ratio_.sum():.2%})")
    
    return X_train_pca, X_test_pca

class BenchmarkRunner:
    def __init__(self, X_train, X_test, y_train, y_test, dataset_name):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.dataset_name = dataset_name
        self.results = []
        
    def evaluate_model(self, name, model, fit_params=None):
        """Évalue un modèle et retourne les métriques"""
        print(f"🔄 {name}...", end=" ", flush=True)
        
        if fit_params is None:
            fit_params = {}
        
        try:
            # Fit
            t_start = time()
            model.fit(self.X_train, self.y_train, **fit_params)
            t_fit = time() - t_start
            
            # Predict
            t_start = time()
            y_pred = model.predict(self.X_test)
            t_pred = time() - t_start
            
            # Predict proba
            try:
                y_pred_proba = model.predict_proba(self.X_test)
                logloss = log_loss(self.y_test, y_pred_proba)
            except:
                logloss = np.nan
            
            # Métriques
            accuracy = accuracy_score(self.y_test, y_pred)
            f1_macro = f1_score(self.y_test, y_pred, average='macro', zero_division=0)
            
            result = {
                'Model': name,
                'Accuracy': accuracy,
                'F1-Macro': f1_macro,
                'LogLoss': logloss,
                'Fit Time': t_fit,
                'Predict Time': t_pred,
            }
            
            self.results.append(result)
            print(f"✓ Acc:{accuracy:.3f} F1:{f1_macro:.3f} ({format_time(t_fit)})")
            
            return result
        except Exception as e:
            print(f"✗ Error: {str(e)[:50]}")
            return None
    
    def run_benchmark(self, models_config):
        """Exécute le benchmark avec une config de modèles"""
        print_header(f"🎯 {self.dataset_name}")
        print(f"📦 Train: {self.X_train.shape[0]}, Test: {self.X_test.shape[0]}, Features: {self.X_train.shape[1]}, Classes: {len(np.unique(self.y_train))}")
        
        for name, model in models_config:
            self.evaluate_model(name, model)
        
        return pd.DataFrame(self.results)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: BENCHMARK STANDARD (baseline)
# ═══════════════════════════════════════════════════════════════════════════════

def test_1_standard_benchmark():
    print_header("🧪 TEST 1: BENCHMARK STANDARD")
    
    def load_fashion_mnist():
        print("📥 Fashion-MNIST...")
        fashion = fetch_openml('Fashion-MNIST', version=1, parser='auto')
        X, y = fashion.data.to_numpy(), fashion.target.astype(int).to_numpy()
        indices = np.random.choice(len(X), 3000, replace=False)
        X_train, X_test, y_train, y_test = train_test_split(
            X[indices], y[indices], train_size=2000, random_state=42
        )
        return apply_pca(X_train, X_test, 20)
    
    X_train, X_test, y_train, y_test = load_fashion_mnist()
    
    models = [
        ("Snake-100", FastAlgorithmeClassifier(n_layers=100, vocal=False, n_jobs=-1)),
        ("LogReg", LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42)),
        ("RF-100", RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)),
        ("HGB", HistGradientBoostingClassifier(max_iter=100, random_state=42)),
    ]
    
    benchmark = BenchmarkRunner(X_train, X_test, y_train, y_test, "Fashion-MNIST")
    df = benchmark.run_benchmark(models)
    
    print("\n📊 RÉSULTATS:")
    print(df.to_string(index=False))
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: ROBUSTESSE AU BRUIT
# ═══════════════════════════════════════════════════════════════════════════════

def test_2_noise_robustness():
    print_header("🧪 TEST 2: ROBUSTESSE AU BRUIT")
    
    # Dataset de base
    print("📥 Génération dataset...")
    X, y = make_classification(n_samples=3000, n_features=20, n_informative=15,
                               n_classes=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=2000, random_state=42)
    
    results_all = []
    
    for noise_level in [0.0, 0.05, 0.10, 0.20, 0.30]:
        print(f"\n🔊 Niveau de bruit: {noise_level*100:.0f}%")
        
        # Ajouter du bruit gaussien
        X_train_noisy = X_train + np.random.normal(0, noise_level, X_train.shape)
        X_test_noisy = X_test + np.random.normal(0, noise_level, X_test.shape)
        
        models = [
            ("Snake-100", FastAlgorithmeClassifier(n_layers=100, vocal=False, n_jobs=-1)),
            ("LogReg", LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42)),
            ("RF-100", RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)),
        ]
        
        benchmark = BenchmarkRunner(X_train_noisy, X_test_noisy, y_train, y_test, 
                                   f"Noise {noise_level*100:.0f}%")
        df = benchmark.run_benchmark(models)
        df['Noise Level'] = noise_level
        results_all.append(df)
    
    df_final = pd.concat(results_all, ignore_index=True)
    print("\n📊 RÉSULTATS GLOBAUX:")
    print(df_final.to_string(index=False))
    return df_final


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: FEW-SHOT LEARNING
# ═══════════════════════════════════════════════════════════════════════════════

def test_3_few_shot():
    print_header("🧪 TEST 3: FEW-SHOT LEARNING")
    
    print("📥 Fashion-MNIST...")
    fashion = fetch_openml('Fashion-MNIST', version=1, parser='auto')
    X, y = fashion.data.to_numpy(), fashion.target.astype(int).to_numpy()
    indices = np.random.choice(len(X), 2000, replace=False)
    X_full, y_full = X[indices], y[indices]
    X_full, _ = apply_pca(X_full, X_full, 20)
    
    results_all = []
    
    for n_train in [100, 250, 500, 1000]:
        print(f"\n📏 Taille train: {n_train}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full, train_size=n_train, random_state=42
        )
        
        models = [
            ("Snake-50", FastAlgorithmeClassifier(n_layers=50, vocal=False, n_jobs=-1)),
            ("LogReg", LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42)),
            ("RF-100", RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)),
        ]
        
        benchmark = BenchmarkRunner(X_train, X_test, y_train, y_test, f"Train={n_train}")
        df = benchmark.run_benchmark(models)
        df['Train Size'] = n_train
        results_all.append(df)
    
    df_final = pd.concat(results_all, ignore_index=True)
    print("\n📊 RÉSULTATS GLOBAUX:")
    print(df_final.to_string(index=False))
    return df_final


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4: SCALABILITÉ n_layers
# ═══════════════════════════════════════════════════════════════════════════════

def test_4_scalability():
    print_header("🧪 TEST 4: SCALABILITÉ n_layers")
    
    print("📥 Fashion-MNIST...")
    fashion = fetch_openml('Fashion-MNIST', version=1, parser='auto')
    X, y = fashion.data.to_numpy(), fashion.target.astype(int).to_numpy()
    indices = np.random.choice(len(X), 3000, replace=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X[indices], y[indices], train_size=2000, random_state=42
    )
    X_train, X_test = apply_pca(X_train, X_test, 20)
    
    results_all = []
    
    for n_layers in [10, 25, 50, 100, 200]:
        print(f"\n🔢 n_layers: {n_layers}")
        
        models = [
            (f"Snake-{n_layers}", FastAlgorithmeClassifier(n_layers=n_layers, vocal=False, n_jobs=-1)),
        ]
        
        benchmark = BenchmarkRunner(X_train, X_test, y_train, y_test, f"Layers={n_layers}")
        df = benchmark.run_benchmark(models)
        df['n_layers'] = n_layers
        
        # Train accuracy pour overfitting check
        snake = FastAlgorithmeClassifier(n_layers=n_layers, vocal=False, n_jobs=-1)
        snake.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, snake.predict(X_train))
        df['Train Accuracy'] = train_acc
        df['Overfit Gap'] = train_acc - df['Accuracy'].iloc[0]
        
        results_all.append(df)
    
    df_final = pd.concat(results_all, ignore_index=True)
    print("\n📊 RÉSULTATS GLOBAUX:")
    print(df_final.to_string(index=False))
    return df_final


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5: DONNÉES DÉSÉQUILIBRÉES
# ═══════════════════════════════════════════════════════════════════════════════

def test_5_imbalanced():
    print_header("🧪 TEST 5: DONNÉES DÉSÉQUILIBRÉES")
    
    results_all = []
    
    for imbalance_ratio in [1.0, 0.1, 0.01]:
        print(f"\n⚖️  Ratio déséquilibre: 1:{1/imbalance_ratio:.0f}")
        
        # Créer dataset déséquilibré
        weights = [imbalance_ratio if i == 0 else 1.0 for i in range(5)]
        X, y = make_classification(
            n_samples=3000, n_features=20, n_informative=15,
            n_classes=5, weights=weights, random_state=42
        )
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=2000, random_state=42)
        
        print(f"   Distribution train: {np.bincount(y_train)}")
        
        models = [
            ("Snake-100", FastAlgorithmeClassifier(n_layers=100, vocal=False, n_jobs=-1)),
            ("LogReg", LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42, class_weight='balanced')),
            ("RF-100", RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, class_weight='balanced')),
        ]
        
        benchmark = BenchmarkRunner(X_train, X_test, y_train, y_test, f"Imbalance 1:{1/imbalance_ratio:.0f}")
        df = benchmark.run_benchmark(models)
        df['Imbalance Ratio'] = imbalance_ratio
        results_all.append(df)
    
    df_final = pd.concat(results_all, ignore_index=True)
    print("\n📊 RÉSULTATS GLOBAUX:")
    print(df_final.to_string(index=False))
    return df_final


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 6: DATASETS CRITIQUES (Médical/Financier)
# ═══════════════════════════════════════════════════════════════════════════════

def test_6_critical_datasets():
    print_header("🧪 TEST 6: DATASETS CRITIQUES (Explicabilité importante)")
    
    results_all = []
    
    # Dataset 1: Breast Cancer
    print("\n🏥 Breast Cancer...")
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
    X_train, X_test = apply_pca(X_train, X_test, 20)
    
    models = [
        ("Snake-100", FastAlgorithmeClassifier(n_layers=100, vocal=False, n_jobs=-1)),
        ("LogReg", LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42)),
        ("RF-100", RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)),
    ]
    
    benchmark = BenchmarkRunner(X_train, X_test, y_train, y_test, "Breast Cancer")
    df = benchmark.run_benchmark(models)
    df['Dataset'] = 'Breast Cancer'
    results_all.append(df)
    
    # Dataset 2: Wine Quality
    print("\n🍷 Wine Quality...")
    wine = load_wine()
    X, y = wine.data, wine.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
    X_train, X_test = apply_pca(X_train, X_test, 13)  # Wine a déjà 13 features
    
    benchmark = BenchmarkRunner(X_train, X_test, y_train, y_test, "Wine Quality")
    df = benchmark.run_benchmark(models)
    df['Dataset'] = 'Wine Quality'
    results_all.append(df)
    
    # Dataset 3: Credit Default (synthétique pour simulation)
    print("\n💳 Credit Default (synthétique)...")
    X, y = make_classification(
        n_samples=2000, n_features=20, n_informative=15,
        n_classes=2, weights=[0.7, 0.3], random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
    
    benchmark = BenchmarkRunner(X_train, X_test, y_train, y_test, "Credit Default")
    df = benchmark.run_benchmark(models)
    df['Dataset'] = 'Credit Default'
    results_all.append(df)
    
    df_final = pd.concat(results_all, ignore_index=True)
    print("\n📊 RÉSULTATS GLOBAUX:")
    print(df_final.to_string(index=False))
    return df_final


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print_header("🐍 SNAKE MEGA-BENCHMARK SUITE COMPLÈTE 🐍")
    
    all_results = {}
    
    try:
        all_results['test1_standard'] = test_1_standard_benchmark()
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
    
    try:
        all_results['test2_noise'] = test_2_noise_robustness()
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
    
    try:
        all_results['test3_fewshot'] = test_3_few_shot()
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
    
    try:
        all_results['test4_scalability'] = test_4_scalability()
    except Exception as e:
        print(f"❌ Test 4 failed: {e}")
    
    try:
        all_results['test5_imbalanced'] = test_5_imbalanced()
    except Exception as e:
        print(f"❌ Test 5 failed: {e}")
    
    try:
        all_results['test6_critical'] = test_6_critical_datasets()
    except Exception as e:
        print(f"❌ Test 6 failed: {e}")
    
    print_header("✅ MEGA-BENCHMARK TERMINÉ")
    print(f"\n🎉 {len(all_results)} tests complétés avec succès!")
    print("\n💾 Pour sauvegarder les résultats:")
    print("   for name, df in all_results.items():")
    print("       df.to_csv(f'{name}_results.csv', index=False)")
