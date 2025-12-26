import numpy as np
import pandas as pd
from time import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import (fetch_covtype, fetch_openml, load_digits, 
                              load_wine, load_breast_cancer)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, f1_score, log_loss, 
                             classification_report, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Imports optionnels (sans spam de warnings)
import sys
import os
HAS_XGB = False
HAS_LGBM = False

# Rediriger stderr temporairement pour éviter les warnings répétés
original_stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    pass

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    pass

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
        print(f"\n🔄 Entraînement de {name}...", end=" ", flush=True)
        
        if fit_params is None:
            fit_params = {}
        
        # Fit
        t_start = time()
        model.fit(self.X_train, self.y_train, **fit_params)
        t_fit = time() - t_start
        print(f"✓ ({format_time(t_fit)})")
        
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
        f1_weighted = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
        
        result = {
            'Model': name,
            'Accuracy': accuracy,
            'F1-Macro': f1_macro,
            'F1-Weighted': f1_weighted,
            'LogLoss': logloss,
            'Fit Time': t_fit,
            'Predict Time': t_pred,
            'Total Time': t_fit + t_pred
        }
        
        self.results.append(result)
        
        print(f"   📊 Acc: {accuracy:.4f} | F1-Macro: {f1_macro:.4f} | LogLoss: {logloss:.4f}")
        
        return result
    
    def run_benchmark(self):
        """Exécute le benchmark complet"""
        print_header(f"🎯 BENCHMARK: {self.dataset_name}")
        print(f"📦 Train: {self.X_train.shape[0]} samples, {self.X_train.shape[1]} features")
        print(f"📦 Test: {self.X_test.shape[0]} samples")
        print(f"🎲 Classes: {len(np.unique(self.y_train))}")
        
        # Snake (FastAlgorithmeClassifier)
        print_header("🐍 SNAKE (FastAlgorithmeClassifier)", "─")
        snake = FastAlgorithmeClassifier(n_layers=100, vocal=False, n_jobs=-1)
        self.evaluate_model("Snake (n_layers=100)", snake)
        
        # Logistic Regression
        print_header("📐 LOGISTIC REGRESSION", "─")
        lr = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42)
        self.evaluate_model("Logistic Regression", lr)
        
        # Linear SVC
        print_header("📏 LINEAR SVC", "─")
        svc = LinearSVC(max_iter=1000, random_state=42)
        self.evaluate_model("Linear SVC", svc)
        
        # Random Forest
        print_header("🌲 RANDOM FOREST", "─")
        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        self.evaluate_model("Random Forest (100 trees)", rf)
        
        # Histogram Gradient Boosting
        print_header("📈 HISTOGRAM GRADIENT BOOSTING", "─")
        hgb = HistGradientBoostingClassifier(max_iter=100, random_state=42)
        self.evaluate_model("HistGradientBoosting", hgb)
        
        # XGBoost
        if HAS_XGB:
            print_header("⚡ XGBOOST", "─")
            xgb = XGBClassifier(n_estimators=100, n_jobs=-1, random_state=42, 
                               eval_metric='logloss', verbosity=0)
            self.evaluate_model("XGBoost (100 trees)", xgb)
        
        # LightGBM
        if HAS_LGBM:
            print_header("💡 LIGHTGBM", "─")
            lgbm = LGBMClassifier(n_estimators=100, n_jobs=-1, random_state=42, 
                                 verbose=-1, force_col_wise=True)
            self.evaluate_model("LightGBM (100 trees)", lgbm)
        
        self.print_results()
        
        return pd.DataFrame(self.results)
    
    def print_results(self):
        """Affiche un tableau récapitulatif"""
        print_header("📊 RÉSULTATS FINAUX")
        
        df = pd.DataFrame(self.results)
        
        # Tableau principal
        print("\n" + "═" * 110)
        print(f"{'MODÈLE':<30} | {'ACC':<8} | {'F1-MACRO':<8} | {'F1-WEIGHT':<8} | {'LOGLOSS':<8} | {'FIT':<10} | {'PRED':<10}")
        print("─" * 110)
        
        for _, row in df.iterrows():
            print(f"{row['Model']:<30} | "
                  f"{row['Accuracy']:<8.4f} | "
                  f"{row['F1-Macro']:<8.4f} | "
                  f"{row['F1-Weighted']:<8.4f} | "
                  f"{row['LogLoss']:<8.4f} | "
                  f"{format_time(row['Fit Time']):<10} | "
                  f"{format_time(row['Predict Time']):<10}")
        
        print("═" * 110)
        
        # Classements
        print("\n🏆 CLASSEMENTS:")
        print(f"   🥇 Meilleure Accuracy:    {df.loc[df['Accuracy'].idxmax(), 'Model']} ({df['Accuracy'].max():.4f})")
        print(f"   🥇 Meilleur F1-Macro:     {df.loc[df['F1-Macro'].idxmax(), 'Model']} ({df['F1-Macro'].max():.4f})")
        print(f"   🥇 Meilleur LogLoss:      {df.loc[df['LogLoss'].idxmin(), 'Model']} ({df['LogLoss'].min():.4f})")
        print(f"   ⚡ Plus rapide (fit):     {df.loc[df['Fit Time'].idxmin(), 'Model']} ({format_time(df['Fit Time'].min())})")
        print(f"   ⚡ Plus rapide (predict): {df.loc[df['Predict Time'].idxmin(), 'Model']} ({format_time(df['Predict Time'].min())})")
        
        # Position de Snake
        snake_row = df[df['Model'].str.contains('Snake')].iloc[0]
        print(f"\n🐍 POSITION DE SNAKE:")
        print(f"   Accuracy:    {(df['Accuracy'] > snake_row['Accuracy']).sum() + 1}/{len(df)}")
        print(f"   F1-Macro:    {(df['F1-Macro'] > snake_row['F1-Macro']).sum() + 1}/{len(df)}")
        print(f"   LogLoss:     {(df['LogLoss'] < snake_row['LogLoss']).sum() + 1}/{len(df)}")
        print(f"   Vitesse Fit: {(df['Fit Time'] < snake_row['Fit Time']).sum() + 1}/{len(df)}")
        
        return df


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


# === DATASETS RÉELS/ORGANIQUES ===

def load_forest_covertype(n_train=2000, n_test=1000):
    """Forest CoverType - 7 classes, données géographiques réelles"""
    print("📥 Chargement de Forest CoverType (7 classes, géographie)...")
    covtype = fetch_covtype()
    X, y = covtype.data, covtype.target - 1
    
    indices = np.random.choice(len(X), n_train + n_test, replace=False)
    X_sample = X[indices]
    y_sample = y[indices]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, train_size=n_train, random_state=42, stratify=y_sample
    )
    
    X_train, X_test = apply_pca(X_train, X_test, n_components=20)
    
    return X_train, X_test, y_train, y_test


def load_mnist_digits(n_train=2000, n_test=1000):
    """MNIST Digits - 10 classes, images de chiffres manuscrits"""
    print("📥 Chargement de MNIST Digits (10 classes, images)...")
    try:
        # Essayer la version complète MNIST
        mnist = fetch_openml('mnist_784', version=1, parser='auto')
        X, y = mnist.data.to_numpy(), mnist.target.astype(int).to_numpy()
        
        indices = np.random.choice(len(X), n_train + n_test, replace=False)
        X_sample = X[indices]
        y_sample = y[indices]
    except:
        # Fallback sur sklearn digits
        print("   → Utilisation de sklearn digits (version réduite)")
        digits = load_digits()
        X_sample, y_sample = digits.data, digits.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, train_size=min(n_train, len(X_sample)//2), 
        test_size=min(n_test, len(X_sample)//2), random_state=42
    )
    
    X_train, X_test = apply_pca(X_train, X_test, n_components=20)
    
    return X_train, X_test, y_train, y_test


def load_fashion_mnist(n_train=2000, n_test=1000):
    """Fashion-MNIST - 10 classes, images de vêtements"""
    print("📥 Chargement de Fashion-MNIST (10 classes, vêtements)...")
    try:
        fashion = fetch_openml('Fashion-MNIST', version=1, parser='auto')
        X, y = fashion.data.to_numpy(), fashion.target.astype(int).to_numpy()
        
        indices = np.random.choice(len(X), n_train + n_test, replace=False)
        X_sample = X[indices]
        y_sample = y[indices]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_sample, y_sample, train_size=n_train, random_state=42, stratify=y_sample
        )
        
        X_train, X_test = apply_pca(X_train, X_test, n_components=20)
        
        return X_train, X_test, y_train, y_test
    except:
        print("   ⚠️ Fashion-MNIST non disponible, skip")
        return None, None, None, None


def load_letter_recognition(n_train=2000, n_test=1000):
    """Letter Recognition - 26 classes, reconnaissance de lettres"""
    print("📥 Chargement de Letter Recognition (26 classes, lettres)...")
    try:
        letter = fetch_openml('letter', version=1, parser='auto')
        X = letter.data.to_numpy() if hasattr(letter.data, 'to_numpy') else letter.data
        y = LabelEncoder().fit_transform(letter.target)
        
        indices = np.random.choice(len(X), n_train + n_test, replace=False)
        X_sample = X[indices]
        y_sample = y[indices]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_sample, y_sample, train_size=n_train, random_state=42, stratify=y_sample
        )
        
        X_train, X_test = apply_pca(X_train, X_test, n_components=20)
        
        return X_train, X_test, y_train, y_test
    except:
        print("   ⚠️ Letter Recognition non disponible, skip")
        return None, None, None, None


def load_satimage(n_train=2000, n_test=1000):
    """Satimage - 6 classes, classification d'images satellite"""
    print("📥 Chargement de Satimage (6 classes, satellite)...")
    try:
        satimage = fetch_openml('sat', version=1, parser='auto')
        X = satimage.data.to_numpy() if hasattr(satimage.data, 'to_numpy') else satimage.data
        y = satimage.target.astype(int).to_numpy() if hasattr(satimage.target, 'to_numpy') else satimage.target.astype(int)
        
        indices = np.random.choice(len(X), min(n_train + n_test, len(X)), replace=False)
        X_sample = X[indices]
        y_sample = y[indices]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_sample, y_sample, train_size=min(n_train, len(X_sample)//2), 
            test_size=min(n_test, len(X_sample)//2), random_state=42
        )
        
        X_train, X_test = apply_pca(X_train, X_test, n_components=20)
        
        return X_train, X_test, y_train, y_test
    except:
        print("   ⚠️ Satimage non disponible, skip")
        return None, None, None, None


if __name__ == "__main__":
    print_header("🐍 SNAKE CLASSIFIER - BENCHMARK DATASETS RÉELS (PCA 20 features) 🐍")
    
    if not HAS_XGB:
        print("⚠️  XGBoost non installé (optionnel)")
    if not HAS_LGBM:
        print("⚠️  LightGBM non installé (optionnel)")
    
    results_summary = []
    
    # Benchmark 1: Forest CoverType
    X_train, X_test, y_train, y_test = load_forest_covertype(n_train=2000, n_test=1000)
    benchmark1 = BenchmarkRunner(X_train, X_test, y_train, y_test, 
                                 "Forest CoverType (7 classes)")
    df1 = benchmark1.run_benchmark()
    results_summary.append(("Forest CoverType", df1))
    
    # Benchmark 2: MNIST Digits
    X_train, X_test, y_train, y_test = load_mnist_digits(n_train=2000, n_test=1000)
    if X_train is not None:
        benchmark2 = BenchmarkRunner(X_train, X_test, y_train, y_test,
                                     "MNIST Digits (10 classes)")
        df2 = benchmark2.run_benchmark()
        results_summary.append(("MNIST Digits", df2))
    
    # Benchmark 3: Fashion-MNIST
    result = load_fashion_mnist(n_train=2000, n_test=1000)
    if result[0] is not None:
        X_train, X_test, y_train, y_test = result
        benchmark3 = BenchmarkRunner(X_train, X_test, y_train, y_test,
                                     "Fashion-MNIST (10 classes)")
        df3 = benchmark3.run_benchmark()
        results_summary.append(("Fashion-MNIST", df3))
    
    # Benchmark 4: Letter Recognition
    result = load_letter_recognition(n_train=2000, n_test=1000)
    if result[0] is not None:
        X_train, X_test, y_train, y_test = result
        benchmark4 = BenchmarkRunner(X_train, X_test, y_train, y_test,
                                     "Letter Recognition (26 classes)")
        df4 = benchmark4.run_benchmark()
        results_summary.append(("Letter Recognition", df4))
    
    # Benchmark 5: Satimage
    result = load_satimage(n_train=2000, n_test=1000)
    if result[0] is not None:
        X_train, X_test, y_train, y_test = result
        benchmark5 = BenchmarkRunner(X_train, X_test, y_train, y_test,
                                     "Satimage (6 classes)")
        df5 = benchmark5.run_benchmark()
        results_summary.append(("Satimage", df5))
    
    # Résumé global
    print_header("🎊 RÉSUMÉ GLOBAL - TOUS LES DATASETS")
    
    if len(results_summary) == 0:
        print("\n⚠️  Aucun dataset n'a pu être chargé. Vérifiez votre connexion internet.")
        print("   Certains datasets nécessitent fetch_openml pour télécharger depuis OpenML.")
    else:
        for dataset_name, df in results_summary:
            snake_row = df[df['Model'].str.contains('Snake')].iloc[0]
            print(f"\n{dataset_name}:")
            print(f"  Snake - Acc: {snake_row['Accuracy']:.4f}, F1: {snake_row['F1-Macro']:.4f}, "
                  f"Time: {format_time(snake_row['Total Time'])}")
            print(f"  Position: {(df['Accuracy'] > snake_row['Accuracy']).sum() + 1}/{len(df)} en accuracy")
    
    print_header("✅ BENCHMARK TERMINÉ")
