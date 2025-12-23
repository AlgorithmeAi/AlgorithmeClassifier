"""
Benchmark rigoureux sur Wine Quality Dataset
Compare AlgorithmeClassifier vs RandomForest vs GradientBoosting

Dataset: Wine Quality (red + white wines)
- 11 features (acidity, sugar, alcohol, etc.)
- 6 quality classes (3-8, remap to 0-5)
- 1000 train samples, reste en test
"""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, 
    log_loss, 
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Import ton AlgorithmeClassifier
try:
    from algorithmeclassifier import AlgorithmeClassifier
    ALGO_AVAILABLE = True
except ImportError:
    print("⚠️  AlgorithmeClassifier non trouvé, sera ignoré dans le benchmark")
    ALGO_AVAILABLE = False

def load_wine_quality():
    """
    Charge le Wine Quality dataset depuis UCI
    Combine red et white wines
    """
    print("📥 Téléchargement du Wine Quality dataset...")
    
    try:
        # Red wine
        red_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        red_wine = pd.read_csv(red_url, sep=';')
        red_wine['type'] = 0  # red
        
        # White wine
        white_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
        white_wine = pd.read_csv(white_url, sep=';')
        white_wine['type'] = 1  # white
        
        # Combine
        wine = pd.concat([red_wine, white_wine], axis=0)
        
        # Préparation
        X = wine.drop('quality', axis=1).values
        y = wine['quality'].values
        
        # Remap quality (3-9) to (0-6) for classification
        y = y - y.min()
        
        print(f"   ✓ Dataset chargé: {len(wine)} samples")
        print(f"   • Features: {X.shape[1]}")
        print(f"   • Classes: {len(np.unique(y))} (quality scores)")
        print(f"   • Distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        return X, y
        
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement: {e}")
        print("💡 Utilisation d'un dataset de remplacement (Breast Cancer)...")
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        return data.data, data.target

def compute_multiclass_auc(y_true, y_proba):
    """Calcule l'AUC multiclasse (one-vs-rest)"""
    n_classes = y_proba.shape[1]
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # AUC macro (moyenne des AUC par classe)
    auc_scores = []
    for i in range(n_classes):
        if len(np.unique(y_true_bin[:, i])) > 1:  # Vérifier qu'il y a les 2 classes
            auc_scores.append(roc_auc_score(y_true_bin[:, i], y_proba[:, i]))
    
    return np.mean(auc_scores) if auc_scores else 0.0

def benchmark_model(name, model, X_train, y_train, X_test, y_test):
    """Benchmark complet d'un modèle"""
    print(f"\n{'='*60}")
    print(f"🔬 Benchmark: {name}")
    print(f"{'='*60}")
    
    results = {"model": name}
    
    # Entraînement
    print("⏳ Entraînement...")
    t_start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t_start
    results["train_time_sec"] = round(train_time, 3)
    print(f"   ✓ Entraînement: {train_time:.3f}s")
    
    # Prédiction
    print("⏳ Prédiction...")
    t_start = time.time()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    pred_time = time.time() - t_start
    results["pred_time_sec"] = round(pred_time, 3)
    print(f"   ✓ Prédiction: {pred_time:.3f}s")
    
    # Métriques
    results["accuracy"] = round(accuracy_score(y_test, y_pred), 4)
    results["log_loss"] = round(log_loss(y_test, y_proba), 4)
    results["auc_ovr"] = round(compute_multiclass_auc(y_test, y_proba), 4)
    results["f1_macro"] = round(f1_score(y_test, y_pred, average='macro'), 4)
    results["f1_weighted"] = round(f1_score(y_test, y_pred, average='weighted'), 4)
    results["precision_macro"] = round(precision_score(y_test, y_pred, average='macro', zero_division=0), 4)
    results["recall_macro"] = round(recall_score(y_test, y_pred, average='macro', zero_division=0), 4)
    
    print(f"\n📊 Résultats:")
    print(f"   • Accuracy:        {results['accuracy']:.4f}")
    print(f"   • Log Loss:        {results['log_loss']:.4f}")
    print(f"   • AUC (OvR):       {results['auc_ovr']:.4f}")
    print(f"   • F1 Macro:        {results['f1_macro']:.4f}")
    print(f"   • F1 Weighted:     {results['f1_weighted']:.4f}")
    print(f"   • Precision Macro: {results['precision_macro']:.4f}")
    print(f"   • Recall Macro:    {results['recall_macro']:.4f}")
    
    # Matrice de confusion (résumé)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📈 Matrice de confusion (diagonale): {np.diag(cm).tolist()}")
    
    return results

def main():
    print("="*60)
    print("🍷 BENCHMARK: Wine Quality Dataset (Multiclass Classification)")
    print("="*60)
    
    # Chargement données
    X, y = load_wine_quality()
    
    # Split: 1000 train, reste en test
    print("\n✂️  Split: 1000 train, reste en test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=1000, random_state=42, stratify=y
    )
    
    print(f"   • Train: {len(X_train)} samples")
    print(f"   • Test:  {len(X_test)} samples")
    print(f"   • Train distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"   • Test distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")
    
    # Modèles à benchmarker
    models = [
        ("Random Forest", RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42,
            n_jobs=1
        )),
        ("Gradient Boosting", GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            random_state=42
        ))
    ]
    
    if ALGO_AVAILABLE:
        models.append(("AlgorithmeClassifier", AlgorithmeClassifier(n_layers=100)))
    
    # Benchmark tous les modèles
    all_results = []
    for name, model in models:
        try:
            results = benchmark_model(name, model, X_train, y_train, X_test, y_test)
            all_results.append(results)
        except Exception as e:
            print(f"\n❌ Erreur avec {name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Tableau comparatif final
    print("\n" + "="*60)
    print("📊 TABLEAU COMPARATIF FINAL")
    print("="*60)
    
    df_results = pd.DataFrame(all_results)
    
    # Identifier les meilleurs pour chaque métrique
    metrics_to_highlight = ["accuracy", "auc_ovr", "f1_macro", "f1_weighted"]
    
    print("\n" + df_results.to_string(index=False))
    
    print("\n🏆 Gagnants par métrique:")
    for metric in metrics_to_highlight:
        if metric in df_results.columns:
            best_idx = df_results[metric].idxmax()
            best_model = df_results.loc[best_idx, "model"]
            best_value = df_results.loc[best_idx, metric]
            print(f"   • {metric:20s}: {best_model:25s} ({best_value:.4f})")
    
    # Efficacité computationnelle
    print("\n⚡ Efficacité computationnelle:")
    for _, row in df_results.iterrows():
        total_time = row['train_time_sec'] + row['pred_time_sec']
        print(f"   • {row['model']:25s}: {total_time:.3f}s total "
              f"(train: {row['train_time_sec']:.3f}s, pred: {row['pred_time_sec']:.3f}s)")
    
    # Sauvegarde
    output_file = "benchmark_wine_results.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n💾 Résultats sauvegardés dans '{output_file}'")
    
    print("\n" + "="*60)
    print("✅ Benchmark terminé!")
    print("="*60)

if __name__ == "__main__":
    main()
