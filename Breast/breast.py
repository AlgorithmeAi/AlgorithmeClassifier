"""
Benchmark rigoureux sur Breast Cancer Wisconsin Dataset
Compare AlgorithmeClassifier vs RandomForest vs GradientBoosting

Dataset: Breast Cancer Wisconsin (Diagnostic)
- 30 features (cell nuclei characteristics)
- 2 classes (Malignant=1, Benign=0)
- 1000 train samples, reste en test
- Problème médical high-stakes où AUC est critique
"""
import time
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, 
    log_loss, 
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# Import ton AlgorithmeClassifier
try:
    from algorithmeclassifier import AlgorithmeClassifier
    ALGO_AVAILABLE = True
except ImportError:
    print("⚠️  AlgorithmeClassifier non trouvé, sera ignoré dans le benchmark")
    ALGO_AVAILABLE = False

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
    y_proba = model.predict_proba(X_test)[:, 1]  # Probabilité classe positive
    pred_time = time.time() - t_start
    results["pred_time_sec"] = round(pred_time, 3)
    print(f"   ✓ Prédiction: {pred_time:.3f}s")
    
    # Métriques binaires
    results["accuracy"] = round(accuracy_score(y_test, y_pred), 4)
    results["auc"] = round(roc_auc_score(y_test, y_proba), 4)
    results["f1"] = round(f1_score(y_test, y_pred), 4)
    results["precision"] = round(precision_score(y_test, y_pred, zero_division=0), 4)
    results["recall"] = round(recall_score(y_test, y_pred, zero_division=0), 4)
    
    # Matrice de confusion
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    results["true_neg"] = int(tn)
    results["false_pos"] = int(fp)
    results["false_neg"] = int(fn)
    results["true_pos"] = int(tp)
    
    # Spécificité (importante en médical)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    results["specificity"] = round(specificity, 4)
    
    print(f"\n📊 Résultats:")
    print(f"   • Accuracy:    {results['accuracy']:.4f}")
    print(f"   • AUC:         {results['auc']:.4f} {'🏆' if results['auc'] == max([r.get('auc', 0) for r in [results]]) else ''}")
    print(f"   • F1 Score:    {results['f1']:.4f}")
    print(f"   • Precision:   {results['precision']:.4f}")
    print(f"   • Recall:      {results['recall']:.4f}")
    print(f"   • Specificity: {results['specificity']:.4f}")
    
    print(f"\n📈 Matrice de confusion:")
    print(f"   • True Negatives:  {tn:4d}  (correctly identified benign)")
    print(f"   • False Positives: {fp:4d}  (benign predicted as malignant)")
    print(f"   • False Negatives: {fn:4d}  (malignant missed - critical!)")
    print(f"   • True Positives:  {tp:4d}  (correctly identified malignant)")
    
    return results

def main():
    print("="*60)
    print("🏥 BENCHMARK: Breast Cancer Wisconsin Dataset")
    print("="*60)
    
    # Chargement données
    print("\n📥 Chargement du Breast Cancer dataset...")
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    print(f"   ✓ Dataset chargé: {len(X)} samples")
    print(f"   • Features: {X.shape[1]} (cell nuclei measurements)")
    print(f"   • Classes: 2 (0=Malignant, 1=Benign)")
    print(f"   • Distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"   • Balance: {100 * y.mean():.1f}% benign")
    
    # Split: 400 train (70% de 569), reste en test
    print("\n✂️  Split: 400 train, reste en test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=400, random_state=42, stratify=y
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
    
    # Colonnes à afficher
    display_cols = ["model", "accuracy", "auc", "f1", "precision", "recall", 
                    "specificity", "false_neg", "train_time_sec", "pred_time_sec"]
    print("\n" + df_results[display_cols].to_string(index=False))
    
    print("\n🏆 Gagnants par métrique:")
    for metric in ["accuracy", "auc", "f1", "precision", "recall", "specificity"]:
        if metric in df_results.columns:
            best_idx = df_results[metric].idxmax()
            best_model = df_results.loc[best_idx, "model"]
            best_value = df_results.loc[best_idx, metric]
            print(f"   • {metric:12s}: {best_model:25s} ({best_value:.4f})")
    
    # Métrique critique en médical
    print("\n🚨 Métrique critique (False Negatives - cancers manqués):")
    for _, row in df_results.iterrows():
        fn = row['false_neg']
        total_malignant = row['false_neg'] + row['true_pos']
        miss_rate = 100 * fn / total_malignant if total_malignant > 0 else 0
        print(f"   • {row['model']:25s}: {fn:2d} cancers manqués ({miss_rate:.1f}% miss rate)")
    
    # Efficacité computationnelle
    print("\n⚡ Efficacité computationnelle:")
    for _, row in df_results.iterrows():
        total_time = row['train_time_sec'] + row['pred_time_sec']
        print(f"   • {row['model']:25s}: {total_time:.3f}s total "
              f"(train: {row['train_time_sec']:.3f}s, pred: {row['pred_time_sec']:.3f}s)")
    
    # Sauvegarde
    output_file = "benchmark_breast_cancer_results.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n💾 Résultats sauvegardés dans '{output_file}'")
    
    print("\n" + "="*60)
    print("✅ Benchmark terminé!")
    print("="*60)
    
    # Recommandation finale
    best_auc_idx = df_results['auc'].idxmax()
    best_auc_model = df_results.loc[best_auc_idx, 'model']
    best_auc = df_results.loc[best_auc_idx, 'auc']
    
    print(f"\n💡 Recommandation pour diagnostic médical:")
    print(f"   {best_auc_model} avec AUC={best_auc:.4f} offre la meilleure")
    print(f"   discrimination pour identifier les cas malignants.")

if __name__ == "__main__":
    main()
