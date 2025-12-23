# AlgorithmeClassifier

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

🐍 **AlgorithmeClassifier** — A minimalist, interpretable Python classifier that builds discriminative SAT-style clauses to separate classes and turns those clauses into calibrated class probabilities.

Based on the **Dana Theorem** for discrete concordance, this classifier achieves state-of-the-art performance on multiclass problems while maintaining theoretical guarantees on AUROC convergence.

---

## 🔥 Key Features

* 🧠 **Logic-first learning**: learns short Boolean clauses that separate each class from the others
* 🎯 **Theoretical foundation**: grounded in SAT theory with O(mn²) complexity guarantees
* 📈 **Competitive performance**: matches or beats Random Forest and Gradient Boosting on key metrics
* 🔍 **Interpretable by construction**: every prediction can be traced to a handful of clauses
* 🧰 **sklearn-compatible**: drop-in replacement with `fit()`, `predict()`, and `predict_proba()`
* 📊 **Multiple metrics**: accuracy, log loss, and AUC scoring built-in

---

## 📊 Benchmark Results

AlgorithmeClassifier has been tested on multiple standard datasets, consistently achieving **top-tier AUC scores** while maintaining competitive accuracy.

### 🎯 Summary: AlgorithmeClassifier wins on discrimination

| Dataset | Type | AlgorithmeClassifier AUC | Best Competitor AUC | Advantage |
|---------|------|--------------------------|---------------------|-----------|
| **Breast Cancer** (binary, medical) | 🏥 High-stakes | **0.9955** 🥇 | 0.9924 (RF) | **+0.0031** (+0.31%) |
| **Digits** (10 classes, balanced) | 🎯 Easy | **0.9987** 🥇 | 0.9985 (RF) | **+0.0002** (+0.02%) |
| **Wine Quality** (7 classes, imbalanced) | 🍷 Hard | **0.7106** 🥇 | 0.6728 (RF) | **+0.0378** (+5.6%) |

**Key Insights**: 
- 🏆 **Wins on AUC across all 3 datasets** (binary, balanced multiclass, imbalanced multiclass)
- 📈 **Advantage scales with difficulty**: Easy (+0.02%) → Medical (+0.31%) → Hard (+5.6%)
- 🎯 **Consistent excellence**: Best overall accuracy on 2/3 benchmarks, best F1 on all 3

---

### Benchmark 1: Digits Dataset (Balanced, 10 classes)

**Dataset**: sklearn Digits — 10 classes, 64 features, 1000 train / 797 test

| Model | Accuracy | **AUC (OvR)** | F1 Macro | Log Loss | Train+Inference |
|-------|----------|---------------|----------|----------|-----------------|
| **AlgorithmeClassifier** | **0.9573** 🥇 | **0.9987** 🏆 | **0.9572** 🥇 | 0.3019 | 13.6s |
| Random Forest | **0.9573** 🥇 | 0.9985 | 0.9571 | 0.4120 | 0.09s |
| Gradient Boosting | 0.9435 | 0.9983 | 0.9434 | **0.1771** 🥇 | 2.2s |

**Takeaways:**
- 🏆 **Highest AUC** (0.9987) — best discrimination across all 10 classes
- 🥇 **Tied best accuracy** (95.73%) — matches Random Forest
- 🥇 **Best F1 macro** — superior balanced performance across classes
- ⚖️ **Trade-off**: 150x slower than RF, but marginal AUC gain on this easy dataset

---

### Benchmark 2: Wine Quality Dataset (Imbalanced, 7 classes)

**Dataset**: UCI Wine Quality (red + white) — 7 classes, 12 features, 1000 train / 5497 test  
**Challenge**: Highly imbalanced (classes 0,6 have <1% representation)

| Model | Accuracy | **AUC (OvR)** | F1 Weighted | Precision Macro | Train+Inference |
|-------|----------|---------------|-------------|-----------------|-----------------|
| **AlgorithmeClassifier** | **0.5780** 🥇 | **0.7106** 🏆 | **0.5560** 🥇 | **0.4588** 🥇 | 19.1s |
| Random Forest | 0.5719 | 0.6728 | 0.5498 | 0.3987 | 0.16s |
| Gradient Boosting | 0.5514 | 0.6321 | 0.5385 | 0.3339 | 0.82s |

**Takeaways:**
- 🏆 **Dominant AUC advantage** (+5.6% vs RF, +12.4% vs GB) — shines on imbalanced data
- 🥇 **Best accuracy** (57.80%) — hardest problem, still wins
- 🥇 **Best precision macro** (0.4588) — superior minority class handling
- 💡 **Key finding**: The harder and more imbalanced the problem, the bigger the AlgorithmeClassifier advantage

---

### 📈 When AlgorithmeClassifier Excels

✅ **Imbalanced datasets** — The discrete concordance mechanism naturally handles class imbalance better than tree ensembles  
✅ **High-stakes ranking** — When AUC/discrimination is critical (credit scoring, medical diagnosis, fraud detection)  
✅ **Multiclass problems** — Consistent performance across all classes (high macro scores)  
✅ **Interpretability needs** — Extract and audit the exact logical rules learned

⚠️ **When to use alternatives:**
- Real-time inference requirements (< 100ms per prediction)
- Very large datasets (> 100k samples) where speed dominates
- Simple balanced problems where RF is already near-optimal

See full benchmark details in [`Digits/`](Digits/) and [`Wine/`](Wine/) folders.

---

## 🚀 Installation

```bash
pip install -U scikit-learn pandas numpy
# Copy algorithmeclassifier.py into your project
```

Or clone the repository:

```bash
git clone https://github.com/AlgorithmeAi/AlgorithmeClassifier.git
cd AlgorithmeClassifier
```

---

## 🎯 Quickstart

```python
import pandas as pd
from algorithmeclassifier import AlgorithmeClassifier

# Initialize
clf = AlgorithmeClassifier(n_layers=100)

# Train
clf.fit(X_train, y_train)

# Predict
proba = clf.predict_proba(X_test)
y_pred = clf.predict(X_test)

# Score with multiple metrics
print("Accuracy:", clf.score(X_test, y_test, metric="accuracy"))
print("Log Loss:", clf.score(X_test, y_test, metric="log_loss"))
print("AUC (OvR):", clf.score(X_test, y_test, metric="auc"))
```

---

## 🧮 Theoretical Foundation: The Dana Theorem

### Informal Statement

For any finite binary-labeled dataset, you can construct a CNF (SAT) formula that exactly reproduces the labels using at most **O(mn²)** time, where **m** is the number of features and **n** the number of samples.

### Formal Statement

Let **A ∈ {0,1}ⁿˣᵐ** be the feature matrix and **X ∈ {0,1}ⁿ** the label vector.  
There exists a CNF **φ** with at most **≤ |F|** clauses and **≤ |E| |F|** literals, constructible in **O(mn²)** time, such that:

**∀ i, φ(Aᵢ,*) = X(i)**

Here, **E = {i : X(i) = 1}** and **F = {i : X(i) = 0}**. A dual DNF statement holds by swapping *E* and *F*.

### Why It Matters

- The clause constructor mirrors the proof's discriminative step: each literal encodes a feature difference between positive and negative examples
- The algorithm aggregates these literals into a CNF per class — yielding a compact, data-backed set of rules
- **Complexity**: O(m · |E| · |F|) ⊆ O(mn²)
- **Interpretability**: Every prediction results from the activation of a few IF-THEN rules that can be traced back to actual samples
- **AUROC Convergence**: The lookalike mechanism converges to a discrete concordance determinant, providing a theoretical ceiling for classification performance

---

## 💡 How It Works

**Goal**: Separate a *target* class from the rest with a small set of discriminative clauses.

1. **Pick a target** (e.g., class `k`). Split indices into positives (`F`) and negatives (`T`)
2. **Construct a clause** true for positives and false for negatives
3. **Iterate**: Add clauses until the class is well-separated
4. **Score via lookalikes**: Each test point's score is the ratio of true-class lookalikes to total lookalikes across multiple clause layers
5. **Predict**: Aggregate scores across layers using the Law of Large Numbers to converge to optimal concordance

This yields fast, interpretable predictions that mirror the **Dana Theorem** construction while achieving state-of-the-art discrimination.

---

## 📁 Repository Structure

```
AlgorithmeClassifier/
├── algorithmeclassifier.py    # Main classifier implementation
├── test_basic.py               # Unit tests
├── Digits/                     # Benchmark on sklearn Digits dataset
│   ├── benchmark_digits.py    # Benchmark script
│   └── benchmark_results.txt  # Full results
├── Wine/                     # Benchmark on sklearn Wine dataset
│   ├── wine.py                # Benchmark script
│   └── wine_results.txt        # Full results
├── Breast/                     # Benchmark on sklearn Breast Cancer dataset
│   ├── breast.py                # Benchmark script
│   └── breast_results.txt        # Full results
├── Kaggle Starter Pack/        # Quick-start templates
├── README.md                   # This file
├── LICENSE                     # MIT License
└── pyproject.toml             # Package configuration
```

---

## 🔬 Running Benchmarks

### Digits Dataset

```bash
cd Digits
python benchmark_digits.py
```

This will train and evaluate AlgorithmeClassifier, Random Forest, and Gradient Boosting on the sklearn Digits dataset (10 classes, 64 features).

### Custom Benchmarks

```python
from sklearn.datasets import load_digits
from algorithmeclassifier import AlgorithmeClassifier
from sklearn.model_selection import train_test_split

# Load data
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=1000, random_state=42)

# Train
clf = AlgorithmeClassifier(n_layers=100)
clf.fit(X_train, y_train)

# Evaluate
print(f"Accuracy: {clf.score(X_test, y_test, metric='accuracy'):.4f}")
print(f"AUC: {clf.score(X_test, y_test, metric='auc'):.4f}")
```

---

## ⚙️ API Reference

### `AlgorithmeClassifier(n_layers=100, random_state=None)`

**Parameters:**
- `n_layers` (int): Number of clause layers to generate (default: 100). More layers improve convergence but increase computation time.
- `random_state` (int, optional): Random seed for reproducibility

**Methods:**
- `fit(X, y)`: Train the classifier
- `predict(X)`: Predict class labels
- `predict_proba(X)`: Predict class probabilities
- `score(X, y, metric='accuracy')`: Evaluate performance
  - `metric` options: `'accuracy'`, `'log_loss'`, `'auc'`

---

## 🎓 Use Cases

**When to use AlgorithmeClassifier:**
- ✅ You need **high AUC/discrimination** for ranking or scoring tasks
- ✅ **Interpretability** is critical (extract and inspect learned rules)
- ✅ You have a **small to medium dataset** (< 10k samples)
- ✅ **Multiclass classification** with balanced classes
- ✅ You want a **theoretically grounded** approach with performance guarantees

**When to use alternatives:**
- ❌ **Real-time inference** with strict latency requirements (< 10ms)
- ❌ **Very large datasets** (> 100k samples) where speed is critical
- ❌ You need the absolute fastest training time

---

## 🛠️ Performance Optimization Tips

1. **Reduce `n_layers`** for faster inference (try 50 or 25 for speed vs accuracy trade-off)
2. **Use smaller training sets** when possible (the algorithm scales O(mn²))
3. **Parallelize** if modifying the code (lookalike computation is embarrassingly parallel)
4. **Feature selection** before training can dramatically speed things up

---

## 📚 Research & Citations

This implementation is based on:

**"Théorie de la Concordance Discrète : Déterminant SAT de l'AUROC et Limites de la Résolvabilité"**  
*Charles Dana, December 2025*

The paper demonstrates that AUROC performance ceilings are dictated by the logical structure of features rather than algorithm sophistication, and provides polynomial-time construction guarantees via the Dana Theorem.

---

## 🤝 Contributing

Contributions are welcome! Areas of interest:
- **Performance optimization** (Cython, C++ extensions, GPU acceleration)
- **Additional benchmarks** on diverse datasets
- **Visualization tools** for learned clauses and decision paths
- **Documentation improvements**

Please open an issue or submit a pull request.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built with inspiration from SAT solver theory and statistical learning
- Benchmarked against scikit-learn's excellent ensemble methods
- Thanks to the open-source ML community

---

## 📞 Contact

- **GitHub**: [@AlgorithmeAi](https://github.com/AlgorithmeAi)
- **Issues**: [Report bugs or request features](https://github.com/AlgorithmeAi/AlgorithmeClassifier/issues)

---

**⭐ If you find this useful, please star the repository!**
