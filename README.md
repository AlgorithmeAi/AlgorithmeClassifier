# AlgorithmeClassifier

A minimalist, interpretable Python classifier that builds **discriminative SAT-style clauses** to separate classes and turns those clauses into calibrated class probabilities. It’s fast, dependency-light, and readable—ideal for tabular problems and competitions where you want **transparent rules with real predictive bite**.

## Features
- 🧠 **Logic-first learning**: learns short Boolean clauses that separate each class from the others.
- 🔍 **Interpretable by construction**: every prediction can be traced to a handful of clauses.
- 📈 **Probabilities + metrics**: `predict_proba`, `predict`, and `score(metric=...)` (accuracy / log_loss / AUC).
- 🧰 **Pandas / NumPy friendly**: accepts DataFrames or ndarrays; includes helper converters.
- 🧪 **Starter pack included**: small Kaggle-style workflow for fast experimentation.

---

## Installation

```bash
pip install -U scikit-learn pandas numpy
# then copy algorithmeclassifier.py into your project, or install from your repo/package
```

> The module lives at `AlgorithmeClassifier/algorithmeclassifier.py` in your ZIP.

---

## Quickstart

```python
import pandas as pd
from algorithmeclassifier import AlgorithmeClassifier

clf = AlgorithmeClassifier()
clf.fit(X_train, y_train)

proba = clf.predict_proba(X_test)
y_pred = clf.predict(X_test)

print("Accuracy:", clf.score(X_test, y_test, metric="accuracy"))
print("LogLoss :", clf.score(X_test, y_test, metric="log_loss"))
print("AUC     :", clf.score(X_test, y_test, metric="auc"))
```

---

## How it works

**Goal:** separate a *target* class from the rest with a small set of **discriminative clauses**.

1. **Pick a target** (e.g., class `k`). Split indices into positives (`F`) and negatives (`T`).
2. **Construct a clause** true for positives and false for negatives.
3. **Iterate**: add clauses until the class is well-separated.
4. **Predict**: each SAT-like *program* votes via its satisfied clauses.

This yields fast, interpretable predictions that mirror the **Dana Theorem** construction.

---

## Mathematical foundations (Dana Theorem)

**Informal idea.**  
For any finite binary-labeled dataset, you can construct a CNF (SAT) formula that exactly reproduces the labels using at most \( O(mn^2) \) time, where \( m \) is the number of features and \( n \) the number of samples.

**Formal statement (condensed).**  
Let \(A\in\{0,1\}^{n\times m}\) and labels \(X\in\{0,1\}^n\). Then there exists a CNF \(\varphi\) with ≤ \(|F|\) clauses and ≤ \(|E||F|\) literals, constructible in \(O(mn^2)\), such that \(\forall i,\ \varphi(A_{i,*})=X(i)\).

This classifier implements that constructive mapping in practice.

---

## License

MIT License

Copyright (c) 2025 Algorithme.ai

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
