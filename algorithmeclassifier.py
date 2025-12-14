from random import choice
import json
import pandas as pd
import numpy as np
from time import time
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

def to_numpy_matrix(x):
    if isinstance(x, pd.DataFrame):
        return x.to_numpy(dtype=float)
    elif isinstance(x, np.ndarray):
        return x.astype(float)
    else:
        raise TypeError(f"Unsupported type: {type(x)} (expected DataFrame or ndarray)")

def to_numpy_vector(y):
    if isinstance(y, pd.Series):
        return y.to_numpy(dtype=int)
    elif isinstance(y, pd.DataFrame):
        # flatten if it's a single-column DataFrame
        if y.shape[1] == 1:
            return y.iloc[:, 0].to_numpy(dtype=int)
        else:
            raise ValueError("DataFrame has more than one column — expected a single target column.")
    elif isinstance(y, np.ndarray):
        return y.astype(int).ravel()  # ensure 1-D
    else:
        raise TypeError(f"Unsupported type: {type(y)} (expected Series, DataFrame, or ndarray)")

class AlgorithmeClassifier():
    def __init__(self, n_layers=100, vocal=False):
        self.log = """
################################################################
#                                                              #
#      Algorithme.ai: AlgorithmeClassifier                     #
#                                                              #
#      Developed by Charles Dana                               #
#                                                              #
################################################################        
"""
        self.layers = n_layers
        self.population = 0
        self.header = 0
        self.datatypes = 0
        self.target = 0
        self.sat_models = []
        self.vocal = vocal
    
    def qprint(self, txt):
        self.log += "\n" + str(txt)
        if self.vocal:
            print(txt)
    
    def oppose(self, T, F):
        while 42:
            head = choice(range(self.header))
            T_head = float(self.population[T][head])
            F_head = float(self.population[F][head])
            if T_head > F_head:
                return [head, (T_head + F_head) / 2, False]
            if T_head < F_head:
                return [head, (T_head + F_head) / 2, True]
    
    def literal_condition(self, X, literal):
        a, b, c = literal
        if c:
            return X[:, a] <= b
        else:
            return X[:, a] > b
    
    def clause_condition(self, X, clause):
        return np.logical_or.reduce([self.literal_condition(X, literal) for literal in clause])
    
    def sat_condition(self, X, sat):
        return np.logical_and.reduce([self.clause_condition(X, triple[0]) for triple in sat])
    
    def construct_clause(self, Ts, F):
        clause = [self.oppose(np.random.choice(Ts), F)]
        X_subset = self.population[Ts]
        mask = self.clause_condition(X_subset, clause)
        Ts_remainder = Ts[~mask]
        while len(Ts_remainder):
            clause += [self.oppose(np.random.choice(Ts_remainder), F)]
            X_subset = self.population[Ts_remainder]
            mask = self.clause_condition(X_subset, clause)
            Ts_remainder = Ts_remainder[~mask]
        i = 0
        while i < len(clause):
            sub_clause = [clause[j] for j in range(len(clause)) if j != i]
            X_subset = self.population[Ts]
            mask = self.clause_condition(X_subset, sub_clause)
            if np.all(mask):
                clause = sub_clause
            else:
                i += 1
        return clause
    
    def construct_sat(self, target=0):
        indexes = np.where(self.target == target)[0]
        Ts = np.where(self.target != target)[0]
        sat = []
        while len(indexes):
            F = np.random.choice(indexes)
            clause = self.construct_clause(Ts, F)
            mask = self.clause_condition(self.population, clause)
            consequence = np.where(~mask)[0]
            sat += [[clause, F, consequence]]
            indexes = np.setdiff1d(indexes, consequence, assume_unique=True)
        X_false_subset = self.population[np.where(self.target == target)[0]]
        X_true_subset = self.population[np.where(self.target != target)[0]]
        if np.all(~self.sat_condition(X_false_subset, sat)) == False:
            self.qprint("# Not all false elements are false")
            print(self.sat_condition(X_false_subset, sat))
            input("# Not all false elements are false")
        if np.all(self.sat_condition(X_true_subset, sat)) == False:
            self.qprint("# Not all true elements are true")
            print(self.sat_condition(X_true_subset, sat))
            input("# Not all true elements are true")
        return sat
        
    def fit(self, X_train, y_train):
        self.population = to_numpy_matrix(X_train)
        self.target = to_numpy_vector(y_train)
        targets = np.sort(np.unique(self.target))
        self.header = X_train.shape[1]
        t_0 = time()
        self.qprint(f"Starting AlgorithmeClassifier Modeling: [{self.header}]")
        for i in range(self.layers):
            for target in targets:
                sat_model = self.construct_sat(target=target)
                self.sat_models += [[target, sat_model]]
            self.qprint(f"# Algorithme.ai : Remainder[{self.layers - i - 1}] {round((time() - t_0) * (self.layers - i - 1) / (i + 1), 2)}s")
    
    def get_lookalikes(self, X):
        X = to_numpy_matrix(X)
        lookalikes = [[] for _ in range(X.shape[0])]
        def build_associations_fast(X, candidates, associations):
            n = X.shape[0]
            keys = np.fromiter(candidates.keys(), dtype=int)
            masks = np.column_stack([np.asarray(candidates[k], dtype=bool) for k in keys])  # (n_rows, n_keys)
            rows, cols = np.where(masks)   # indexes where mask is True
            for r, c in zip(rows, cols):
                associations[r].append(int(keys[c]))
            return associations
        # Snake Mode
        i = 0
        t_0 = time()
        for pair in self.sat_models:
            sat_model = pair[1]
            target = pair[0]
            candidates = {i : np.ones(X.shape[0], dtype=bool) for i in range(len(self.target)) if self.target[i] == target}
            for triple in sat_model:
                #print(triple[0])
                mask = self.clause_condition(X, triple[0])
                #print(mask)
                for index in triple[2]:
                    candidates[index] = np.logical_and(candidates[index], ~mask)
            lookalikes = build_associations_fast(X, candidates, lookalikes)
            remainder = round((time() - t_0) * (len(self.sat_models) - i) / (i + 1), 2)
            self.qprint(f"# Algorithme.ai : Remainder in Lookalikes [{i} / {len(self.sat_models)}] {remainder}s")
            i += 1
        return lookalikes
        
    def predict_proba(self, X_test):
        population = to_numpy_matrix(X_test)
        lookalikes = self.get_lookalikes(population)
        t_0 = time()
        i = 0
        targets = sorted(list(set(self.target)))
        confidences = [[] for target in range(1+max(targets))]
        for item in population:
            lookalike = lookalikes[i]
            values = [int(self.target[l]) for l in lookalike]
            for target in targets:
                confidences[target] += [sum((v == target for v in values)) / len(values)]
            i += 1
        y_proba = np.column_stack([np.array(confidences[i]) for i in targets])
        return y_proba
        
    def predict(self, X_test):
        return np.argmax(self.predict_proba(X_test), axis=1)
        
    def score(self, X, y, metric="accuracy"):
        proba = self.predict_proba(X)
        y_pred = np.argmax(proba, axis=1)
        if metric == "accuracy":
            return accuracy_score(y, y_pred)
        elif metric == "log_loss":
            return -log_loss(y, proba)
        elif metric == "auc":
            return roc_auc_score(y, proba, multi_class="ovr")
        else:
            raise ValueError(f"Unsupported metric: {metric}")
