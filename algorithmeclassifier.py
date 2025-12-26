import numpy as np
import pandas as pd
from time import time
from random import choice
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from multiprocessing import Pool, cpu_count
from sklearn.base import BaseEstimator, ClassifierMixin

def to_numpy_matrix(x):
    if isinstance(x, pd.DataFrame):
        return x.to_numpy(dtype=float)
    elif isinstance(x, np.ndarray):
        return x.astype(float)
    else:
        raise TypeError(f"Unsupported type: {type(x)}")

def to_numpy_vector(y):
    if isinstance(y, (pd.Series, pd.DataFrame)):
        return y.values.ravel().astype(int)
    return np.asarray(y).astype(int).ravel()

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

# Fonction globale pour paralléliser construct_sat avec layer index
def _parallel_construct_sat_with_layer(args):
    """Worker function to construct SAT model for a single (layer, target) pair"""
    layer_idx, target_label, population, target, header = args
    np.random.seed()  # Important pour avoir des résultats aléatoires différents
    
    target_indices = np.where(target == target_label)[0]
    non_target_indices = np.where(target != target_label)[0]
    
    sat = []
    remaining_f = target_indices.copy()
    
    def oppose(T, F):
        while True:
            head = choice(range(header))
            T_head = float(population[T][head])
            F_head = float(population[F][head])
            if T_head > F_head:
                return [head, (T_head + F_head) / 2, False]
            if T_head < F_head:
                return [head, (T_head + F_head) / 2, True]
    
    def literal_condition(X, literal):
        idx, thresh, is_le = literal
        return X[:, idx] <= thresh if is_le else X[:, idx] > thresh
    
    def clause_condition(X, clause):
        mask = np.zeros(X.shape[0], dtype=bool)
        for lit in clause:
            mask |= literal_condition(X, lit)
        return mask
    
    def construct_clause(Ts, F):
        clause = [oppose(np.random.choice(Ts), F)]
        while True:
            mask = clause_condition(population[Ts], clause)
            Ts_remainder = Ts[~mask]
            if len(Ts_remainder) == 0:
                break
            clause.append(oppose(np.random.choice(Ts_remainder), F))
        
        # Iterative Pruning
        i = 0
        while i < len(clause):
            sub_clause = [clause[j] for j in range(len(clause)) if j != i]
            if np.all(clause_condition(population[Ts], sub_clause)):
                clause = sub_clause
            else:
                i += 1
        return clause
    
    while len(remaining_f):
        F = np.random.choice(remaining_f)
        clause = construct_clause(non_target_indices, F)
        full_mask = clause_condition(population, clause)
        consequence = np.where(~full_mask & (target == target_label))[0]
        sat.append([clause, F, consequence])
        remaining_f = np.setdiff1d(remaining_f, consequence, assume_unique=True)
    
    return (layer_idx, target_label, sat)


class FastAlgorithmeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_layers=100, vocal=False, n_jobs=-1):
        self.log = """
################################################################
#                                                              #
#      Algorithme.ai: FastAlgorithmeClassifier                 #
#                                                              #
#      Developed by Charles Dana                               #
#                                                              #
################################################################        
"""
        self.n_layers = n_layers
        self.layers = n_layers
        self.vocal = vocal
        self.n_jobs = n_jobs if n_jobs > 0 else cpu_count()
        self.population = None
        self.target = None
        self.header = 0
        self.sat_models = []
        self.clauses = []
        self.lookalikes = {}

    def qprint(self, txt):
        if self.vocal:
            print(txt)

    def oppose(self, T, F):
        while True:
            head = choice(range(self.header))
            T_head = float(self.population[T][head])
            F_head = float(self.population[F][head])
            if T_head > F_head:
                return [head, (T_head + F_head) / 2, False]
            if T_head < F_head:
                return [head, (T_head + F_head) / 2, True]

    def literal_condition(self, X, literal):
        idx, thresh, is_le = literal
        return X[:, idx] <= thresh if is_le else X[:, idx] > thresh

    def clause_condition(self, X, clause):
        mask = np.zeros(X.shape[0], dtype=bool)
        for lit in clause:
            mask |= self.literal_condition(X, lit)
        return mask

    def construct_clause(self, Ts, F):
        clause = [self.oppose(np.random.choice(Ts), F)]
        while True:
            mask = self.clause_condition(self.population[Ts], clause)
            Ts_remainder = Ts[~mask]
            if len(Ts_remainder) == 0:
                break
            clause.append(self.oppose(np.random.choice(Ts_remainder), F))
        
        i = 0
        while i < len(clause):
            sub_clause = [clause[j] for j in range(len(clause)) if j != i]
            if np.all(self.clause_condition(self.population[Ts], sub_clause)):
                clause = sub_clause
            else:
                i += 1
        return clause

    def construct_sat(self, target_label):
        target_indices = np.where(self.target == target_label)[0]
        non_target_indices = np.where(self.target != target_label)[0]
        
        sat = []
        remaining_f = target_indices.copy()
        
        while len(remaining_f):
            F = np.random.choice(remaining_f)
            clause = self.construct_clause(non_target_indices, F)
            full_mask = self.clause_condition(self.population, clause)
            consequence = np.where(~full_mask & (self.target == target_label))[0]
            sat.append([clause, F, consequence])
            remaining_f = np.setdiff1d(remaining_f, consequence, assume_unique=True)
        return sat

    def fit(self, X_train, y_train):
        self.population = to_numpy_matrix(X_train)
        self.target = to_numpy_vector(y_train)
        self.header = self.population.shape[1]
        self.clauses = []
        targets = np.unique(self.target)
        self.lookalikes = {l : [] for l in range(self.population.shape[0])}
        t_0 = time()
        
        # Préparer TOUS les jobs (layer, target) à l'avance
        all_jobs = []
        for i in range(self.layers):
            for t in targets:
                all_jobs.append((i, t, self.population, self.target, self.header))
        
        self.qprint(f"Starting parallel computation of {len(all_jobs)} sat models across {self.n_jobs} cores...")
        
        # Paralléliser TOUS les construct_sat d'un coup
        with Pool(processes=self.n_jobs) as pool:
            all_results = pool.map(_parallel_construct_sat_with_layer, all_jobs)
        
        self.qprint(f"Parallel computation completed in {round(time()-t_0, 2)}s. Now regrouping results...")
        
        # Regrouper les résultats par layer pour respecter l'ordre
        results_by_layer = {}
        for layer_idx, target_label, sat_model in all_results:
            if layer_idx not in results_by_layer:
                results_by_layer[layer_idx] = []
            results_by_layer[layer_idx].append((target_label, sat_model))
        
        # Reconstruction séquentielle layer par layer (logique strictement identique)
        for i in range(self.layers):
            for target_label, sat_model in results_by_layer[i]:
                self.sat_models.append((target_label, sat_model))
                lookalikes = {l : [] for l in self.lookalikes if self.target[l] == target_label}
                clause_index_start = len(self.clauses)
                idx = 0
                for triple in sat_model:
                    self.clauses += [triple[0]]
                    for l in triple[2]:
                        lookalikes[l] += [clause_index_start + idx]
                    idx += 1
                for l in lookalikes:
                    self.lookalikes[l] += [lookalikes[l]]
            
            self.qprint(f"Layer {i+1}/{self.layers} regrouped.")

    def predict_proba(self, X_test):
        X_test = to_numpy_matrix(X_test)
        n_test = X_test.shape[0]
        targets = sorted(np.unique(self.target))
        
        # Matrix to accumulate class votes
        class_scores = np.zeros((n_test, len(targets)))
        clauses_masks = {idx : ~self.clause_condition(X_test, self.clauses[idx]) for idx in range(len(self.clauses))}
        for l in self.lookalikes:
            target = self.target[l]
            for array in self.lookalikes[l]:
                lookalike_mask = np.logical_and.reduce([clauses_masks[idx] for idx in array])
                class_scores[:, target] += lookalike_mask.astype(float)

        # Normalize to get a probability distribution
        row_sums = class_scores.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        
        return class_scores / row_sums

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

if __name__ == "__main__":
    # Load Digits
    print("Initializing Digits Test...")
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

    # Modeling
    clf = FastAlgorithmeClassifier(n_layers=100, vocal=True, n_jobs=-1)
    
    start_fit = time()
    clf.fit(X_train, y_train)
    print(f"Fit completed in {round(time() - start_fit, 2)}s")

    # Prediction
    start_pred = time()
    y_pred = clf.predict(X_test)
    print(f"Prediction completed in {round(time() - start_pred, 2)}s")

    # Results
    print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
