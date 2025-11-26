import os, json
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args
import numpy as np


HERE = Path(__file__).resolve().parent            # .../src
ROOT = HERE.parent                                # project root

RUN_DIR = ROOT / "runs" / "cat_run_01"

TRAIN_FEAT = ROOT / "data" / "feature_train.csv"
TRAIN_OUT = ROOT / "data" / "outcome_train.csv"
VAL_FEAT = ROOT / "data" / "feature_val.csv"
VAL_OUT = ROOT / "data" / "outcome_val.csv"
TEST_FEAT = ROOT / "data" / "feature_test.csv"
TEST_OUT = ROOT / "data" / "outcome_test.csv"

cat_features = ['person_home_ownership', 'loan_intent',
                'loan_grade', 'cb_person_default_on_file']

X_train = pd.read_csv(TRAIN_FEAT)
X_val = pd.read_csv(VAL_FEAT)
X_test = pd.read_csv(TEST_FEAT)

y_train = pd.read_csv(TRAIN_OUT).squeeze()   # Series instead of DataFrame
y_test  = pd.read_csv(TEST_OUT).squeeze()
y_val   = pd.read_csv(VAL_OUT).squeeze()

y_train = y_train.astype(int)
y_val   = y_val.astype(int)
y_test  = y_test.astype(int)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def get_weight():
    neg = y_train.value_counts()[0]
    pos = y_train.value_counts()[1]

    pos_weight = neg / pos

    class_weights = [1.0, pos_weight]

    return class_weights


SPACE = [
    Integer(4, 8, name='depth'),
    Real(0.01, 0.2, prior='log-uniform', name='learning_rate'),
    Real(1.0, 5.0, prior='log-uniform', name='l2_leaf_reg'),
    Real(0.0, 1.0, name='bagging_temperature'), # Adds randomness in sampling rows when building each tree. 
    Real(0.0, 5.0, name='random_strength') # Adds noise to split score evaluation. Higher → more randomness in tree structure → less overfitting.
]


class_weight = get_weight()

fixed_params = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'iterations': 2000,             
    'od_type': 'Iter',     # enable overfitting detector
    'od_wait': 50,
    'class_weights': class_weight,    
    'random_seed': 42,
    'verbose': False
}

def main():

    ensure_dir(RUN_DIR)

    @use_named_args(SPACE) # Transforms positional values → named keyword arguments. Like a dictionary
    def objective(depth, learning_rate, l2_leaf_reg,
                bagging_temperature, random_strength):

        params = fixed_params.copy()
        params.update({
            'depth': int(depth),
            'learning_rate': float(learning_rate),
            'l2_leaf_reg': float(l2_leaf_reg),
            'bagging_temperature': float(bagging_temperature),
            'random_strength': float(random_strength),
        })

        model = CatBoostClassifier(**params)

        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            cat_features=cat_features,
            verbose=False
        )

        probs = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, probs)

        return -auc
    

    res = gp_minimize(
        func=objective,
        dimensions=SPACE,
        n_calls=25,              # total evaluations
        n_initial_points=5,      # random starts
        random_state=42,
        verbose=True
    )

    best = {
        'depth': int(res.x[0]),
        'learning_rate': float(res.x[1]),
        'l2_leaf_reg': float(res.x[2]),
        'bagging_temperature': float(res.x[3]),
        'random_strength': float(res.x[4]),
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "class_weights": class_weight, 
        "random_seed": 42,
        "verbose": False
    }

    # Save artifacts
    best_path_json = os.path.join(RUN_DIR, "best_params.json") 
    best_path_csv  = os.path.join(RUN_DIR, "best_params.csv")
    hist_path_csv  = os.path.join(RUN_DIR, "opt_history.csv")

    with open(best_path_json, "w") as f: # will be used by train.py
        json.dump(best, f, indent=2)

    pd.DataFrame([best]).to_csv(best_path_csv, index=False)
    pd.DataFrame({"iteration": np.arange(len(res.func_vals)), "val_auc": -res.func_vals}).to_csv(
        hist_path_csv, index=False
    )


    print("\nOptimization complete")
    print(f"Best Val AUC: {-res.fun:.6f}")
    print(f"Saved best params -> {best_path_json}")
    print(f"Trials history -> {hist_path_csv}")

if __name__ == "__main__":
    main()