import os, json
import pandas as pd
from pathlib import Path
from joblib import dump
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier

HERE = Path(__file__).resolve().parent            # .../src
ROOT = HERE.parent                                # project root

TRAIN_FEAT = ROOT / "data" / "feature_train.csv"
TRAIN_OUT = ROOT / "data" / "outcome_train.csv"
VAL_FEAT = ROOT / "data" / "feature_val.csv"
VAL_OUT = ROOT / "data" / "outcome_val.csv"
TEST_FEAT = ROOT / "data" / "feature_test.csv"
TEST_OUT = ROOT / "data" / "outcome_test.csv"

X_train = pd.read_csv(TRAIN_FEAT)
X_val = pd.read_csv(VAL_FEAT)
X_test = pd.read_csv(TEST_FEAT)

y_train = pd.read_csv(TRAIN_OUT).squeeze()   # Series instead of DataFrame
y_test  = pd.read_csv(TEST_OUT).squeeze()
y_val   = pd.read_csv(VAL_OUT).squeeze()

y_train = y_train.astype(int)
y_val   = y_val.astype(int)
y_test  = y_test.astype(int)

cat_features = ['person_home_ownership', 'loan_intent',
                'loan_grade', 'cb_person_default_on_file']

RUN_DIR = ROOT / "runs" / "cat_run_01"
BEST_HP = ROOT / "runs" / "cat_run_01" / "best_params.json"


with open(BEST_HP) as f:
    CAT_PARAMS = json.load(f)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def fit_model():

    model = CatBoostClassifier(**CAT_PARAMS)
    eval_set = [(X_val, y_val)]
    model.fit(X_train, y_train, eval_set=eval_set, 
              cat_features=cat_features, verbose=False)
    return model, model.get_evals_result()


def plot_history(metric='Logloss'):
    model, res = fit_model()  

    train_metric = res['learn'][metric]
    val_metric   = res['validation'][metric]

    epochs = len(train_metric)
    x_axis = range(epochs)

    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, train_metric, label=f'Train {metric}')
    plt.plot(x_axis, val_metric, label=f'Validation {metric}')
    plt.xlabel("Boosting Round")
    plt.ylabel(metric)
    plt.title(f"Training vs Validation {metric}")
    plt.legend()
    plt.show()



def main():
    ensure_dir(RUN_DIR)

    model, evals_result = fit_model()

    # model.joblib contains all model trees, learned weights, hyperparameters etc
    # this is what will be used in evaluate.py, production deployments etc
    dump(model, os.path.join(RUN_DIR, "model.joblib"))
    # train_columns.json = the exact list of feature columns
    with open(os.path.join(RUN_DIR, "train_columns.json"), "w") as f:
        json.dump(list(X_train.columns), f, indent=2)

    # evals_result.json includes per-iteration training & validation metrics
    with open(os.path.join(RUN_DIR, "evals_result.json"), "w") as f:
        json.dump(evals_result, f, indent=2)

    print(f"Model saved -> {os.path.join(RUN_DIR,'model.joblib')}")
    print(f"Columns saved -> {os.path.join(RUN_DIR,'train_columns.json')}")

if __name__ == "__main__":
    main()