import json
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import load
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent            # .../src
ROOT = HERE.parent  

RUN_DIR = ROOT / "runs" / "cat_run_01"

MODEL_PATH = ROOT / "runs" / "cat_run_01" / "model.joblib"
COLS_PATH = ROOT / "runs" / "cat_run_01" / "train_columns.json"

TEST_FEAT = ROOT / "data" / "feature_test.csv"
TEST_OUT = ROOT / "data" / "outcome_test.csv"

X_test = pd.read_csv(TEST_FEAT)
y_test  = pd.read_csv(TEST_OUT).squeeze()
y_test  = y_test.astype(int)


def prob_predictions():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not COLS_PATH.exists():
        raise FileNotFoundError(f"Train columns not found: {COLS_PATH}")

    model = load(MODEL_PATH)

    with open(COLS_PATH) as f:
        train_columns = json.load(f)

    X_test_final = X_test.reindex(columns=train_columns)

    probs = model.predict_proba(X_test_final)[:, 1]

    return probs


def best_threshold():
    best_score = 0 # worst possible f1 score
    final_threshold = 0.5
    all_threshold = np.linspace(0.01, 0.99, 99)
    preds = prob_predictions()
    for i in all_threshold:
        y_pred = (preds >= i).astype(int)
        score_temp = f1_score(y_test, y_pred)

        if score_temp > best_score:
            best_score = score_temp
            final_threshold = i

    return final_threshold, best_score


def main():

    probs = prob_predictions()
    threshold, score = best_threshold()

    preds = (probs >= threshold).astype(int)

    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    pd.DataFrame([{"precision": precision, "recall": recall, "f1": f1, "AUC": auc, "threshold": threshold}]).to_csv(
        RUN_DIR / "test_metrics.csv", index=False
    )

    out_preds = X_test.copy()
    out_preds["actual"] = y_test
    out_preds["predictions"] = preds
    out_preds.to_csv(RUN_DIR / "test_predictions.csv", index=False)

    print({"precision": precision, "recall": recall, "f1": f1, "AUC": auc, "threshold": threshold})
    print(f"Saved predictions -> {RUN_DIR/'test_predictions.csv'}")
    print(f"Saved metrics -> {RUN_DIR/'test_metrics.csv'}")


if __name__ == "__main__":
    main()