"""UCI dataset benchmark runner. Produces holdout metrics, predictions and a reusable model."""
from __future__ import annotations
import argparse, json
import csv
import hashlib
import io
from pathlib import Path
import urllib.request
import zipfile
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, fbeta_score, mean_absolute_error, mean_squared_error, precision_score, recall_score, r2_score, roc_auc_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from ucimlrepo import fetch_ucirepo

DATASET_ID = 228
PROJECT_TITLE = "SMS Spam Triage Filter"
CLASSIFICATION_IDS = {602, 228, 159, 357, 186}

def load_data(data_dir: Path | None = None):
    if DATASET_ID == 228:
        # UCI lists this dataset in its catalog but the ucimlrepo API currently
        # does not expose its file for import. Download the official CC BY 4.0
        # archive and read the single-tab-delimited corpus file directly.
        cache_dir = data_dir or Path("data")
        corpus = cache_dir / "SMSSpamCollection"
        cache_dir.mkdir(parents=True, exist_ok=True)
        if not corpus.exists():
            url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
            with urllib.request.urlopen(url, timeout=60) as response:
                archive_bytes = response.read()
            with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
                with archive.open("SMSSpamCollection") as source:
                    corpus.write_bytes(source.read())
        rows = []
        with corpus.open(encoding="utf-8", newline="") as handle:
            for fields in csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE):
                if len(fields) >= 2 and fields[0] in {"ham", "spam"}:
                    rows.append((fields[0], "\t".join(fields[1:])))
        frame = pd.DataFrame(rows, columns=["label", "message"])
        X = frame[["message"]]
        y = frame[["label"]]
        return X.reset_index(drop=True), y.reset_index(drop=True), "random 75/25 stratified holdout; seed=42"
    dataset = fetch_ucirepo(id=DATASET_ID)
    X, y = dataset.data.features.copy(), dataset.data.targets.copy()
    split_mode = "random"
    if DATASET_ID == 357:
        dates = pd.to_datetime(X["date"], errors="coerce")
        order = dates.sort_values(kind="stable").index
        X = X.loc[order].drop(columns=[c for c in ["id", "date"] if c in X.columns])
        y, split_mode = y.loc[order], "chronological"
    if DATASET_ID == 477:
        X = X.drop(columns=[c for c in X.columns if c.lower() in {"no", "id"}], errors="ignore")
    keep = y.notna().all(axis=1)
    return X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True), split_mode

def main():
    parser = argparse.ArgumentParser(description=PROJECT_TITLE)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    X, y, split_mode = load_data(args.data_dir)
    if DATASET_ID == 228:
        run_sms_benchmark(X, y, args.output_dir, args.data_dir)
        return
    classification = DATASET_ID in CLASSIFICATION_IDS
    if DATASET_ID == 357:
        cut = int(len(X) * 0.7)
        X_train, X_test, y_train, y_test = X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]
    else:
        stratify = y.iloc[:, 0] if classification and y.shape[1] == 1 else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=stratify)
    y_train_fit = y_train.iloc[:, 0] if classification or y_train.shape[1] == 1 else y_train
    y_test_eval = y_test.iloc[:, 0] if classification or y_test.shape[1] == 1 else y_test
    text_cols = [c for c in X.columns if X[c].dtype == "object" and X[c].astype(str).str.len().mean() > 25]
    numeric = [c for c in X.select_dtypes(include=np.number).columns if c not in text_cols]
    categorical = [c for c in X.columns if c not in numeric and c not in text_cols]
    transformers = []
    if numeric:
        transformers.append(("numeric", SimpleImputer(strategy="median"), numeric))
    if categorical:
        cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
        transformers.append(("categorical", cat_pipe, categorical))
    for i, col in enumerate(text_cols):
        text_pipe = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="")), ("flatten", FunctionTransformer(lambda a: a.ravel(), validate=False)), ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True))])
        transformers.append((f"text_{i}", text_pipe, [col]))
    preprocess = ColumnTransformer(transformers, remainder="drop")
    if classification:
        estimator = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
        baseline = DummyClassifier(strategy="most_frequent")
    else:
        estimator = RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=-1)
        baseline = DummyRegressor(strategy="median")
    model = Pipeline([("preprocess", preprocess), ("model", estimator)])
    model.fit(X_train, y_train_fit)
    pred = model.predict(X_test)
    baseline.fit(X_train, y_train_fit)
    base_pred = baseline.predict(X_test)
    metrics = {"project": PROJECT_TITLE, "dataset_id": DATASET_ID, "rows": len(X), "features": list(X.columns), "targets": list(y.columns), "split": "chronological 70/30 forward holdout" if split_mode == "chronological" else split_mode if "holdout" in split_mode else "random 75/25 holdout; seed=42", "task": "classification" if classification else "regression"}
    if classification:
        metrics["holdout"] = {"accuracy": float(accuracy_score(y_test_eval, pred)), "balanced_accuracy": float(balanced_accuracy_score(y_test_eval, pred)), "macro_f1": float(f1_score(y_test_eval, pred, average="macro")), "weighted_f1": float(f1_score(y_test_eval, pred, average="weighted")), "majority_baseline_accuracy": float(accuracy_score(y_test_eval, base_pred)), "per_class": classification_report(y_test_eval, pred, output_dict=True, zero_division=0)}
        if len(np.unique(y_test_eval)) == 2:
            metrics["holdout"]["roc_auc"] = float(roc_auc_score(y_test_eval, model.predict_proba(X_test)[:, 1]))
    else:
        actual = np.asarray(y_test_eval)
        prediction = np.asarray(pred)
        base_prediction = np.asarray(base_pred)
        if actual.ndim == 1: actual = actual.reshape(-1, 1)
        if prediction.ndim == 1: prediction = prediction.reshape(-1, 1)
        if base_prediction.ndim == 1: base_prediction = base_prediction.reshape(-1, 1)
        metrics["holdout"] = {str(y.columns[i]): {"mae": float(mean_absolute_error(actual[:, i], prediction[:, i])), "rmse": float(np.sqrt(mean_squared_error(actual[:, i], prediction[:, i]))), "r2": float(r2_score(actual[:, i], prediction[:, i])), "median_baseline_mae": float(mean_absolute_error(actual[:, i], base_prediction[:, i]))} for i in range(actual.shape[1])}
    metrics["limitations"] = "Public benchmark; holdout results do not establish production performance." + (" Telescope events are Monte Carlo simulations." if DATASET_ID == 159 else "")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    actual = np.asarray(y_test_eval)
    prediction = np.asarray(pred)
    if actual.ndim == 1: actual = actual.reshape(-1, 1)
    if prediction.ndim == 1: prediction = prediction.reshape(-1, 1)
    predictions = pd.DataFrame({f"actual_{y.columns[i]}": actual[:, i] for i in range(actual.shape[1])})
    for i in range(prediction.shape[1]): predictions[f"predicted_{y.columns[i]}"] = prediction[:, i]
    predictions.to_csv(args.output_dir / "holdout_predictions.csv", index=False)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump(model, args.output_dir / "model.joblib")
    print(json.dumps(metrics, indent=2))

def run_sms_benchmark(X: pd.DataFrame, y: pd.DataFrame, output_dir: Path, data_dir: Path) -> None:
    """Text-focused model selection with train-only threshold choice and a final holdout."""
    X_train, X_test, y_train, y_test = train_test_split(
        X["message"].astype(str), y["label"].astype(str),
        test_size=0.25, random_state=42, stratify=y["label"].astype(str),
    )
    model = Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True)),
            ("character", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, max_features=100_000, sublinear_tf=True)),
        ])),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=2000, solver="liblinear", random_state=42)),
    ])
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = cross_val_predict(model, X_train, y_train, cv=folds, method="predict_proba", n_jobs=1)
    oof_spam = oof[:, list(sorted(y_train.unique())).index("spam")]
    precision, recall, thresholds = precision_recall_curve((y_train == "spam").astype(int), oof_spam)
    f2 = (5 * precision[:-1] * recall[:-1]) / (4 * precision[:-1] + recall[:-1] + 1e-12)
    threshold_index = int(np.argmax(f2))
    threshold = float(thresholds[threshold_index])
    model.fit(X_train, y_train)
    spam_index = list(model.named_steps["classifier"].classes_).index("spam")
    scores = model.predict_proba(X_test)[:, spam_index]
    predictions = np.where(scores >= threshold, "spam", "ham")
    default_predictions = np.where(scores >= 0.5, "spam", "ham")
    y_binary = (y_test == "spam").astype(int)
    dataset_path = data_dir / "SMSSpamCollection"
    metrics = {
        "project": PROJECT_TITLE,
        "dataset_id": DATASET_ID,
        "dataset": "UCI SMS Spam Collection",
        "rows": int(len(X)),
        "features": ["message text (word and character TF-IDF)"],
        "targets": ["label"],
        "split": "stratified 75/25 train/holdout; seed=42; threshold tuned on 5-fold out-of-fold training predictions",
        "model": "word+character TF-IDF with class-weighted logistic regression",
        "threshold_objective": "maximize F2 for the spam class on training-only out-of-fold predictions",
        "spam_threshold": threshold,
        "holdout": {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
            "spam_precision": float(precision_score(y_test, predictions, pos_label="spam", zero_division=0)),
            "spam_recall": float(recall_score(y_test, predictions, pos_label="spam", zero_division=0)),
            "spam_f2": float(fbeta_score(y_test, predictions, beta=2, pos_label="spam", zero_division=0)),
            "average_precision": float(average_precision_score(y_binary, scores)),
            "roc_auc": float(roc_auc_score(y_binary, scores)),
            "confusion_matrix_labels_ham_spam": confusion_matrix(y_test, predictions, labels=["ham", "spam"]).tolist(),
            "default_0_5_spam_recall": float(recall_score(y_test, default_predictions, pos_label="spam", zero_division=0)),
            "default_0_5_spam_precision": float(precision_score(y_test, default_predictions, pos_label="spam", zero_division=0)),
            "majority_baseline_accuracy": float(accuracy_score(y_test, np.full(len(y_test), "ham"))),
        },
        "source_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "limitations": "Random holdout from a non-chronological corpus; threshold is selected for a research demo, not a deployed spam filter. Messages may contain personal information; the report stores labels and scores only, not message text.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"actual_label": y_test.to_numpy(), "spam_score": scores, "predicted_label": predictions}).to_csv(output_dir / "holdout_predictions.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump(model, output_dir / "model.joblib")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()

