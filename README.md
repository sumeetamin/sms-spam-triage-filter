# SMS Spam Triage Filter

Rank and triage spam messages using word and character text features.

## Dataset

[UCI SMS Spam Collection dataset](https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection) (dataset ID 228), 5,574 records. The source is public and is fetched by the runner; the raw dataset is not committed.

## Run

```bash
python -m pip install -r requirements.txt
python main.py --output-dir artifacts
```

The runner saves `artifacts/metrics.json`, `artifacts/holdout_predictions.csv`, and `artifacts/model.joblib`. For SMS, the UCI CC BY 4.0 archive is downloaded to the ignored `data/` cache because the repository's importer does not currently expose that dataset. The raw messages are never written to the report.

## Holdout results

Word and character TF-IDF features feed a class-weighted logistic-regression model. Five-fold out-of-fold scores on the training partition select a threshold that maximizes spam-class F2; the final holdout is untouched during selection. stratified 75/25 train/holdout; seed=42; threshold tuned on 5-fold out-of-fold training predictions.

| Accuracy | Balanced accuracy | Macro F1 | Weighted F1 | Majority baseline accuracy |
|---:|---:|---:|---:|---:|
| 0.985 | 0.980 | NaN | NaN | 0.866 |

ROC-AUC: **0.995**.

Full aggregate metrics and the exact split seed are in [`reports/benchmark.json`](reports/benchmark.json).

## Limits

This historical corpus is not chronological. A research threshold is not a production policy. Public SMS messages can contain personal details; only aggregate metrics and message-free predictions are published. These are one-time benchmark results, not service-level, production, or causal claims.

## Reproducibility

The model and split use seed 42. Re-running downloads source data from UCI and regenerates local artifacts. Model binaries, raw inputs, and row-level predictions are excluded from this public repository.
