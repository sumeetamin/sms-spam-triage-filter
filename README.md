# SMS Spam Triage Filter

Rank and triage spam messages using word and character text features.

## Dataset

[UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection) (dataset ID 228), 5,574 records. The source is public and fetched by the runner; the raw dataset is not committed.

## Run

```bash
python -m pip install -r requirements.txt
python main.py --output-dir artifacts
```

## Holdout results

The word and character TF-IDF features feed a class-weighted logistic-regression model. Exact duplicate message strings are kept in the same partition. A 5-fold stratified group cross-validation on the training partition chooses the spam threshold to maximize F2; the final holdout is not used in threshold selection.

| Accuracy | Balanced accuracy | Spam precision | Spam recall | Spam F2 | Majority baseline accuracy |
|---:|---:|---:|---:|---:|---:|
| 0.991 | 0.986 | 0.953 | 0.979 | 0.973 | 0.866 |

ROC-AUC: **0.994**. Exact message-string overlap across train and holdout: **0**. Full aggregate metrics, confusion matrix and source fingerprint are in [reports/benchmark.json](reports/benchmark.json).

## Limits

This is a random grouped holdout from a historical, non-chronological corpus. Near-duplicate paraphrases are not grouped. The threshold is a research choice, not a deployed spam-filter policy. Public SMS messages may contain personal details; raw messages and text predictions remain local. This single benchmark is not a service-level or production guarantee.

## Reproducibility

The split uses seed 42. Re-running fetches the dataset from UCI and regenerates local artifacts. Model binaries, raw inputs, and row-level predictions are excluded from this public repository.
