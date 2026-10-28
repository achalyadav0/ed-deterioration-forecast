# Running on Kaggle

Code lives in this GitHub repo; Kaggle only *runs* it. Two notebooks are chained so preprocessing
is never repeated when you retrain.

## One-time setup

1. **Private dataset.** Upload the MC-MED CSVs you need (`visits.csv`, `numerics.csv`,
   `split_chrono_*.csv`) as a **private** Kaggle dataset, e.g. `mc-med`. Check your PhysioNet data
   use agreement before uploading anywhere.
2. **GitHub token** (if the repo is private). Create a fine-grained token with read access to this
   repo, and add it in a notebook under *Add-ons → Secrets* as `GITHUB_TOKEN`.
3. In both notebooks, set `REPO` in the first cell to `github.com/<you>/ed-deterioration-forecast.git`.

## Notebook 01: preprocess (CPU session is enough)

`notebooks/01_preprocess_kaggle.ipynb`. Attach the MC-MED dataset and run all. It caches
ingest → featurize → label → split → index under `/kaggle/working/work`. Then
**Save Version → Save & Run All** so the output becomes an attachable dataset.

## Notebook 02: train, evaluate, export (GPU session)

`notebooks/02_train_evaluate_kaggle.ipynb`. Attach the MC-MED dataset **and** notebook 01's output,
and set `WORK` to that output's `work` folder. The data stages are found in the cache, so it goes
straight to training.

* Keep `RAW` identical in both notebooks: it is part of the cache key.
* Training checkpoints every epoch (`last.pt`). If the session ends, rerun the cell; it resumes.
* XGBoost and the baselines need no GPU.
* Optional: `train.tracker=wandb` with `WANDB_API_KEY` set from Kaggle Secrets. Only configs and
  aggregate metrics are sent.

## Changing preprocessing

Edit the config (or pass an override) and rerun notebook 01. Only the stages whose config changed,
and those after them, are recomputed. Then point notebook 02 at the new output version.
