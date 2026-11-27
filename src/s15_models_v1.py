"""Day 3, step 2, v1 (kept for the record; superseded by s15_models.py).

NOTE: v1 ran on the day-2 features, now saved as data/features/features_v1.csv.gz.
s10 has since been rebuilt with leave-one-year-out climatology, so running this file now
reads the new features.csv.gz and will NOT reproduce the v1 numbers (outputs/*_v1_*).

Model A (Landsat) and Model B (Landsat + Sentinel-2), pixel level, scored per cell-month.

  python s15_models.py xgb A      Model A, XGBoost, 2014-2026
  python s15_models.py rf  A      Model A, Random Forest
  python s15_models.py xgb A19    Model A features on 2019-2026 rows only (fair partner for B)
  python s15_models.py xgb B      Model B: A features + Sentinel-2, 2019-2026
(RF and XGBoost run as separate processes: each brings its own OpenMP runtime.)

Leakage rules:
  - fit rows: TRAINING cells x TRAINING years only. Test years / test cells are scored once, at the end.
  - tuning: leave-one-YEAR-out CV on the training rows (labels vary almost only by month, so
    random or pixel folds would let a model memorise each month from neighbouring pixels).
    The best setting by mean CV macro F1 (cell-month majority vote) is frozen to
    outputs/models/<name>_config.json BEFORE the test sets are predicted.
  - excluded on purpose: year, cell_id, coordinates, scene counts (NOBS*), L2 LST (has the
    emissivity gap), strata / irrigation flag (a reporting device, not a measurement).
Rows: pixel-months with a valid VHI (same as the baseline). Up to CAP random pixels per cell-month
are used for fitting (pixels in one cell-month share one label, so more adds little but time).
Classes are weighted inversely to their frequency.
Feature sets compared in CV:
  pixel   - the pixel's own indices, z-anomalies, previous-month z, VCI/TCI/VHI, month, elev, land cover
  +cell   - plus the cell-month mean of the z-anomalies and VHI over the cell's valid pixels
            (satellite data only, no labels; it matches the 5 km label scale)
"""
import json
import sys
import time

import numpy as np
import pandas as pd

from config import DATA, ROOT
from s13_eval import CLASSES, TEST_YEARS, majority, metrics, score_table, stratum_table, confusion

ALGO, NAME = sys.argv[1], sys.argv[2]
SEED, CAP = 0, 12
MODELS = ROOT / "outputs" / "models"
TAB = ROOT / "outputs" / "tables"
PRED = DATA / "preds"
for d in (MODELS, TAB, PRED):
    d.mkdir(parents=True, exist_ok=True)

LS = ["NDVI", "EVI", "SAVI", "NDMI", "LST_TOA"]
S2 = ["S2_NDVI", "S2_EVI", "S2_NDMI", "S2_NDRE"]
PIX = (LS + [f"{v}_z" for v in LS] + [f"{v}_z_lag1" for v in LS] + ["VCI", "TCI", "VHI_clip", "month", "elev", "lc"])
CELL_VARS = [f"{v}_z" for v in LS] + ["VHI_clip"]
if NAME == "B":
    PIX += S2 + [f"{v}_z" for v in S2] + [f"{v}_z_lag1" for v in S2]
    CELL_VARS += [f"{v}_z" for v in S2]
FIRST_YEAR = 2019 if NAME in ("A19", "B") else 2014

from s13_eval import load_rows  # noqa: E402

rows = load_rows()
rows = rows[rows.year >= FIRST_YEAR].reset_index(drop=True)
cellmean = rows.groupby(["cell_id", "year", "month"])[CELL_VARS].mean().add_prefix("cell_")
rows = rows.join(cellmean, on=["cell_id", "year", "month"])
CELL = list(cellmean.columns)
FEATSETS = {"pixel": PIX, "pixel+cell": PIX + CELL}

train = rows[rows.set == "train"]
rng = np.random.default_rng(SEED)
fit = (train.assign(r=rng.random(len(train))).sort_values("r")
       .groupby(["cell_id", "year", "month"]).head(CAP).sort_index())
w_cls = len(fit) / (len(CLASSES) * np.bincount(fit.cls, minlength=4))
print(f"{NAME}/{ALGO}: {len(rows)} rows, fit rows {len(fit)} (cap {CAP}/cell-month), "
      f"training years {sorted(fit.year.unique())}, class weights {np.round(w_cls, 2)}", flush=True)


def make(params):
    if ALGO == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(objective="multi:softprob", num_class=4, tree_method="hist", n_jobs=8,
                             random_state=SEED, learning_rate=0.05, subsample=0.7, colsample_bytree=0.7,
                             **params)
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_jobs=8, random_state=SEED, max_features="sqrt", max_samples=0.5, **params)


GRID = ({"xgb": [dict(n_estimators=300, max_depth=3, min_child_weight=50, reg_lambda=5),
                 dict(n_estimators=300, max_depth=5, min_child_weight=50, reg_lambda=5),
                 dict(n_estimators=150, max_depth=3, min_child_weight=200, reg_lambda=10)],
         "rf": [dict(n_estimators=200, min_samples_leaf=50),
                dict(n_estimators=200, min_samples_leaf=300)]})[ALGO]


def predict(model, X):
    return np.argmax(model.predict_proba(X), axis=1)


def fit_model(params, df, feats):
    m = make(params)
    m.fit(df[feats], df.cls, sample_weight=w_cls[df.cls.values])
    return m


# ---- leave-one-year-out CV on training rows
cfg_path = MODELS / f"{NAME}_{ALGO}_config.json"
cv_rows = []
years = sorted(fit.year.unique())
for fs_name, feats in FEATSETS.items():
    for i, params in enumerate(GRID):
        t0, f1s, qwks = time.time(), [], []
        for y in years:
            tr_, va = fit[fit.year != y], train[train.year == y].copy()
            if va.cls.nunique() < 2:
                continue
            m = fit_model(params, tr_, feats)
            va["p"] = predict(m, va[feats])
            cm = majority(va, "p")
            s = metrics(cm.cls.values, cm.pred.values)
            f1s.append(s["macro_F1"]); qwks.append(s["QWK"])
        cv_rows.append({"featset": fs_name, "config": i, **params, "cv_macro_F1": np.mean(f1s),
                        "cv_F1_sd": np.std(f1s), "cv_QWK": np.mean(qwks), "folds": len(f1s)})
        print(f"  CV {fs_name:10s} cfg {i}: macro F1 {np.mean(f1s):.3f} (sd {np.std(f1s):.3f}), "
              f"QWK {np.mean(qwks):.3f}, {time.time() - t0:.0f} s", flush=True)
cv = pd.DataFrame(cv_rows)
cv.to_csv(TAB / f"{NAME}_{ALGO}_cv.csv", index=False)
best = cv.sort_values("cv_macro_F1", ascending=False).iloc[0]
params = GRID[int(best.config)]
feats = FEATSETS[best.featset]
cfg = {"name": NAME, "algo": ALGO, "featset": best.featset, "features": feats, "params": params,
       "cap": CAP, "first_year": FIRST_YEAR, "test_years": TEST_YEARS, "class_weights": w_cls.tolist(),
       "cv_macro_F1": float(best.cv_macro_F1), "cv_QWK": float(best.cv_QWK)}
cfg_path.write_text(json.dumps(cfg, indent=2))
print(f"frozen: {best.featset}, config {int(best.config)} -> {cfg_path.name}", flush=True)

# ---- final fit on all training rows (capped), then score every set once
model = fit_model(params, fit, feats)
if ALGO == "xgb":
    model.save_model(MODELS / f"{NAME}_{ALGO}.json")
else:
    import joblib
    joblib.dump(model, MODELS / f"{NAME}_{ALGO}.joblib", compress=3)
rows["p"] = predict(model, rows[feats])
cm = majority(rows, "p")
label = f"{NAME} {ALGO}"
scores = score_table(cm, label)
scores.to_csv(TAB / f"{NAME}_{ALGO}_scores.csv", index=False)
stratum_table(rows, "p", label).to_csv(TAB / f"{NAME}_{ALGO}_strata.csv", index=False)
cm.to_csv(PRED / f"{NAME}_{ALGO}_cell_month.csv", index=False)
with open(TAB / f"{NAME}_{ALGO}_confusion.txt", "w") as fh:
    for s in ["temporal test", "strict test"]:
        fh.write(f"{label} - {s}\n{confusion(cm, s).to_string()}\n\n")
imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
imp.to_csv(TAB / f"{NAME}_{ALGO}_importance.csv", header=["importance"])
pd.set_option("display.width", 200)
print(scores[["set", "n_cell_months", "macro_F1", "F1_lo", "F1_hi", "QWK", "QWK_lo", "QWK_hi", "accuracy",
              "recall_severe", "recall_moderate", "recall_mild", "recall_none"]].round(3).to_string(index=False))
print("top features:", imp.head(10).round(3).to_dict())
