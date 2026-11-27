"""Day 3: shared data loading and scoring for the VHI baseline and the models.

Every method is scored the SAME way, on the SAME cell-months:
  - pixel rows: growing season (April-October), only pixel-months with a valid VHI
    (so the models never get credit for months the baseline cannot see)
  - a cell-month is scored when at least MIN_PIX of its pixels have a valid VHI (as in s11)
  - cell-month class = majority vote of its pixel classes. Ties go to the tied class closest
    to the mean pixel class; a remaining tie goes to the drier class
  - buffer cells are never used
Scored sets (cells from split.json x years):
  train         training cells, training years   (in-sample for the baseline climatology)
  temporal test training cells, 2021-2022        <- headline
  spatial test  test cells, training years       (shares its months with training: optimistic)
  strict test   test cells, 2021-2022            <- headline
Metrics: macro F1, quadratic-weighted kappa (ordinal: severe-as-none is worse than
severe-as-moderate), accuracy, recall per class, confusion matrix. 95% intervals by bootstrap
over year-months, because labels vary mostly between months (2021-2022 = 14 year-months).
"""
import json

import os

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, recall_score

from config import CLIMATE_DIR, DATA

MIN_PIX = 5
MIN_PIX_STRATUM = 3
CLASSES = [0, 1, 2, 3]                       # 0 severe, 1 moderate, 2 mild, 3 none
NAMES = ["severe", "moderate", "mild", "none"]
TEST_YEARS = json.loads((CLIMATE_DIR / "split.json").read_text())["test_years"]
SETS = ["train", "temporal test", "spatial test", "strict test"]


# CORAK_FEATURES=features_corr.csv.gz switches every scoring script to the s08c corrected LST (day 5)
FEATURES = os.environ.get("CORAK_FEATURES", "features.csv.gz")


def load_rows(target="spei3"):
    """Pixel-month table with label, split, set and static attributes; buffer cells dropped.
    target "spei6" swaps in the SPEI-6 labels (s16_timescale). The label value is kept in the
    column `spei3` either way, so the scoring code is unchanged."""
    f = pd.read_csv(DATA / "features" / FEATURES)
    f = f[f.VHI.notna()]
    pts = pd.read_csv(DATA / "points" / "points_flags.csv")[["pid", "cell_id", "lc", "elev", "stratum"]]
    if target == "spei3":
        lab = pd.read_csv(CLIMATE_DIR / "labels_cell_month.csv")[["cell_id", "year", "month", "spei3", "cls", "split"]]
    else:
        lab = pd.read_csv(CLIMATE_DIR / f"labels_{target}_cell_month.csv")
        lab = lab.rename(columns={target: "spei3"})[["cell_id", "year", "month", "spei3", "cls", "split"]]
    f = f.merge(pts, on="pid").merge(lab, on=["cell_id", "year", "month"])
    f = f[f.split != "buffer"].reset_index(drop=True)
    ty = f.year.isin(TEST_YEARS)
    f["set"] = np.select([(f.split == "train") & ~ty, (f.split == "train") & ty, (f.split == "test") & ~ty],
                         SETS[:3], SETS[3])
    return f


def vhi_class(v):
    """Kogan cut-offs on the clipped VHI: <20 severe, 20-30 moderate, 30-40 mild, >=40 none."""
    return np.digitize(v, [20, 30, 40])


def vote(gid, ngroups, pix_cls):
    """Majority class per group (gid = 0..ngroups-1). Returns (class, number of pixels)."""
    k = len(CLASSES)
    c = np.bincount(gid * k + pix_cls.astype(int), minlength=ngroups * k).reshape(-1, k).astype(float)
    n = c.sum(1)
    mean_cls = (c * np.array(CLASSES)).sum(1) / np.maximum(n, 1)
    top = c == c.max(axis=1, keepdims=True)
    dist = np.abs(np.array(CLASSES)[None, :] - mean_cls[:, None])
    score = np.where(top, -dist, -np.inf)           # among tied classes: closest to the mean
    return np.argmax(score, axis=1), n              # argmax takes the first = drier on a full tie


def majority(df, pred_col, min_pix=MIN_PIX):
    """Cell-month majority vote of pixel classes. Returns one row per scored cell-month."""
    g = df.groupby(["cell_id", "year", "month"], sort=True)
    pred, n = vote(g.ngroup().values, g.ngroups, df[pred_col].values)
    out = g[["cls", "set", "spei3"]].first()
    out["pred"] = pred
    out["npix"] = n.astype(int)
    return out[out.npix >= min_pix].reset_index()


def metrics(y, p):
    present = sorted(set(np.unique(y)) | set(np.unique(p)))   # a class absent from both is not scored
    return {"macro_F1": f1_score(y, p, labels=present, average="macro", zero_division=0),
            "QWK": cohen_kappa_score(y, p, labels=CLASSES, weights="quadratic"),
            "accuracy": float(np.mean(y == p)),
            **{f"recall_{n}": r for n, r in zip(NAMES, recall_score(y, p, labels=CLASSES, average=None,
                                                                     zero_division=0))}}


def bootstrap(cm, n_boot=1000, seed=0):
    """95% interval of macro F1 and QWK, resampling whole year-months."""
    rng = np.random.default_rng(seed)
    ym = (cm.year * 100 + cm.month).values
    groups = {k: np.flatnonzero(ym == k) for k in np.unique(ym)}
    keys = np.array(list(groups))
    f1s, qs = [], []
    for _ in range(n_boot):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys))])
        y, p = cm.cls.values[idx], cm.pred.values[idx]
        f1s.append(f1_score(y, p, labels=CLASSES, average="macro", zero_division=0))
        qs.append(cohen_kappa_score(y, p, labels=CLASSES, weights="quadratic"))
    return np.percentile(f1s, [2.5, 97.5]), np.percentile(qs, [2.5, 97.5])


def score_table(cm, method, boot=True):
    """One row per scored set: n, metrics, bootstrap intervals."""
    rows = []
    for s in SETS:
        d = cm[cm.set == s]
        if len(d) == 0:
            continue
        r = {"method": method, "set": s, "n_cell_months": len(d), "n_year_months": d.groupby(["year", "month"]).ngroups,
             **metrics(d.cls.values, d.pred.values)}
        if boot:
            (f_lo, f_hi), (q_lo, q_hi) = bootstrap(d)
            r.update(F1_lo=f_lo, F1_hi=f_hi, QWK_lo=q_lo, QWK_hi=q_hi)
        rows.append(r)
    return pd.DataFrame(rows)


def confusion(cm, s):
    d = cm[cm.set == s]
    return pd.DataFrame(confusion_matrix(d.cls, d.pred, labels=CLASSES),
                        index=[f"true {n}" for n in NAMES], columns=[f"pred {n}" for n in NAMES])


def stratum_table(rows, pred_col, method):
    """Scores per stratum: only that stratum's pixels vote; needs MIN_PIX_STRATUM of them."""
    out = []
    for st, d in rows.groupby("stratum"):
        cm = majority(d, pred_col, MIN_PIX_STRATUM)
        t = score_table(cm, method, boot=False)
        t.insert(1, "stratum", st)
        out.append(t)
    return pd.concat(out, ignore_index=True)
