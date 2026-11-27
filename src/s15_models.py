"""Day 3, step 2 (v2): model design chosen by leave-one-year-out CV against references.

Why v2: v1 (s15_models_v1.py, outputs *_v1_*) was scored once on 2021-2022 and lost to "always
none". It had learned the calendar (late summer is often droughty in the training years) through
`month` and the raw index/temperature levels. v2 changes, all decided on training years only:
  - features rebuilt with leave-one-year-out climatology (s10), so a held-out year is truly held out
  - anomaly-only features: no month, no raw levels
  - every candidate is compared, fold by fold, with reference rules in the same CV

  python s15_models.py A     Landsat, training years 2014-2020, 2023-2026 (11 folds)
  python s15_models.py A19   same features, 2019+ only (fair partner for B)
  python s15_models.py B     + Sentinel-2, 2019+ (6 folds)
  python s15_models.py A spei6   side check against SPEI-6 classes (CV only, no test scoring)
  python s15_models.py A rf      Random Forest candidates C4/C5 instead of C1-C3 (day 5)

CV: for each training year y, fit on the other training years (training cells), predict
training cells in year y, score per cell-month (majority vote, s13_eval). Folds = years.
References (no fitting, or fitted on the other years):
  always none | calendar (most common class per calendar month) | VHI Kogan 20/30/40 |
  VHI tuned (cut-offs re-fitted on the other years, grid step 5)
Candidates, declared before running (fixed settings, no further tuning):
  C1 XGB classifier, anomaly + cell-mean features, class-balanced weights
  C2 XGB regressor of SPEI-3, anomaly + cell-mean features, prediction cut at -1.5/-1.0/-0.5
  C3 XGB regressor of SPEI-3, anomaly features only (no cell means)
Day 5 (Fri 27 Nov), Random Forest, declared before running, same references and pass rule:
  C4 RF classifier, anomaly + cell-mean features, class_weight="balanced"
  C5 RF regressor of SPEI-3, anomaly + cell-mean features, prediction cut as C2
  both: 300 trees, min_samples_leaf=100, max_features="sqrt", max_samples=0.7, seed 0
PASS RULE (declared before running): a candidate passes if its mean CV macro F1 is at least
0.03 above the best reference's AND it beats that reference in at least 2/3 of the folds.
The passing candidate with the best mean CV macro F1 is frozen and fitted on all training rows;
the test sets are then scored once more, reported beside v1 as a second (exploratory) look.
If none passes, no model is re-scored on the test years: the result is the reference table.
"""
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from config import DATA, ROOT, SPEI_CUTS
from s13_eval import (CLASSES, MIN_PIX, TEST_YEARS, confusion, load_rows, majority, metrics, score_table,
                      stratum_table, vote)

NAME = sys.argv[1]
ARGS = sys.argv[2:]
RF = "rf" in ARGS                                         # day 5: Random Forest candidates
TARGET = next((a for a in ARGS if a.startswith("spei")), "spei3")   # "spei6": side check, CV only
TAG = f"{NAME}_v2" if TARGET == "spei3" else f"{NAME}_v2_{TARGET}"
if RF:
    TAG += "_rf"
if os.environ.get("CORAK_FEATURES", "features.csv.gz") != "features.csv.gz":
    TAG += "_lstcorr"                                        # day 5: corrected LST features
SEED, CAP = 0, 12
MODELS, TAB, PRED = ROOT / "outputs" / "models", ROOT / "outputs" / "tables", DATA / "preds"
for d in (MODELS, TAB, PRED):
    d.mkdir(parents=True, exist_ok=True)

LS = ["NDVI", "EVI", "SAVI", "NDMI", "LST_TOA"]
S2 = ["S2_NDVI", "S2_EVI", "S2_NDMI", "S2_NDRE"]
ANOM = [f"{v}_z" for v in LS] + [f"{v}_z_lag1" for v in LS] + ["VCI", "TCI", "VHI_clip", "elev", "lc"]
CELL_VARS = [f"{v}_z" for v in LS] + ["VHI_clip"]
if NAME == "B":
    ANOM += [f"{v}_z" for v in S2] + [f"{v}_z_lag1" for v in S2]
    CELL_VARS += [f"{v}_z" for v in S2]
FIRST_YEAR = 2019 if NAME in ("A19", "B") else 2014
RFP = dict(n_estimators=300, min_samples_leaf=100, max_features="sqrt", max_samples=0.7, n_jobs=-1,
           random_state=SEED)
XGB = dict(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.7, colsample_bytree=0.7,
           min_child_weight=100, reg_lambda=10, tree_method="hist", n_jobs=8, random_state=SEED)

rows = load_rows(TARGET)
rows = rows[rows.year >= FIRST_YEAR].reset_index(drop=True)
cellmean = rows.groupby(["cell_id", "year", "month"])[CELL_VARS].mean().add_prefix("cell_")
rows = rows.join(cellmean, on=["cell_id", "year", "month"])
CELL = list(cellmean.columns)
CANDIDATES = {"C1 classifier +cell": ("cls", ANOM + CELL), "C2 regressor +cell": ("reg", ANOM + CELL),
              "C3 regressor pixel": ("reg", ANOM)}
if RF:
    CANDIDATES = {"C4 RF classifier +cell": ("rf_cls", ANOM + CELL), "C5 RF regressor +cell": ("rf_reg", ANOM + CELL)}

train = rows[rows.set == "train"]
rng = np.random.default_rng(SEED)
fit_all = (train.assign(r=rng.random(len(train))).sort_values("r")
           .groupby(["cell_id", "year", "month"]).head(CAP).sort_index())
years = sorted(train.year.unique())
print(f"{NAME}: {len(rows)} rows, fit rows {len(fit_all)}, folds {years}", flush=True)


def fit_predict(kind, feats, fit, X):
    """Fit on `fit`, return pixel classes for X."""
    if kind == "rf_cls":
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(class_weight="balanced", **RFP).fit(fit[feats], fit.cls)
        return m, m.predict(X[feats])
    if kind == "rf_reg":
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(**RFP).fit(fit[feats], fit.spei3)
        return m, np.digitize(m.predict(X[feats]), SPEI_CUTS)
    if kind == "cls":
        from xgboost import XGBClassifier
        w = len(fit) / (4 * np.bincount(fit.cls, minlength=4))
        m = XGBClassifier(objective="multi:softprob", num_class=4, **XGB)
        m.fit(fit[feats], fit.cls, sample_weight=w[fit.cls.values])
        return m, np.argmax(m.predict_proba(X[feats]), axis=1)
    from xgboost import XGBRegressor
    m = XGBRegressor(objective="reg:squarederror", **XGB)
    m.fit(fit[feats], fit.spei3)
    return m, np.digitize(m.predict(X[feats]), SPEI_CUTS)


def tune_cuts(df):
    g = df.groupby(["cell_id", "year", "month"], sort=True)
    gid, ng, y = g.ngroup().values, g.ngroups, g.cls.first().values
    best = (-1, (20, 30, 40))
    for cuts in itertools.combinations(np.arange(10, 65, 5), 3):
        p, n = vote(gid, ng, np.digitize(df.VHI_clip.values, cuts))
        f1 = metrics(y[n >= MIN_PIX], p[n >= MIN_PIX])["macro_F1"]
        if f1 > best[0]:
            best = (f1, cuts)
    return best[1]


def score(va, pix):
    va = va.assign(p=pix)
    cm = majority(va, "p")
    return metrics(cm.cls.values, cm.pred.values)


# ---- leave-one-year-out CV
res = []
for y in years:
    t0 = time.time()
    tr_, va = fit_all[fit_all.year != y], train[train.year == y]
    other = train[train.year != y]
    cal = other.groupby(["cell_id", "year", "month"]).cls.first().reset_index()
    cal = cal.groupby("month").cls.agg(lambda s: s.value_counts().idxmax())
    pred = {"always none": np.full(len(va), 3),
            "calendar": va.month.map(cal).values,
            "VHI Kogan": np.digitize(va.VHI_clip, [20, 30, 40]),
            "VHI tuned": np.digitize(va.VHI_clip, tune_cuts(other))}
    for name, (kind, feats) in CANDIDATES.items():
        pred[name] = fit_predict(kind, feats, tr_, va)[1]
    for name, p in pred.items():
        s = score(va, p)
        res.append({"year": y, "method": name, "macro_F1": s["macro_F1"], "QWK": s["QWK"]})
    print(f"  fold {y}: " + ", ".join(f"{r['method']} {r['macro_F1']:.2f}" for r in res if r["year"] == y)
          + f"  ({time.time() - t0:.0f} s)", flush=True)
cv = pd.DataFrame(res)
cv.to_csv(TAB / f"{TAG}_cv_folds.csv", index=False)
wide = cv.pivot(index="year", columns="method", values="macro_F1")
summary = cv.groupby("method")[["macro_F1", "QWK"]].agg(["mean", "std"]).round(3)
refs = ["always none", "calendar", "VHI Kogan", "VHI tuned"]
best_ref = wide[refs].mean().idxmax()
summary["beats_best_ref_folds"] = [(wide[m] > wide[best_ref]).sum() for m in summary.index]
summary.to_csv(TAB / f"{TAG}_cv_summary.csv")
pd.set_option("display.width", 200)
print(f"\nCV summary ({len(years)} folds), best reference: {best_ref}\n{summary.to_string()}")
print("\nmacro F1 per fold:\n" + wide.round(2).to_string())

passing = [m for m in CANDIDATES
           if wide[m].mean() >= wide[best_ref].mean() + 0.03 and (wide[m] > wide[best_ref]).sum() >= 2 / 3 * len(years)]
decision = {"name": NAME, "best_reference": best_ref, "passing": passing,
            "cv_mean_macro_F1": wide.mean().round(4).to_dict()}
if not passing:
    decision["frozen"] = None
    (MODELS / f"{TAG}_decision.json").write_text(json.dumps(decision, indent=2))
    print(f"\nNo candidate passes the rule against {best_ref}. Test sets NOT re-scored.")
    sys.exit(0)

if TARGET != "spei3":
    decision["frozen"] = None
    (MODELS / f"{TAG}_decision.json").write_text(json.dumps(decision, indent=2))
    print(f"\n{TARGET} is a side check: CV only, test sets NOT scored. Passing: {passing}")
    sys.exit(0)

chosen = max(passing, key=lambda m: wide[m].mean())
kind, feats = CANDIDATES[chosen]
decision.update(frozen=chosen, kind=kind, features=feats, xgb=XGB, cap=CAP, first_year=FIRST_YEAR,
                test_years=TEST_YEARS)
(MODELS / f"{TAG}_decision.json").write_text(json.dumps(decision, indent=2))
print(f"\nfrozen: {chosen}")

model, rows["p"] = fit_predict(kind, feats, fit_all, rows)
if kind.startswith("rf"):
    import joblib
    joblib.dump(model, MODELS / f"{TAG}.joblib")
else:
    model.save_model(MODELS / f"{NAME}_v2.json")
cm = majority(rows, "p")
label = f"{NAME} v2 ({chosen})"
scores = score_table(cm, label)
scores.to_csv(TAB / f"{TAG}_scores.csv", index=False)
stratum_table(rows, "p", label).to_csv(TAB / f"{TAG}_strata.csv", index=False)
cm.to_csv(PRED / f"{TAG}_cell_month.csv", index=False)
with open(TAB / f"{TAG}_confusion.txt", "w") as fh:
    for s in ["temporal test", "strict test"]:
        fh.write(f"{label} - {s}\n{confusion(cm, s).to_string()}\n\n")
imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
imp.to_csv(TAB / f"{TAG}_importance.csv", header=["importance"])
print(scores[["set", "n_cell_months", "macro_F1", "F1_lo", "F1_hi", "QWK", "QWK_lo", "QWK_hi", "accuracy",
              "recall_severe", "recall_moderate", "recall_mild", "recall_none"]].round(3).to_string(index=False))
print("top features:", imp.head(10).round(3).to_dict())
