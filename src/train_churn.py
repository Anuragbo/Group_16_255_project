import os
import warnings

# sklearn can emit matmul overflow warnings during weighted LR; fits are still valid
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    module=r"sklearn\.utils\.extmath",
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    module=r"sklearn\.linear_model\._linear_loss",
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    fbeta_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier


RANDOM_STATE = 42
FBETA_BETA = 2.0

DATA_PATH = os.path.abspath("data/curated.parquet")
OUT_INTERP = os.path.abspath("outputs/interpretability")
OUT_MODELS = os.path.abspath("outputs/churn_models")

df = pd.read_parquet(DATA_PATH)
X = df.drop(columns=["Churn"])
y = df["Churn"]
# float64 + finite values avoid lbfgs overflow warnings with class_weight
X = X.astype(np.float64)
X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
feature_names = list(X.columns)
print(
    f"Loaded {len(df):,} rows, {X.shape[1]} features. Churn rate: {y.mean():.2%}"
)

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=RANDOM_STATE
)
print(
    "Train:",
    len(X_train),
    "| Val:",
    len(X_val),
    "| Test:",
    len(X_test),
    "| Pos rate train/val/test:",
    f"{y_train.mean():.3f} / {y_val.mean():.3f} / {y_test.mean():.3f}",
)


def make_lr():
    # liblinear is numerically stable for binary + class_weight (default lbfgs can overflow)
    return LogisticRegression(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=2000,
        solver="liblinear",
        dual=False,
    )


def make_rf():
    return RandomForestClassifier(
        class_weight="balanced",
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def make_xgb(y_tr):
    n_neg = int((y_tr == 0).sum())
    n_pos = int((y_tr == 1).sum())
    spw = n_neg / max(1, n_pos)
    return XGBClassifier(
        scale_pos_weight=spw,
        random_state=RANDOM_STATE,
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        n_jobs=-1,
    )


MODEL_BUILDERS = {
    "LogisticRegression": lambda yt: make_lr(),
    "RandomForest": lambda yt: make_rf(),
    "XGBoost": lambda yt: make_xgb(yt),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def cv_metrics(builder):
    probs = np.zeros(len(y_train))
    for tr_idx, va_idx in cv.split(X_train, y_train):
        xtr, xva = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        ytr = y_train.iloc[tr_idx]
        est = builder(ytr)
        est.fit(xtr, ytr)
        probs[va_idx] = est.predict_proba(xva)[:, 1]
    auc = roc_auc_score(y_train, probs)
    pred05 = (probs >= 0.5).astype(int)
    fb = fbeta_score(y_train, pred05, beta=FBETA_BETA, zero_division=0)
    return auc, fb


cv_rows = []
for name, builder in MODEL_BUILDERS.items():
    auc, fb = cv_metrics(builder)
    cv_rows.append(
        {"model": name, "cv_roc_auc": auc, f"cv_f{FBETA_BETA:g}_at_0.5": fb}
    )
cv_summary = pd.DataFrame(cv_rows)
print("\n5-fold CV on train (OOF probs):")
print(cv_summary.to_string(index=False))


def tune_threshold_fbeta(y_true, y_prob, beta=FBETA_BETA, n=201):
    ts = np.linspace(0.0, 1.0, n)
    best_t, best_f = 0.5, -1.0
    for t in ts:
        pred = (y_prob >= t).astype(int)
        f = fbeta_score(y_true, pred, beta=beta, zero_division=0)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t, best_f


thresholds = {}
for name, builder in MODEL_BUILDERS.items():
    est = builder(y_train)
    est.fit(X_train, y_train)
    p_val = est.predict_proba(X_val)[:, 1]
    t_best, f_best = tune_threshold_fbeta(y_val.to_numpy(), p_val)
    thresholds[name] = t_best
    print(
        f"{name}: best threshold={t_best:.4f}, val F{FBETA_BETA:g}={f_best:.4f}"
    )

X_tv = pd.concat([X_train, X_val], axis=0)
y_tv = pd.concat([y_train, y_val], axis=0)

os.makedirs(OUT_MODELS, exist_ok=True)
final_models = {}

print("\nTest set (refit on train+val, tuned threshold):")
for name, builder in MODEL_BUILDERS.items():
    est = builder(y_tv)
    est.fit(X_tv, y_tv)
    final_models[name] = est
    p_test = est.predict_proba(X_test)[:, 1]
    t = thresholds[name]
    y_hat = (p_test >= t).astype(int)
    print("=" * 60)
    print(name, f"| threshold={t:.4f}")
    print(classification_report(y_test, y_hat, digits=4))

    prec, rec, _ = precision_recall_curve(y_test, p_test)
    plt.figure(figsize=(6, 4))
    plt.plot(rec, prec, label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR curve — {name} (test)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    pr_path = os.path.join(OUT_MODELS, f"pr_{name.lower()}_test.png")
    plt.savefig(pr_path)
    plt.close()
    print(f"Saved PR curve → {pr_path}")

os.makedirs(OUT_INTERP, exist_ok=True)
lr_final = final_models["LogisticRegression"]
coef_df = (
    pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": lr_final.coef_.ravel(),
            "odds_ratio": np.exp(lr_final.coef_.ravel()),
        }
    )
    .sort_values("coefficient", key=lambda s: s.abs(), ascending=False)
)
coef_path = os.path.join(OUT_INTERP, "lr_coefficients.csv")
coef_df.to_csv(coef_path, index=False)
print(f"\nSaved LR coefficients → {coef_path}")

xgb_final = final_models["XGBoost"]
rng = np.random.default_rng(RANDOM_STATE)
n_sample = min(500, len(X_test))
idx = rng.choice(len(X_test), size=n_sample, replace=False)
X_shap = X_test.iloc[idx]

explainer = shap.TreeExplainer(xgb_final)
shap_values = explainer.shap_values(X_shap)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_shap, show=False, max_display=20)
plt.tight_layout()
shap_path = os.path.join(OUT_INTERP, "xgb_shap_summary.png")
plt.savefig(shap_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved XGBoost SHAP summary → {shap_path}")
print("done")
