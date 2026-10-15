import os
import re
import time
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIG_DIR = os.path.join(BASE_DIR, "figures")
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

HEADER_RE = re.compile(
    r"^(from|subject|organization|organisation|lines|reply-to|nntp-posting-host|"
    r"distribution|summary|keywords|article-i\.d\.|x-[a-z-]*|path|date|newsgroups|"
    r"sender|followup-to|expires|references|received|message-id|approved|supersedes|"
    r"return-receipt-to|mime-version|content-type|content-transfer-encoding|"
    r"fax|phone|address|posting-host|host):.*$",
    re.IGNORECASE | re.MULTILINE,
)
QUOTE_RE = re.compile(r"^>.*$", re.MULTILINE)
EMAIL_RE = re.compile(r"\S+@[\w.\-]+")
URL_RE = re.compile(r"(http|https|ftp)://\S+|www\.\S+")
NUM_RE = re.compile(r"\b\d+\b")
NONWORD_RE = re.compile(r"[^a-z\s]+")
SPACES_RE = re.compile(r"\s+")


def clean_text(text):
    text = str(text).lower()
    text = HEADER_RE.sub(" ", text)
    text = QUOTE_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = NUM_RE.sub(" ", text)
    text = NONWORD_RE.sub(" ", text)
    text = SPACES_RE.sub(" ", text).strip()
    return text


def safe_clean(text):
    cleaned = clean_text(text)
    if not cleaned:
        cleaned = SPACES_RE.sub(" ", re.sub(r"[^a-z\s]+", " ", str(text).lower())).strip()
    return cleaned


def load_data():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_data.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_data_unlabeled.csv"))
    X_raw = train_df["text"].astype(str).tolist()
    y = train_df["target"].values
    X_test_raw = test_df["text"].astype(str).tolist()
    return X_raw, y, X_test_raw


def build_feature_sets(Xtr_r, Xva_r, Xte_r, Xtr_c, Xva_c, Xte_c):
    specs = {
        "baseline": (
            TfidfVectorizer(max_features=5000),
            Xtr_r, Xva_r, Xte_r,
        ),
        "improved": (
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
            ),
            Xtr_r, Xva_r, Xte_r,
        ),
        "header_removed": (
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
                stop_words="english",
            ),
            Xtr_c, Xva_c, Xte_c,
        ),
    }
    matrices = {}
    for name, (vec, a, b, c) in specs.items():
        matrices[name] = {
            "vectorizer": vec,
            "Xtr": vec.fit_transform(a),
            "Xva": vec.transform(b),
            "Xte": vec.transform(c),
        }
        print(f"feature set [{name}] dim = {matrices[name]['Xtr'].shape[1]}")
    return matrices


def make_models():
    return {
        "Naive Bayes": {
            "fn": lambda p: MultinomialNB(alpha=p["alpha"]),
            "param_name": "alpha",
            "grid": [{"alpha": a} for a in [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]],
        },
        "Linear SVM": {
            "fn": lambda p: LinearSVC(C=p["C"], random_state=SEED, max_iter=5000, dual="auto"),
            "param_name": "C",
            "grid": [{"C": c} for c in [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 3.0, 5.0]],
        },
        "Logistic Regression": {
            "fn": lambda p: LogisticRegression(
                C=p["C"], max_iter=3000, random_state=SEED, solver="lbfgs", n_jobs=-1
            ),
            "param_name": "C",
            "grid": [{"C": c} for c in [0.1, 0.5, 1.0, 3.0, 5.0, 10.0, 100.0]],
        },
        "MLP": {
            "fn": lambda p: MLPClassifier(
                hidden_layer_sizes=p["hidden_layer_sizes"],
                alpha=p["alpha"],
                max_iter=200,
                early_stopping=True,
                n_iter_no_change=10,
                random_state=SEED,
            ),
            "param_name": "structure",
            "grid": [
                {"hidden_layer_sizes": (100,), "alpha": 1e-4},
                {"hidden_layer_sizes": (100,), "alpha": 1e-2},
                {"hidden_layer_sizes": (100, 50), "alpha": 1e-4},
            ],
        },
    }


def evaluate(model, X_tr, y_tr, X_va, y_va):
    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - t0
    pred = model.predict(X_va)
    acc = accuracy_score(y_va, pred)
    f1 = f1_score(y_va, pred, average="macro")
    return acc, f1, train_time, pred


def main():
    print("[1/6] loading data ...")
    X_raw, y, X_test_raw = load_data()
    X_clean = [safe_clean(t) for t in X_raw]
    X_test_clean = [safe_clean(t) for t in X_test_raw]

    tr_idx, va_idx = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=SEED, stratify=y
    )
    y_tr, y_va = y[tr_idx], y[va_idx]
    Xtr_r = [X_raw[i] for i in tr_idx]
    Xva_r = [X_raw[i] for i in va_idx]
    Xtr_c = [X_clean[i] for i in tr_idx]
    Xva_c = [X_clean[i] for i in va_idx]
    print(f"train={len(tr_idx)}  val={len(va_idx)}  test={len(X_test_raw)}")

    print("[2/6] building feature sets ...")
    features = build_feature_sets(
        Xtr_r, Xva_r, X_test_raw, Xtr_c, Xva_c, X_test_clean
    )

    models = make_models()
    rows = []
    best_overall = None
    best_pred_val = None

    print("[3/6] grid search over models x feature sets ...")
    for model_name, spec in models.items():
        for feat_name in ["baseline", "improved", "header_removed"]:
            Xtr = features[feat_name]["Xtr"]
            Xva = features[feat_name]["Xva"]
            for params in spec["grid"]:
                print(f"  {model_name:19s} | {feat_name:14s} | {params}")
                acc, f1, train_time, pred = evaluate(
                    spec["fn"](params), Xtr, y_tr, Xva, y_va
                )
                rows.append(
                    {
                        "model": model_name,
                        "feature_set": feat_name,
                        "params": str(params),
                        "param_value": params.get(spec["param_name"], str(params)),
                        "val_accuracy": round(acc, 5),
                        "val_macro_f1": round(f1, 5),
                        "train_time_s": round(train_time, 2),
                    }
                )
                if best_overall is None or acc > best_overall[0]:
                    best_overall = (
                        acc,
                        f1,
                        model_name,
                        feat_name,
                        params,
                        train_time,
                    )
                    best_pred_val = pred

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(RESULT_DIR, "validation_results.csv"), index=False)

    best_acc, best_f1, best_model_name, best_feat_name, best_params, best_time = best_overall
    print(
        f"\n[4/6] best overall = {best_model_name} on [{best_feat_name}] "
        f"with {best_params}  val_acc={best_acc:.4f}  macro_f1={best_f1:.4f}"
    )

    summary = (
        results.groupby(["model", "feature_set"])
        .agg(
            best_val_accuracy=("val_accuracy", "max"),
            best_macro_f1=("val_macro_f1", "max"),
        )
        .reset_index()
    )
    print("\n===== validation summary =====")
    print(summary.to_string(index=False))

    labels = sorted(np.unique(y))
    cm = confusion_matrix(y_va, best_pred_val, labels=labels)
    p, r, f, support = precision_recall_fscore_support(
        y_va, best_pred_val, labels=labels, zero_division=0
    )

    per_class = pd.DataFrame(
        {
            "class": labels,
            "precision": p.round(4),
            "recall": r.round(4),
            "f1": f.round(4),
            "support": support,
        }
    )
    per_class.to_csv(os.path.join(RESULT_DIR, "per_class_metrics.csv"), index=False)
    print("\n===== per-class metrics (best model) =====")
    print(per_class.to_string(index=False))

    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    model_order = ["Naive Bayes", "Linear SVM", "Logistic Regression", "MLP"]
    short = ["Naive Bayes", "Linear SVM", "Logistic Reg.", "MLP"]
    feat_order = ["baseline", "improved", "header_removed"]
    colors = ["#9bb7d4", "#c0504d", "#4f8150"]
    x = np.arange(len(model_order))
    w = 0.26
    for k, (fn, color) in enumerate(zip(feat_order, colors)):
        vals = [
            summary[(summary["model"] == m) & (summary["feature_set"] == fn)][
                "best_val_accuracy"
            ].iloc[0]
            for m in model_order
        ]
        bars = ax.bar(x + (k - 1) * w, vals, w, label=fn.replace("_", " "), color=color)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(short)
    ax.set_ylim(0.75, 1.0)
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Best validation accuracy by model and feature set")
    ax.legend(fontsize=8, ncol=3, loc="lower center")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "model_comparison.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion matrix ({best_model_name}, {best_feat_name})")
    thresh = cm.max() / 2
    for i in labels:
        for j in labels:
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "confusion_matrix.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(np.array(labels) - 0.22, p, 0.22, label="Precision", color="#4f81bd")
    ax.bar(np.array(labels), r, 0.22, label="Recall", color="#9bbb59")
    ax.bar(np.array(labels) + 0.22, f, 0.22, label="F1", color="#f0a04b")
    ax.set_xticks(labels)
    ax.set_xticklabels([str(i) for i in labels])
    ax.set_ylim(0.72, 1.02)
    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_title(f"Per-class precision / recall / F1 ({best_model_name})")
    ax.legend(fontsize=8, ncol=3, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "per_class_metrics.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    cur = results[results["feature_set"] == "improved"]
    curves = [
        ("Linear SVM", "C", "o-", "#c0504d"),
        ("Logistic Regression", "C", "s-", "#4f81bd"),
        ("Naive Bayes", "alpha", "^-", "#9bbb59"),
    ]
    for mname, pname, style, color in curves:
        sub = cur[cur["model"] == mname].copy()
        sub["pv"] = sub["param_value"].astype(float)
        sub = sub.sort_values("pv")
        ax.plot(
            sub["pv"],
            sub["val_accuracy"],
            style,
            color=color,
            label=mname,
            markersize=4,
        )
    ax.set_xscale("log")
    ax.set_xlabel("regularization parameter (C for SVM/LR, alpha for NB)")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Hyperparameter tuning curves (improved feature set)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "tuning_curves.png"))
    plt.close(fig)

    print("[5/6] error analysis on validation set done.")

    print(f"[6/6] retraining best pipeline on full training data ...")
    final_vec = features[best_feat_name]["vectorizer"]
    if best_feat_name == "header_removed":
        X_all_text, X_test_final = X_clean, X_test_clean
    else:
        X_all_text, X_test_final = X_raw, X_test_raw
    X_all = final_vec.fit_transform(X_all_text)
    X_test_final = final_vec.transform(X_test_final)
    final_model = models[best_model_name]["fn"](best_params)
    final_model.fit(X_all, y)
    predictions = final_model.predict(X_test_final)
    pd.DataFrame(predictions).to_csv(
        os.path.join(BASE_DIR, "prediction.csv"), index=False, header=False
    )

    with open(os.path.join(RESULT_DIR, "best_model.txt"), "w", encoding="utf-8") as f:
        f.write(f"best_model={best_model_name}\n")
        f.write(f"best_feature_set={best_feat_name}\n")
        f.write(f"best_params={best_params}\n")
        f.write(f"val_accuracy={best_acc:.4f}\n")
        f.write(f"val_macro_f1={best_f1:.4f}\n")
        f.write(f"num_test_predictions={len(predictions)}\n")

    print(f"prediction.csv saved ({len(predictions)} rows)")
    print("done.")


if __name__ == "__main__":
    main()
