"""Offline evaluation of the subsystem router, written to reports/.

Compares the model against two baselines on a held-out split (stratified, grouped by run
so failures from the same run never leak across train/test):
  1. majority class  — the floor any model must beat
  2. log-module rule — route by the module name in the first log line (what a script would do)

Usage:  python -m scripts.evaluate --data data/runs.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit

from app.ml.router_model import REVIEW_THRESHOLD, SubsystemRouter


def load_frame(path: str) -> pd.DataFrame:
    runs = json.loads(Path(path).read_text())
    rows = [
        {"run": i, "text": f"{f['description']}\n{f['log_excerpt']}", "label": f["subsystem"], "log": f["log_excerpt"]}
        for i, r in enumerate(runs) for f in r["failures"]
    ]
    return pd.DataFrame(rows)


def rule_baseline(log: str, fallback: str) -> str:
    m = re.search(r"(perception|prediction|planning|controls|vehicle|localization|sensors|infra)/", log)
    if not m:
        return fallback
    return {"vehicle": "controls", "sensors": "localization", "infra": "infrastructure"}.get(m.group(1), m.group(1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/runs.json")
    p.add_argument("--out", default="reports")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)

    df = load_frame(a.data)
    train_idx, test_idx = next(GroupShuffleSplit(test_size=0.25, random_state=42).split(df, groups=df["run"]))
    train, test = df.iloc[train_idx], df.iloc[test_idx]

    model = SubsystemRouter.train(train["text"].tolist(), train["label"].tolist())
    preds = [model.predict(t) for t in test["text"]]
    y_pred = [p.label for p in preds]
    conf = np.array([p.confidence for p in preds])
    y = test["label"].tolist()

    majority = train["label"].mode()[0]
    results = {
        "majority_class": {
            "accuracy": accuracy_score(y, [majority] * len(y)),
            "macro_f1": f1_score(y, [majority] * len(y), average="macro"),
        },
        "log_module_rule": {},
        "tfidf_logreg": {"accuracy": accuracy_score(y, y_pred), "macro_f1": f1_score(y, y_pred, average="macro")},
    }
    rule = [rule_baseline(log, majority) for log in test["log"]]
    results["log_module_rule"] = {"accuracy": accuracy_score(y, rule), "macro_f1": f1_score(y, rule, average="macro")}

    auto = conf >= REVIEW_THRESHOLD
    auto_acc = accuracy_score(np.array(y)[auto], np.array(y_pred)[auto]) if auto.any() else float("nan")
    results["routing_policy"] = {
        "review_threshold": REVIEW_THRESHOLD,
        "auto_routed_share": float(auto.mean()),
        "accuracy_on_auto_routed": float(auto_acc),
        "sent_to_human_review": int((~auto).sum()),
    }
    results["sizes"] = {"train": len(train), "test": len(test)}

    labels = sorted(df["label"].unique())
    cm = confusion_matrix(y, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted owner")
    ax.set_ylabel("Human label")
    ax.set_title("Subsystem router: confusion matrix (held-out runs)")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)

    report = classification_report(y, y_pred, digits=3)
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    md = ["# Router evaluation", "", f"Train failures: {len(train)} · Test failures: {len(test)} (split by run)", "",
          "| Method | Accuracy | Macro F1 |", "|---|---|---|"]
    for name in ("majority_class", "log_module_rule", "tfidf_logreg"):
        md.append(f"| {name} | {results[name]['accuracy']:.3f} | {results[name]['macro_f1']:.3f} |")
    rp = results["routing_policy"]
    md += ["", f"**Routing policy** (confidence ≥ {REVIEW_THRESHOLD}): "
           f"{rp['auto_routed_share']:.0%} of failures auto-routed "
           f"at {rp['accuracy_on_auto_routed']:.1%} accuracy; {rp['sent_to_human_review']} sent to human review.",
           "", "## Per-class report", "", "```", report, "```", "", "![confusion matrix](confusion_matrix.png)"]
    (out / "evaluation.md").write_text("\n".join(md))
    print("\n".join(md[:9]))
    print(md[11])


if __name__ == "__main__":
    main()
