"""Subsystem router: predicts which team owns a failure from its text.

Model: TF-IDF (word 1-2 grams + character 3-5 grams, so log tokens like `nav_planner`
still carry signal) feeding a class-balanced logistic regression. Simple, fast, and the
probabilities are calibrated enough to decide when a human must look.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

# Below this confidence the prediction is shown, but the failure is flagged for human review.
REVIEW_THRESHOLD = 0.60


def build_pipeline() -> Pipeline:
    features = FeatureUnion(
        [
            ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
        ]
    )
    clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])


@dataclass
class Prediction:
    label: str
    confidence: float
    ranked: list[tuple[str, float]]

    @property
    def needs_review(self) -> bool:
        return self.confidence < REVIEW_THRESHOLD


class SubsystemRouter:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    @classmethod
    def train(cls, texts: list[str], labels: list[str]) -> SubsystemRouter:
        pipe = build_pipeline()
        pipe.fit(texts, labels)
        return cls(pipe)

    @classmethod
    def load(cls, path: str | Path) -> SubsystemRouter:
        return cls(joblib.load(path))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    def predict(self, text: str, top_k: int = 3) -> Prediction:
        probs = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        order = np.argsort(probs)[::-1][:top_k]
        ranked = [(str(classes[i]), round(float(probs[i]), 4)) for i in order]
        return Prediction(label=ranked[0][0], confidence=ranked[0][1], ranked=ranked)

    def predict_many(self, texts: list[str]) -> list[str]:
        return [str(x) for x in self.pipeline.predict(texts)]
