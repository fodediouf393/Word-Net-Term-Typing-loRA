from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


@dataclass
class MetricsResult:
    accuracy: float
    macro_f1: float
    confusion: np.ndarray


def compute_metrics_from_logits(logits: np.ndarray, labels: np.ndarray) -> MetricsResult:
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    mf1 = f1_score(labels, preds, average="macro")
    cm = confusion_matrix(labels, preds)
    return MetricsResult(accuracy=float(acc), macro_f1=float(mf1), confusion=cm)


def hf_compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    res = compute_metrics_from_logits(np.asarray(logits), np.asarray(labels))
    return {"accuracy": res.accuracy, "macro_f1": res.macro_f1}
