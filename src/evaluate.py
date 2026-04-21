"""Evaluation metrics and diagnostic outputs for the churn model."""

from dataclasses import dataclass, asdict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


@dataclass
class EvaluationReport:
    """Collected test-set metrics. Serialisable with asdict()."""
    roc_auc: float
    pr_auc: float
    threshold: float
    precision: float    # positive class (churn)
    recall: float
    f1: float
    tn: int
    fp: int
    fn: int
    tp: int

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> EvaluationReport:
    """Compute the canonical metric set at a given threshold.

    Threshold-independent: ROC-AUC, PR-AUC.
    Threshold-dependent:   precision/recall/F1 (positive class), CM counts.
    """
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Guard against degenerate predictions at extreme thresholds.
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return EvaluationReport(
        roc_auc   = float(roc_auc_score(y_true, y_proba)),
        pr_auc    = float(average_precision_score(y_true, y_proba)),
        threshold = float(threshold),
        precision = float(precision),
        recall    = float(recall),
        f1        = float(f1),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


def format_report(report: EvaluationReport) -> str:
    """Human-readable summary for logs and stdout."""
    r = report
    return (
        f"ROC-AUC: {r.roc_auc:.4f}\n"
        f"PR-AUC:  {r.pr_auc:.4f}\n"
        f"Threshold: {r.threshold:.3f}\n"
        f"Confusion matrix: TN={r.tn}  FP={r.fp}  FN={r.fn}  TP={r.tp}\n"
        f"Churn class: precision={r.precision:.3f}  "
        f"recall={r.recall:.3f}  f1={r.f1:.3f}"
    )
