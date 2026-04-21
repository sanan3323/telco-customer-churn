"""Full ML pipeline builder — preprocessing + SMOTE + classifier."""

from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.features import build_preprocessor


def build_pipeline(
    classifier: BaseEstimator | None = None,
    random_state: int = 42,
) -> ImbPipeline:
    """Assemble the full training pipeline.

    Steps:
      pre:   ColumnTransformer (scale numerics, one-hot categoricals)
      smote: SMOTE oversampling (training folds only — imblearn handles this)
      clf:   The classifier. Defaults to GradientBoosting, which won the
             baseline comparison in the notebook.

    Parameters
    ----------
    classifier
        Any sklearn-compatible estimator. Must expose `predict_proba`.
    random_state
        Seed for SMOTE and the default classifier.

    Returns
    -------
    imblearn.pipeline.Pipeline
        Unfitted pipeline. Call `.fit(X, y)` to train.
    """
    if classifier is None:
        classifier = GradientBoostingClassifier(random_state=random_state)

    return ImbPipeline(steps=[
        ("pre",   build_preprocessor()),
        ("smote", SMOTE(random_state=random_state)),
        ("clf",   classifier),
    ])
