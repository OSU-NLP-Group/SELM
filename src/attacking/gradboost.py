"""
Tries scikit learn's default gradient boosting classifier.
"""

import sklearn.decomposition
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

from . import semantic_security


def init_model(seed) -> semantic_security.Model:
    return sklearn.pipeline.make_pipeline(
        sklearn.preprocessing.StandardScaler(),
        sklearn.ensemble.GradientBoostingClassifier(random_state=seed),
    )
