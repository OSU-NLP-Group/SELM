import sklearn.ensemble
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

from . import semantic_security


def init_model() -> semantic_security.Model:
    return sklearn.model_selection.GridSearchCV(
        sklearn.ensemble.RandomForestClassifier(max_features=None),
        {
            "max_features": [1.0, 0.3, "sqrt", "log2"],
            "max_depth": [None],
            "min_samples_split": [2],
        },
        n_jobs=-1,
    )
