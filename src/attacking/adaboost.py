import scipy.stats
import sklearn.ensemble
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

from . import semantic_security


def init_model() -> semantic_security.Model:
    return sklearn.model_selection.RandomizedSearchCV(
        sklearn.ensemble.AdaBoostClassifier(n_estimators=100),
        {
            "learning_rate": scipy.stats.loguniform(a=1e-2, b=1e1),
        },
        n_jobs=-1,
        n_iter=100,
    )
