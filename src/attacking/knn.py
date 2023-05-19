import sklearn.ensemble
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

from . import semantic_security


def init_model() -> semantic_security.Model:
    return sklearn.model_selection.GridSearchCV(
        sklearn.neighbors.KNeighborsClassifier(algorithm="auto", n_jobs=-1),
        {"n_neighbors": [5, 25, 100]},
        n_jobs=-1,
    )
