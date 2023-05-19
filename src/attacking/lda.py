"""
Applies linear discriminant analysis to hand crafted features on ciphertexts.
"""

import sklearn.discriminant_analysis
import sklearn.pipeline
import sklearn.preprocessing

from . import semantic_security


def init_model() -> semantic_security.Model:
    return sklearn.pipeline.make_pipeline(
        sklearn.preprocessing.StandardScaler(),
        sklearn.discriminant_analysis.LinearDiscriminantAnalysis(),
    )
