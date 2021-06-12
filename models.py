import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB, MultinomialNB, BernoulliNB
from sklearn.pipeline import Pipeline

from marvin.metrics import (
    mean_entropy,
    cm_purity,
    probabilistic_confusion_matrix as pcm,
)


MODELS = [
    (
        "bnb",
        Pipeline(
            [
                ("tfidf", TfidfVectorizer(binary=True, use_idf=False, norm=False)),
                ("clf", BernoulliNB()),
            ]
        ),
    ),
    ("cnb", Pipeline([("tfidf", TfidfVectorizer()), ("clf", ComplementNB())])),
    ("mnb", Pipeline([("tfidf", TfidfVectorizer()), ("clf", MultinomialNB())])),
]


def fit(X, y):
    for _, model in MODELS:
        model.fit(X, y)


def evaluate(X, y):
    results = []

    for _, model in MODELS:
        Y_pred = model.predict_proba(X)
        cm = pcm(np.array(y), Y_pred, model.classes_)
        results.append((model.score(X, y), 1 - mean_entropy(Y_pred), cm_purity(cm)))
    return results


def fit_evaluate(X_train, X_test, y_train, y_test):
    fit(X_train, y_train)
    return evaluate(X_test, y_test)
