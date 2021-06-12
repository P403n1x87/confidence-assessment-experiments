from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline

from marvin.metrics import entropy_score
from pandas import read_csv

# Collect data
wiki = read_csv("wiki_movie_plots_deduped.csv")
genres = ["sci-fi", "horror", "fantasy", "thriller", "crime", "comedy"]

# data = [(p, g) for p, g in zip(wiki.Plot, wiki.Genre) if g in genres]
data = [(p, g) for p, g in zip(wiki.Plot, wiki.Genre)]
class_map = {c: [d for d, t in data if t == c] for c in wiki.Genre if c != "unknown"}
hist = {c: len(v) for c, v in class_map.items()}

THRESHOLD = 500

data = [(p, g) for p, g in data if g != "unknown" and hist[g] > THRESHOLD]
hist = {c: l for c, l in hist.items() if l > THRESHOLD}

print(hist)
print(len(hist))

# Make training data

X, y = zip(*data)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)


# Fit models
bnb = Pipeline(
    [
        ("tfidf", TfidfVectorizer(binary=True, use_idf=False, norm=False)),
        ("clf", BernoulliNB()),
    ]
)
cnb = Pipeline([("tfidf", TfidfVectorizer()), ("clf", ComplementNB())])
mnb = Pipeline([("tfidf", TfidfVectorizer()), ("clf", MultinomialNB())])

bnb.fit(X_train, y_train)
cnb.fit(X_train, y_train)
mnb.fit(X_train, y_train)


# Evaluate models

bnb_pred = bnb.predict_proba(X_test)
mnb_pred = mnb.predict_proba(X_test)
cnb_pred = cnb.predict_proba(X_test)

print(
    f"BernoulliNB   : {bnb.score(X_test, y_test)*100:.1f}%    {entropy_score(y_test, bnb_pred, bnb.classes_, 1)*100:.1f}%"
)
print(
    f"MultinomialNB : {mnb.score(X_test, y_test)*100:.1f}%    {entropy_score(y_test, mnb_pred, mnb.classes_, 1)*100:.1f}%"
)
print(
    f"ComplementNB  : {cnb.score(X_test, y_test)*100:.1f}%    {entropy_score(y_test, cnb_pred, cnb.classes_, 1)*100:.1f}%"
)
