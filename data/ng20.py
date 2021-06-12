from sklearn.datasets import fetch_20newsgroups


def get_training_data(categories, weights=None):
    bunch = fetch_20newsgroups(
        subset="train", categories=categories, shuffle=True, random_state=42
    )

    data = list(zip(bunch.data, bunch.target))

    hist = {
        c: [d for d, t in data if bunch.target_names[t] == c]
        for c in bunch.target_names
    }

    if weights is None:
        weights = [1] * len(categories)

    # Prepare unbalanced data set
    ub_hist = {}

    for c, w in zip(categories, weights):
        ub_hist[c] = hist[c][: int(w * len(hist[c]))]

    print({k: len(v) for k, v in ub_hist.items()})

    # Make training data
    lookup = {c: i for i, c in enumerate(bunch.target_names)}

    return zip(*[(f, lookup[c]) for c, features in ub_hist.items() for f in features])


def get_test_data(categories):
    test = fetch_20newsgroups(
        subset="test", categories=categories, shuffle=True, random_state=42
    )

    return test.data, test.target


def get_data(categories, weights):
    X_train, y_train = get_training_data(categories, weights)
    X_test, y_test = get_test_data(categories)

    return X_train, X_test, y_train, y_test
