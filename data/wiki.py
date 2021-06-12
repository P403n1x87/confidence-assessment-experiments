from sklearn.model_selection import train_test_split

from pandas import read_csv


def get_data(threshold=100):
    # Collect data
    wiki = read_csv("wiki_movie_plots_deduped.csv")

    data = [(p, g) for p, g in zip(wiki.Plot, wiki.Genre)]
    class_map = {
        c: [d for d, t in data if t == c] for c in wiki.Genre if c != "unknown"
    }
    hist = {c: len(v) for c, v in class_map.items()}

    data = [(p, g) for p, g in data if g != "unknown" and hist[g] > threshold]
    hist = {c: l for c, l in hist.items() if l > threshold}

    print(hist)
    print(len(hist))

    # Make training data

    X, y = zip(*data)

    return train_test_split(X, y, stratify=y)
