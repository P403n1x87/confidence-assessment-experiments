import sys

from numpy import linspace

import data.ng20 as NG
import data.wiki as W

from models import fit_evaluate

from reports import write_csv


__module__ = sys.modules[__name__]


def experiment_ng20_balanced():
    categories = ["soc.religion.christian", "comp.graphics", "sci.med"]

    return [
        fit_evaluate(*NG.get_data(categories, [a] * len(categories)))
        for a in linspace(0.1, 1, 10)
    ]


def experiment_ng20_unbalanced():
    categories = ["soc.religion.christian", "comp.graphics", "sci.med"]

    return [
        fit_evaluate(*NG.get_data(categories, [a * 0.2, a * 0.5, a * 1]))
        for a in linspace(0.1, 1, 10)
    ]


def experiment_wiki():
    return [fit_evaluate(*W.get_data(t)) for t in [100, 200, 500, 1000]]


def main():
    for experiment in [_ for _ in dir(__module__) if _.startswith("experiment_")]:
        print(f"Running {experiment}")
        result = getattr(__module__, experiment)()
        write_csv(experiment, result)


if __name__ == "__main__":
    main()
