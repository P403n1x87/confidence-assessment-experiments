def write_csv(file, data):
    if not file.endswith(".csv"):
        file += ".csv"
    with open(file, "w") as fout:
        fout.writelines(
            [",".join([str(_) for m in row for _ in m]) + "\n" for row in data]
        )
