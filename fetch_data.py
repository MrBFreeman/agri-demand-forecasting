import pandas as pd

def load_timeseries(path, item_name, area_name):
    df = pd.read_csv(path, sep="\t")
    df.columns = df.columns.str.strip()

    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Value"])

    df = df[df["Item"].str.contains(item_name, case=False)]
    df = df[df["Area"].str.contains(area_name, case=False)]

    # production values are largest each year
    df = df.sort_values("Value", ascending=False)
    yearly = df.groupby("Year").first().reset_index()

    return yearly[["Year", "Value"]]

