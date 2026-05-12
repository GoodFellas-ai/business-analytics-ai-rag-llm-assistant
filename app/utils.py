import pandas as pd


def load_csv(csv_path):

    df = pd.read_csv(csv_path)

    if "Quantity" in df.columns and "UnitPrice" in df.columns:

        df["Revenue"] = (
            df["Quantity"] * df["UnitPrice"]
        )

    return df