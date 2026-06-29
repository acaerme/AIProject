from pathlib import Path

import pandas as pd

DATA_PATH = "data/marketing_campaign.csv"
EXPECTED_SHAPE = (2240, 29)

def load(path: str = DATA_PATH) -> pd.DataFrame:
    """Read the tab-separated dataset into a DataFrame."""
    csv = Path(path)
    if not csv.exists():
        raise FileNotFoundError(
            f"Could not find '{path}'.\n"
            "Download 'Customer Personality Analysis' from Kaggle "
            "(https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis) "
            "and place marketing_campaign.csv in the data/ folder. See README.md."
        )

    df = pd.read_csv(path, sep="\t")
    return df

def summarize(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("STEP 1 — DATA LOADING & SANITY CHECK")
    print("=" * 70)

    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")

    print("\nColumn dtypes:")
    print(df.dtypes.to_string())

    missing = df.isna().sum()
    print("\nMissing values per column:")
    print(missing.to_string())
    print(f"\nTotal missing cells: {int(missing.sum())} "
          f"(all in 'Income': {int(missing.get('Income', 0))})")


def check_shape(df: pd.DataFrame, expected: tuple = EXPECTED_SHAPE) -> None:
    assert df.shape == expected, (
        f"Unexpected shape {df.shape}; expected {expected}. "
        "The dataset on disk does not match the documented version."
    )
    print(f"\n[OK] Shape check passed: {df.shape} == {expected}")


if __name__ == "__main__":
    data = load()
    summarize(data)
    check_shape(data)
