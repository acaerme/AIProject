import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

# The 9 numeric features we cluster on.
CLUSTER_FEATURES = [
    "Age",
    "Income",
    "Total_Spend",
    "Total_Children",
    "Recency",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "Tenure",
]

# Campaign/complaint columns kept out of clustering and used only to profile clusters.
DESCRIPTOR_COLS = [
    "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5",
    "Response",
    "Complain",
]

# Number of clusters, chosen from the elbow + silhouette plots (see select_k.py).
N_CLUSTERS = 3

# One row has Income = 666666 (data-entry error); the next-highest real income is 162397.
INCOME_OUTLIER_CAP = 200_000


# Collapse junk Marital_Status values into the real categories.
MARITAL_MAP = {
    "Married": "Married",
    "Together": "Together",
    "Single": "Single",
    "Divorced": "Divorced",
    "Widow": "Widow",
    "Alone": "Single",
    "Absurd": "Single",
    "YOLO": "Single",
}


def clean(df: pd.DataFrame, drop_income_outliers: bool = True) -> pd.DataFrame:
    """Drop unusable/implausible rows and tidy Marital_Status."""
    print("=" * 70)
    print("STEP 2 — CLEANING")
    print("=" * 70)
    df = df.copy()
    start = len(df)
    print(f"Starting rows: {start}")

    df = df.dropna(subset=["Income"])
    print(f"  - dropped {start - len(df)} rows with missing Income -> {len(df)}")

    if drop_income_outliers:
        before = len(df)
        df = df[df["Income"] <= INCOME_OUTLIER_CAP]
        print(f"  - dropped {before - len(df)} income outlier(s) > "
              f"{INCOME_OUTLIER_CAP:,} -> {len(df)}")

    before = len(df)
    df = df[df["Year_Birth"] >= 1940]
    print(f"  - dropped {before - len(df)} rows with Year_Birth < 1940 -> {len(df)}")

    print("\n  Marital_Status BEFORE collapse:")
    print(df["Marital_Status"].value_counts().to_string().replace("\n", "\n    "))
    df["Marital_Status"] = df["Marital_Status"].map(MARITAL_MAP)
    print("\n  Marital_Status AFTER collapse (Alone/Absurd/YOLO -> Single):")
    print(df["Marital_Status"].value_counts().to_string().replace("\n", "\n    "))

    print(f"\nRows after cleaning: {len(df)} (removed {start - len(df)} total)")
    return df


MNT_COLS = [
    "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds",
]

# 3-bucket education for readable profiling (descriptor only, not a clustering feature).
EDUCATION_MAP = {
    "Basic": "Undergraduate",
    "2n Cycle": "Undergraduate",
    "Graduation": "Graduate",
    "Master": "Postgraduate",
    "PhD": "Postgraduate",
}


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Derive modelling features and drop ID / zero-variance columns."""
    print("\n" + "=" * 70)
    print("STEP 3 — FEATURE ENGINEERING")
    print("=" * 70)
    df = df.copy()

    # Raw format is day-month-year; pandas 3.0 requires an explicit format.
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y")

    # Anchor time-based features to the latest enrollment date (data is a 2014 snapshot).
    reference_date = df["Dt_Customer"].max()
    reference_year = int(reference_date.year)
    print(f"Reference date (latest enrollment): {reference_date.date()} "
          f"-> reference year {reference_year}")

    df["Age"] = reference_year - df["Year_Birth"]
    df["Total_Spend"] = df[MNT_COLS].sum(axis=1)
    df["Total_Children"] = df["Kidhome"] + df["Teenhome"]
    df["Tenure"] = (reference_date - df["Dt_Customer"]).dt.days

    # Descriptor-only engineered columns (not used for clustering).
    df["Has_Partner"] = df["Marital_Status"].isin(["Married", "Together"]).astype(int)
    df["Education_Level"] = df["Education"].map(EDUCATION_MAP)

    print("Derived: Age, Total_Spend, Total_Children, Tenure "
          "(+ descriptors Has_Partner, Education_Level)")

    df = df.drop(columns=["ID"])

    # Drop zero-variance columns (catches Z_CostContact, Z_Revenue).
    zero_var = [c for c in df.columns if df[c].nunique() == 1]
    if zero_var:
        df = df.drop(columns=zero_var)
        print(f"Dropped zero-variance columns: {zero_var}")

    return df


def build_matrix(df: pd.DataFrame):
    """Select the 9 clustering features and standardize them.

    Returns (X_scaled, fitted_scaler, feature_df_in_original_units). Scaling is
    required because K-Means uses Euclidean distance, which would otherwise be
    dominated by the large-range features (Income, Total_Spend).
    """
    print("\n" + "=" * 70)
    print("STEP 4 & 5 — FEATURE SELECTION + SCALING")
    print("=" * 70)
    print("Clustering on these 9 features:")
    for f in CLUSTER_FEATURES:
        print(f"  - {f}")

    feature_df = df[CLUSTER_FEATURES].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df)

    print(f"\nScaled matrix shape: {X_scaled.shape}")
    print("Per-feature mean after scaling (~0):",
          np.round(X_scaled.mean(axis=0), 3))
    print("Per-feature std  after scaling (~1):",
          np.round(X_scaled.std(axis=0), 3))
    return X_scaled, scaler, feature_df


def run_preprocessing(df: pd.DataFrame):
    """Convenience: clean -> engineer -> build_matrix. Returns (df_feat, X, scaler)."""
    df_clean = clean(df)
    df_feat = engineer(df_clean)
    X_scaled, scaler, _ = build_matrix(df_feat)
    return df_feat, X_scaled, scaler


if __name__ == "__main__":
    from src.load_data import load

    raw = load()
    df_feat, X, scaler = run_preprocessing(raw)
    print("\n[OK] Preprocessing complete. "
          f"Final rows: {len(df_feat)}, clustering matrix: {X.shape}")
