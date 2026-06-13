# Customer Personality Analysis — K-Means Clustering

University group project, *Basic Lab: Introduction to AI (26S)*.
**Authors:** Maksim Poksevatkin, Islam Elikhanov.

We segment the customers of a retail company into a small number of interpretable
groups using **K-Means clustering**, implemented **twice**:

1. **From scratch in NumPy** (`src/kmeans_scratch.py`) — to demonstrate we
   understand the algorithm (k-means++ seeding, assignment/update loop,
   convergence, empty-cluster handling).
2. **With scikit-learn** (`src/cluster.py`) — as a reference, and to validate the
   scratch version against it.

The number of clusters *k* is chosen with the **Elbow method** and the
**Silhouette score**, and all features are standardized with **StandardScaler**
before clustering (K-Means is distance-based, so scale matters).

---

## Dataset

**Customer Personality Analysis** (2240 customers × 29 columns), from Kaggle:
<https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis>

The dataset is **not committed** to this repo (see `.gitignore`). You must
download it manually:

1. Open the Kaggle link above and download the dataset (a free Kaggle account is
   required).
2. Inside the archive you will find **`marketing_campaign.csv`** (a
   **tab-separated** file despite the `.csv` extension).
3. Place that file at **`data/marketing_campaign.csv`** in this project.

The pipeline reads only from `data/marketing_campaign.csv`; there is no automatic
download.

---

## Setup

Requires **Python 3.11+** (developed on 3.14). From the project root:

```bash
# 1. Create an isolated virtual environment
python3 -m venv .venv

# 2. Activate it (re-run this in every new terminal session)
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows (PowerShell)

# 3. Install the exact pinned dependencies
pip install -r requirements.txt
```

---

## Running

With the virtual environment active and `data/marketing_campaign.csv` in place:

```bash
# Run the whole pipeline end to end (Steps 1-10). Saves all plots to figures/.
python main.py
```

Each step is also runnable on its own for inspection:

```bash
python -m src.load_data    # Step 1: load + shape/dtype/missingness report
python -m src.preprocess   # Steps 2-5: clean, engineer, select, scale
python -m src.eda          # EDA: feature distributions + correlation heatmap -> figures/
python -m src.kmeans_scratch  # Step 7: self-test of the scratch K-Means on toy data
python -m src.select_k     # Step 6: elbow + silhouette sweep -> figures/
python -m src.cluster      # Steps 8 & 10: final fit + profiling + figures/
python -m src.compare      # Step 9: scratch vs sklearn agreement
```

> Run these as modules (`python -m src.xxx`) from the project root, not as
> `python src/xxx.py`, so the `from src...` imports resolve.

---

## Project structure

```
.
├── data/                  # marketing_campaign.csv goes here (gitignored)
├── figures/               # generated plots (committed)
├── src/
│   ├── load_data.py       # Step 1  — load + sanity checks
│   ├── preprocess.py      # Steps 2-5 — clean, engineer, select, scale (+ constants)
│   ├── eda.py             # EDA — feature distributions + correlation heatmap
│   ├── kmeans_scratch.py  # Step 7  — from-scratch NumPy K-Means
│   ├── select_k.py        # Step 6  — elbow + silhouette sweep
│   ├── cluster.py         # Steps 8 & 10 — final model + profiling + naming
│   └── compare.py         # Step 9  — scratch vs sklearn comparison
├── main.py                # runs the full pipeline
├── requirements.txt       # pinned dependencies
└── README.md
```

`src/preprocess.py` is the single source of truth for the shared constants
(`RANDOM_STATE`, the 9 `CLUSTER_FEATURES`, the `DESCRIPTOR_COLS`, and `N_CLUSTERS`);
all other steps import them from there. To try a different *k*, change
`N_CLUSTERS` in that one file and re-run.

---

## Method summary

- **Cleaning:** drop the 24 rows with missing `Income`; drop 1 implausible income
  (666666, a data-entry outlier); drop 3 rows with `Year_Birth < 1940`; collapse
  junk `Marital_Status` values (`Alone`/`Absurd`/`YOLO`) into `Single`.
  → **2212** customers remain.
- **Feature engineering:** `Age`, `Total_Spend` (sum of the 6 product columns),
  `Total_Children`, `Tenure` (days enrolled, relative to the latest enrollment
  date in the data — the data is a 2014 snapshot, not "today"); drop `ID` and the
  zero-variance `Z_CostContact` / `Z_Revenue` columns.
- **Clustering features (9):** `Age, Income, Total_Spend, Total_Children, Recency,
  NumWebPurchases, NumCatalogPurchases, NumStorePurchases, Tenure`, all
  standardized.
- **Descriptors (not clustered on):** the 5 `AcceptedCmp*` flags, `Response`,
  `Complain` — used only to *profile* the clusters afterwards.
- **k = 3**, chosen from the elbow + silhouette plots.

## Outputs (in `figures/`)

| File | What it shows |
|------|---------------|
| `eda_distributions.png` | Histogram of each of the 9 clustering features |
| `eda_correlation.png` | Correlation heatmap of the 9 features |
| `elbow.png` | Inertia vs k (the elbow) |
| `silhouette.png` | Mean silhouette vs k |
| `cluster_heatmap.png` | Standardized centroids — each segment's personality |
| `cluster_feature_bars.png` | Income / Total_Spend / Age per cluster (real units) |
| `cluster_campaign_rates.png` | Campaign-acceptance & complaint rates per cluster |
| `cluster_sizes.png` | Number of customers per cluster |
| `cluster_pca.png` | 2-D PCA projection coloured by cluster |

## Results

Three segments emerge (k = 3):

| Segment | Share | Profile |
|---------|-------|---------|
| **Budget / Low-Engagement** | ~46% | Lowest income & spend, fewest purchases on every channel, rarely accepts campaigns. |
| **Mid-Market Regulars** | ~30% | Mid income, web-leaning buyers, longest tenure, moderate campaign response. |
| **Premium High-Spenders** | ~25% | Highest income & spend, few children, catalog/store buyers, accepts campaigns at ~3× the budget segment's rate. |

The from-scratch and scikit-learn implementations agree on **~99.7%** of
customers (Adjusted Rand Index ≈ **0.99**), with near-identical inertia —
confirming the scratch implementation is correct.
