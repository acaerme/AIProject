from src.load_data import load, summarize, check_shape
from src.preprocess import clean, engineer, build_matrix, N_CLUSTERS
from src.eda import run as run_eda
from src.select_k import run as run_select_k
from src.compare import compare
from src.cluster import fit_final, profile_and_plot


def main():
    df_raw = load()
    summarize(df_raw)
    check_shape(df_raw)

    df_clean = clean(df_raw)
    df_feat = engineer(df_clean)

    X, scaler, _ = build_matrix(df_feat)

    run_eda(df_feat)

    run_select_k(X)
    print(f"\n>>> Using N_CLUSTERS = {N_CLUSTERS} (set in src/preprocess.py)\n")

    compare(X)

    model = fit_final(X)

    profile_and_plot(df_feat, model.labels_, model, X)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE — all figures saved to figures/")
    print("=" * 70)

if __name__ == "__main__":
    main()
