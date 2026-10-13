"""Reproducible customer segmentation demonstration using K-Means."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
RANDOM_STATE = 42
FEATURES = [
    "Age",
    "AnnualIncome_k",
    "SpendingScore",
    "PurchaseFrequency",
    "AverageOrderValue",
]


def generate_customer_data(n_samples=600, random_state=RANDOM_STATE):
    """Create deterministic synthetic customer-like data for demonstration."""
    rng = np.random.default_rng(random_state)
    # Latent profiles are used only to generate plausible variation. They are
    # not exposed as labels or used by the clustering algorithm.
    profiles = [
        # age, income, spending, frequency, order value
        (27, 32, 78, 18, 42),
        (48, 88, 35, 8, 115),
        (36, 72, 82, 16, 105),
        (52, 38, 28, 6, 48),
        (31, 55, 52, 12, 72),
    ]
    proportions = np.array([0.22, 0.18, 0.20, 0.18, 0.22])
    segment_ids = rng.choice(len(profiles), size=n_samples, p=proportions)
    rows = []
    for segment_id in segment_ids:
        age, income, spending, frequency, order_value = profiles[segment_id]
        rows.append(
            [
                rng.normal(age, 5.0),
                rng.normal(income, 12.0),
                rng.normal(spending, 10.0),
                rng.normal(frequency, 3.0),
                rng.normal(order_value, 14.0),
            ]
        )
    data = pd.DataFrame(rows, columns=FEATURES)
    data["Age"] = data["Age"].round().clip(18, 80).astype(int)
    data["AnnualIncome_k"] = data["AnnualIncome_k"].round(1).clip(10, 200)
    data["SpendingScore"] = data["SpendingScore"].round(1).clip(1, 100)
    data["PurchaseFrequency"] = data["PurchaseFrequency"].round().clip(1, 40).astype(int)
    data["AverageOrderValue"] = data["AverageOrderValue"].round(2).clip(5, 300)
    data.insert(0, "CustomerID", [f"C{i:04d}" for i in range(1, n_samples + 1)])
    return data


def evaluate_cluster_counts(X_scaled, k_values=range(2, 9), random_state=RANDOM_STATE):
    """Calculate inertia and silhouette score for each candidate cluster count."""
    rows = []
    for k in k_values:
        model = KMeans(n_clusters=int(k), random_state=random_state, n_init=10)
        labels = model.fit_predict(X_scaled)
        rows.append(
            {
                "k": int(k),
                "inertia": float(model.inertia_),
                "silhouette_score": float(silhouette_score(X_scaled, labels)),
            }
        )
    return pd.DataFrame(rows)


def fit_segmentation(data, k_values=range(2, 9), random_state=RANDOM_STATE):
    """Return standardized features, selection metrics, and assigned clusters."""
    missing = set(FEATURES) - set(data.columns)
    if missing:
        raise ValueError(f"Missing required feature columns: {sorted(missing)}")
    if data[FEATURES].isna().any().any():
        raise ValueError("Feature data contains missing values.")
    if len(data) < 10:
        raise ValueError("At least 10 customer rows are required.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[FEATURES])
    selection = evaluate_cluster_counts(X_scaled, k_values, random_state)
    best_row = selection.loc[selection["silhouette_score"].idxmax()]
    best_k = int(best_row["k"])
    model = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
    labels = model.fit_predict(X_scaled)

    assigned = data.copy()
    assigned["Cluster"] = labels
    profiles = assigned.groupby("Cluster")[FEATURES].mean().round(2).reset_index()
    return assigned, profiles, selection, best_k


def save_outputs(assigned, profiles, selection, output_dir=OUTPUT_DIR):
    """Save CSV summaries and diagnostic plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assigned.to_csv(output_dir / "customer_segments.csv", index=False)
    profiles.to_csv(output_dir / "cluster_profiles.csv", index=False)
    selection.to_csv(output_dir / "cluster_selection_metrics.csv", index=False)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(selection["k"], selection["inertia"], marker="o")
    axis.set(title="Elbow Curve", xlabel="Number of clusters (k)", ylabel="Inertia")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "elbow_curve.png", dpi=160)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(selection["k"], selection["silhouette_score"], marker="o")
    axis.set(title="Silhouette Score by Cluster Count", xlabel="Number of clusters (k)", ylabel="Silhouette score")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "silhouette_scores.png", dpi=160)

    figure, axis = plt.subplots(figsize=(7, 5))
    scatter = axis.scatter(
        assigned["AnnualIncome_k"],
        assigned["SpendingScore"],
        c=assigned["Cluster"],
        cmap="viridis",
        alpha=0.75,
        edgecolors="none",
    )
    axis.set(
        title="Customer Segments (Synthetic Data)",
        xlabel="Annual income (thousands, synthetic units)",
        ylabel="Spending score (1-100, synthetic)",
    )
    figure.colorbar(scatter, ax=axis, label="Cluster")
    figure.tight_layout()
    figure.savefig(output_dir / "customer_segments_scatter.png", dpi=160)


def main():
    data = generate_customer_data()
    assigned, profiles, selection, best_k = fit_segmentation(data)
    save_outputs(assigned, profiles, selection)
    print("Synthetic customer data quality:")
    print(f"  Rows: {len(data)}")
    print(f"  Missing feature values: {int(data[FEATURES].isna().sum().sum())}")
    print(f"  Duplicate customer IDs: {int(data['CustomerID'].duplicated().sum())}")
    print("\nCluster-selection metrics:")
    print(selection.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSelected k by silhouette score: {best_k}")
    print("\nCluster profiles (feature averages):")
    print(profiles.to_string(index=False))
    print(f"\nOutputs saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
