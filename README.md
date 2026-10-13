# Customer Segmentation Analysis

A reproducible Python project demonstrating customer segmentation with K-Means clustering.

## What this project does
- Generates a deterministic synthetic customer dataset (no real customer data).
- Checks basic data quality.
- Standardizes features before clustering.
- Compares candidate cluster counts using the elbow method and silhouette score.
- Fits a K-Means model and summarizes the resulting customer groups.
- Saves evaluation tables and plots.

## Features used
- `Age`
- `AnnualIncome_k`
- `SpendingScore`
- `PurchaseFrequency`
- `AverageOrderValue`

All records are synthetic and intended for learning and portfolio demonstration only. The clusters should not be interpreted as findings about real customers.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\\Scripts\\activate`.

## Run

```bash
python src/customer_segmentation.py
```

Outputs are saved under `outputs/`:
- `customer_segments.csv`: synthetic records with assigned cluster IDs
- `cluster_profiles.csv`: cluster-level feature averages
- `cluster_selection_metrics.csv`: inertia and silhouette scores for candidate k values
- `elbow_curve.png`: inertia by candidate cluster count
- `silhouette_scores.png`: silhouette score by candidate cluster count
- `customer_segments_scatter.png`: two-feature visualization of the clusters

## Tests

```bash
python -m unittest discover -s tests -v
```

## Method notes
- Features are standardized with `StandardScaler` before K-Means.
- Candidate values of k are evaluated using inertia and silhouette score.
- The selected k is the candidate with the highest silhouette score among the configured candidates. This is a heuristic, not a guarantee of the true number of customer segments.
- A fixed random seed makes the generated data and model results reproducible.

## Project structure

```text
customer-segmentation-analysis/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── src/
│   ├── __init__.py
│   └── customer_segmentation.py
├── tests/
│   └── test_segmentation.py
├── data/
│   └── .gitkeep
└── outputs/
    └── .gitkeep
```

## License
MIT
