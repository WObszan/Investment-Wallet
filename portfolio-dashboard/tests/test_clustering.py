import pandas as pd

from clustering import cluster_assets


def test_cluster_assets_separates_obviously_different_groups():
    # Two low-risk/return assets and two high-risk/return assets; K-Means with n_clusters=2 should separate them
    risk_profile = pd.DataFrame(
        {
            "returns": [0.05, 0.06, 0.40, 0.42],
            "risk": [0.10, 0.11, 0.60, 0.62],
        },
        index=["Safe_A", "Safe_B", "Risky_A", "Risky_B"],
    )

    result = cluster_assets(risk_profile, n_clusters=2)

    assert "Cluster" in result.columns
    assert result.loc["Safe_A", "Cluster"] == result.loc["Safe_B", "Cluster"]
    assert result.loc["Risky_A", "Cluster"] == result.loc["Risky_B", "Cluster"]
    assert result.loc["Safe_A", "Cluster"] != result.loc["Risky_A", "Cluster"]