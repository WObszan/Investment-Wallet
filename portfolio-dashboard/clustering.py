import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def cluster_assets(risk_profile: pd.DataFrame, n_clusters: int = 2) -> pd.DataFrame:
    """K-Means clustering of assets by their risk profile, same as the notebook."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(risk_profile)

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    result = risk_profile.copy()
    result["Cluster"] = kmeans.fit_predict(X_scaled)

    return result