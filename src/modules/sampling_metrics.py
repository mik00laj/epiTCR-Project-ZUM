import numpy as np
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors


def GOF(X, k=5, metric='euclidean', threshold=None, quantile=0.95):
    """
    Args:
        X (np.ndarray lub pd.DataFrame): Zbiór próbek (macierz cech).
        k (int): Liczba najbliższych sąsiadów do uwzględnienia.
        metric (str): Metryka odległości (np. 'euclidean', 'cosine').
        threshold (float or None): Próg uznania za outlier. Jeśli None, zostanie wyliczony z quantile.
        quantile (float): Który percentyl przyjąć jako próg, jeśli threshold=None.

    Returns:
        tuple: (gof_scores, outlier_labels)
    """
    if hasattr(X, 'values'):
        X = X.values

    neigh = NearestNeighbors(n_neighbors=k+1, metric=metric)
    neigh.fit(X)
    distances, _ = neigh.kneighbors(X)
    # pomijamy pierwszy sąsiad (to punkt siebie samego)
    gwn_scores= distances[:, 1:].mean(axis=1)

    if threshold is None:
        threshold = np.quantile(gwn_scores, quantile)
    outlier_labels = np.where(gwn_scores > threshold, -1, 1)

    return gwn_scores, outlier_labels


def LOF(X, k=5, metric='euclidean', threshold=None, quantile=0.95):
    """
    Prosta implementacja metody LOF (Local Outlier Factor).

    Args:
        X (np.ndarray or pd.DataFrame): Macierz cech.
        k (int): Liczba najbliższych sąsiadów.
        metric (str): Metryka odległości.
        threshold (float or None): Próg uznania za outlier.
        quantile (float): Używany, jeśli threshold=None.

    Returns:
        tuple: (lof_scores, outlier_labels)
    """
    if hasattr(X, 'values'):
        X = X.values

    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn.fit(X)

    distances, indices = nn.kneighbors(X)
    distances = distances[:, 1:]   # pomijamy pierwszy sąsiad (to punkt siebie samego)
    indices = indices[:, 1:]       # indeksy jego k najbliższych sąsiadów (z wyłączeniem samego siebie).
    n_samples = X.shape[0]
    lof_scores = np.zeros(n_samples)
    reach_dists = np.zeros((n_samples, k))

    for i in range(n_samples):
        for j_idx, j in enumerate(indices[i]):
            dist_ij = np.linalg.norm(X[i] - X[j]) if metric == 'euclidean' else distances[i][j_idx]
            k_dist_j = distances[j].max()
            reach_dists[i, j_idx] = max(dist_ij, k_dist_j)

    lrd = 1 / (reach_dists.mean(axis=1) + 1e-10)

    for i in range(n_samples):
        neighbor_lrds = lrd[indices[i]] / lrd[i]
        lof_scores[i] = neighbor_lrds.mean()

    if threshold is None:
        threshold = np.quantile(lof_scores, quantile)
    outlier_labels = np.where(lof_scores > threshold, -1, 1)

    return lof_scores, outlier_labels

