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
        np.ndarray lub tuple: Wektor wartości GWN lub (GWN, etykiety)
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

def LOF(X, k=5, metric='euclidean', contamination='auto'):
    """
    Args:
        X (np.ndarray or pd.DataFrame): Zbiór próbek (macierz cech).
        k (int): Liczba najbliższych sąsiadów do uwzględnienia.
        metric (str): Metryka odległości (np. 'euclidean', 'cosine').
        contamination (float or 'auto'): Oczekiwana proporcja outlierów.

    Returns:
        tuple: (lof_scores, outlier_labels)
            - lof_scores: Wartości LOF dla każdej próbki (wyższe = bardziej odstające)
            - outlier_labels: Binarne etykiety (1 dla normalnych, -1 dla outlierów)
    """
    if hasattr(X, 'values'):
        X = X.values

    lof = LocalOutlierFactor(
        n_neighbors=k,
        metric=metric,
        contamination=contamination,
        novelty=False
    )

    outlier_labels = lof.fit_predict(X)
    lof_scores = -lof.negative_outlier_factor_

    return lof_scores, outlier_labels
