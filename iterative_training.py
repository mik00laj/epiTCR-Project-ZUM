from src.modules.sampling_metrics import LOF, GOF
import numpy as np

def iterative_training_with_lof(X_train, y_train, model, n_iterations=5, k = 5, tree_increment = 0):
    """
    Args:
        X_train: dane treningowe
        y_train: etykiety treningowe
        n_iterations: liczba iteracji
    Returns:
        model: wytrenowany model RandomForest
    """

    print(f"Rozpoczynam iteracyjne uczenie z {n_iterations} iteracjami...")
    print("Obliczanie LOF scores...")
    lof_scores, _ = LOF(X_train, k=k, metric='euclidean', threshold=None, quantile=0.95)
    # Lista indeksów próbek z X_train posortowanych według metryki od najniższego LOF score
    sorted_indices = np.argsort(lof_scores)

    if tree_increment == False:
        model = model.set_params(warm_start=True, n_estimators = 300)
    else:
        n_initial_estimator = int(300/n_iterations)
        n_increment_estimator = n_initial_estimator
        model = model.set_params(warm_start = True, n_estimators = n_initial_estimator)

    # Obliczenie rozmiarów batchy dla każdej iteracji
    total_samples = len(X_train)
    batch_size = total_samples // n_iterations

    for i in range(n_iterations):
        print(f"\nIteracja {i + 1}/{n_iterations}")

        # Wybór próbek dla tej iteracji
        batch_start = i * batch_size
        batch_end = total_samples if i == n_iterations - 1 else (i + 1) * batch_size
        current_indices = sorted_indices[batch_start:batch_end]

        # iloc służy do wybierania wierszy ze zbioru treningowego używając indeksów
        X_current = X_train.iloc[current_indices]
        y_current = y_train.iloc[current_indices]

        model.fit(X_current, np.ravel(y_current))
        if tree_increment:
            model = model.set_params( n_estimators = model.n_estimators + n_increment_estimator)

    print(f"\nIteracyjne uczenie zakończone!")

    return model


def iterative_training_with_gof(X_train, y_train, model, n_iterations=5, k = 5, tree_increment = 0):
    """
    Args:
        X_train: dane treningowe
        y_train: etykiety treningowe
        n_iterations: liczba iteracji
    Returns:
        model: wytrenowany model RandomForest
    """

    print(f"Rozpoczynam iteracyjne uczenie z {n_iterations} iteracjami...")
    print("Obliczanie GOF scores...")
    gof_scores, _ = GOF(X_train, k=k, metric='euclidean', threshold=None, quantile=0.95)

    # Lista indeksów próbek z X_train posortowanych według metryki od najniższego GOF score
    sorted_indices = np.argsort(gof_scores)

    if tree_increment == False:
        model = model.set_params(warm_start=True, n_estimators = 300)
    else:
        n_initial_estimator = int(300/n_iterations)
        n_increment_estimator = n_initial_estimator
        model = model.set_params(warm_start = True, n_estimators = n_initial_estimator)

    # Obliczenie rozmiarów batchy dla każdej iteracji
    total_samples = len(X_train)
    batch_size = total_samples // n_iterations

    for i in range(n_iterations):
        print(f"\nIteracja {i + 1}/{n_iterations}")

        # Wybór próbek dla tej iteracji
        batch_start = i * batch_size
        batch_end = total_samples if i == n_iterations - 1 else (i + 1) * batch_size
        current_indices = sorted_indices[batch_start:batch_end]

        # iloc służy do wybierania wierszy ze zbioru treningowego używając indeksów
        X_current = X_train.iloc[current_indices]
        y_current = y_train.iloc[current_indices]

        model.fit(X_current, np.ravel(y_current))
        if tree_increment:
            model = model.set_params( n_estimators = model.n_estimators + n_increment_estimator)

    print(f"\nIteracyjne uczenie zakończone!")

    return model