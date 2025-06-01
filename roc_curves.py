import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

def plot_roc_from_csvs(file_paths, labels):
    """
    Plots ROC curves from multiple CSV files.

    Args:
        file_paths (list of str): List of paths to the CSV files.
        labels (list of str): List of labels for each ROC curve (corresponding to the files).
    """
    plt.figure(figsize=(8, 6))
    for i, file_path in enumerate(file_paths):
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"Error: CSV file not found at {file_path}")
            continue

        if 'binder' not in df.columns or 'predict_proba' not in df.columns:
            print(f"Error: CSV file {file_path} must contain 'binder' and 'predict_proba' columns.")
            continue

        y_true = df['binder']
        y_proba = df['predict_proba']

        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)

        plt.plot(fpr, tpr, lw=2, label=f'{labels[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # csv_files = ['output_withMHC_metric=GOF_k=12.csv', 'output_withMHC_metric=LOF_k=2.csv', 'output_withMHC_metric=standard_k=5.csv']  # Replace with your file paths
    csv_files = ['output_withoutMHC_metric=GOF_k=1.csv', 'output_withoutMHC_metric=LOF_k=1.csv', 'output_withoutMHC_metric=standard_k=5.csv']  # Replace with your file paths
    curve_labels = ['GOF', 'LOF', 'baseline']

    plot_roc_from_csvs(csv_files, curve_labels)