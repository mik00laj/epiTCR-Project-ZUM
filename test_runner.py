from prettytable import PrettyTable
import pandas as pd
from natsort import natsorted
import os


def run_tests(model_fn_with_mhc, model_fn_without_mhc, group_by_column="test_file", k_mode=False, k = 1, tree_increment = False, max_features = 15, bootstrap = False):
    chains_config = {
        "cem": {
            "model_fn": model_fn_with_mhc,
            "train_path": "data/splitData/withMHC/train/train.csv",
            "test_path": "data/splitData/withMHC/test/"
        },
        "ce": {
            "model_fn": model_fn_without_mhc,
            "train_path": "data/splitData/withoutMHC/train/train.csv",
            "test_path": "data/splitData/withoutMHC/test/"
        }
    }

    results = []

    for chain, config in chains_config.items():
        print(f"Testing Chain {chain}...")

        train_data = pd.read_csv(config["train_path"])

        for metric in ["LOF", "GOF"] if k_mode else ["LOF", "GOF", "standard"]:
            print(f"Testing Metric {metric}...")

            if k_mode:
                for k in range(1, 100, 1):
                    print(f"Test k = {k}")
                    test_data = pd.read_csv(os.path.join(config["test_path"], "test01.csv"))
                    if bootstrap:
                        if chain == 'ce':
                            test_acc, test_auc, sensitivity, specificity = config["model_fn"](train_data, test_data, metric=metric, k=k, max_features = max_features, bootstrap = True, tree_increment = tree_increment)
                        else:
                            test_acc, test_auc, sensitivity, specificity = config["model_fn"](train_data, test_data, metric=metric, k=k, max_features = max_features, bootstrap = False, tree_increment = tree_increment)
                    else:
                        if chain == 'ce':
                            test_acc, test_auc, sensitivity, specificity = config["model_fn"](train_data, test_data, metric=metric, k=k, max_features = max_features, bootstrap = False, tree_increment = tree_increment)
                        else:
                            test_acc, test_auc, sensitivity, specificity = config["model_fn"](train_data, test_data, metric=metric, k=k, max_features = max_features, bootstrap = True, tree_increment = tree_increment)
                    
                    results.append({
                        'chain': chain,
                        'metric': metric,
                        'k': k,
                        'test_acc': test_acc,
                        'test_auc': test_auc,
                        'sensitivity': sensitivity,
                        'specificity': specificity
                    })
            else:
                test_files = natsorted([f for f in os.listdir(config["test_path"]) if f.endswith(".csv")])
                i = 0
                for test_file in test_files:
                    i += 1
                    if group_by_column != 'test_file':
                        if i >= 4:
                            break
                    print(f"Testing File = {test_file}")
                    test_data = pd.read_csv(os.path.join(config["test_path"], test_file))
                    if chain == 'ce':
                        test_acc, test_auc, sensitivity, specificity = config["model_fn"](train_data, test_data, metric=metric, k=k, max_features = max_features, bootstrap = bootstrap, tree_increment = tree_increment)
                    else:
                        test_acc, test_auc, sensitivity, specificity = config["model_fn"](train_data, test_data, metric=metric, k=k, max_features = max_features+5, bootstrap = bootstrap, tree_increment = tree_increment)
                    results.append({
                        'chain': chain,
                        'metric': metric,
                        'test_file': test_file,
                        'test_acc': test_acc,
                        'test_auc': test_auc,
                        'sensitivity': sensitivity,
                        'specificity': specificity
                    })

    return pd.DataFrame(results)


def display_results(df, group_by_column, k_mode=False, tree_increment = False, max_features = 15, bootstrap = False):
    print(f"\nZakończono testowanie. Zapisano {len(df)} wyników.")
    print("=" * 80)

    detailed_table = PrettyTable()
    fields = ["Chain", "Metric", group_by_column.capitalize(), "Accuracy", "AUC", "Sensitivity", "Specificity"]
    detailed_table.field_names = fields

    for _, row in df.iterrows():
        detailed_table.add_row([
            row["chain"],
            row["metric"],
            row[group_by_column],
            f"{row['test_acc']:.4f}",
            f"{row['test_auc']:.4f}",
            f"{row['sensitivity']:.4f}",
            f"{row['specificity']:.4f}"
        ])

    print(f"\nSZCZEGÓŁOWE WYNIKI:")
    print(detailed_table)

    summary_table = PrettyTable()
    summary_table.field_names = ["Chain", "Metric", "Śr. Accuracy", "Odch. Acc", "Śr. AUC", "Odch. AUC",
                                 "Śr. Sensitivity", "Odch. Sens", "Śr. Specificity", "Odch. Spec", "Liczba testów"]

    grouped = df.groupby(["chain", "metric"]).agg({
        "test_acc": ['mean', 'std', 'count'],
        "test_auc": ['mean', 'std'],
        "sensitivity": ['mean', 'std'],
        "specificity": ['mean', 'std']
    }).round(4)

    for (chain, metric), row in grouped.iterrows():
        summary_table.add_row([
            chain.upper(),
            metric,
            f"{row[('test_acc', 'mean')]:.4f}",
            f"{row[('test_acc', 'std')]:.4f}",
            f"{row[('test_auc', 'mean')]:.4f}",
            f"{row[('test_auc', 'std')]:.4f}",
            f"{row[('sensitivity', 'mean')]:.4f}",
            f"{row[('sensitivity', 'std')]:.4f}",
            f"{row[('specificity', 'mean')]:.4f}",
            f"{row[('specificity', 'std')]:.4f}",
            int(row[('test_acc', 'count')])
        ])

    print("\nTABELA UŚREDNIONYCH WYNIKÓW:")
    print(summary_table)

    extra_data = []

    if k_mode and group_by_column == "k":
        print("\nMAKSYMALNE I MINIMALNE WYNIKI DLA KAŻDEGO ŁAŃCUCHA I METRYKI:")
        extra_table = PrettyTable()
        extra_table.field_names = [
            "Chain", "Metric",
            "Max Accuracy", "k (max acc)",
            "Min Accuracy", "k (min acc)",
            "Max AUC", "k (max auc)",
            "Min AUC", "k (min auc)",
            "Max Sensitivity", "k (max sens)",
            "Min Sensitivity", "k (min sens)",
            "Max Specificity", "k (max spec)",
            "Min Specificity", "k (min spec)"
        ]

        for (chain, metric), group in df.groupby(["chain", "metric"]):
            max_acc_row = group.loc[group['test_acc'].idxmax()]
            min_acc_row = group.loc[group['test_acc'].idxmin()]
            max_auc_row = group.loc[group['test_auc'].idxmax()]
            min_auc_row = group.loc[group['test_auc'].idxmin()]
            max_sens_row = group.loc[group['sensitivity'].idxmax()]
            min_sens_row = group.loc[group['sensitivity'].idxmin()]
            max_spec_row = group.loc[group['specificity'].idxmax()]
            min_spec_row = group.loc[group['specificity'].idxmin()]

            extra_table.add_row([
                chain.upper(),
                metric,
                f"{max_acc_row['test_acc']:.4f}", int(max_acc_row['k']),
                f"{min_acc_row['test_acc']:.4f}", int(min_acc_row['k']),
                f"{max_auc_row['test_auc']:.4f}", int(max_auc_row['k']),
                f"{min_auc_row['test_auc']:.4f}", int(min_auc_row['k']),
                f"{max_sens_row['sensitivity']:.4f}", int(max_sens_row['k']),
                f"{min_sens_row['sensitivity']:.4f}", int(min_sens_row['k']),
                f"{max_spec_row['specificity']:.4f}", int(max_spec_row['k']),
                f"{min_spec_row['specificity']:.4f}", int(min_spec_row['k'])
            ])

            extra_data.append({
                "chain": chain.upper(),
                "metric": metric,
                "max_acc": max_acc_row['test_acc'],
                "k_max_acc": int(max_acc_row['k']),
                "min_acc": min_acc_row['test_acc'],
                "k_min_acc": int(min_acc_row['k']),
                "max_auc": max_auc_row['test_auc'],
                "k_max_auc": int(max_auc_row['k']),
                "min_auc": min_auc_row['test_auc'],
                "k_min_auc": int(min_auc_row['k']),
                "max_sens": max_sens_row['sensitivity'],
                "k_max_sens": int(max_sens_row['k']),
                "min_sens": min_sens_row['sensitivity'],
                "k_min_sens": int(min_sens_row['k']),
                "max_spec": max_spec_row['specificity'],
                "k_max_spec": int(max_spec_row['k']),
                "min_spec": min_spec_row['specificity'],
                "k_min_spec": int(min_spec_row['k'])
            })

        print(extra_table)

    elif not k_mode and group_by_column == "test_file":
        print("\nMAKSYMALNE I MINIMALNE WYNIKI DLA KAŻDEGO ŁAŃCUCHA I METRYKI:")
        extra_table = PrettyTable()
        extra_table.field_names = [
            "Chain", "Metric",
            "Max Accuracy", "File (max acc)",
            "Min Accuracy", "File (min acc)",
            "Max AUC", "File (max auc)",
            "Min AUC", "File (min auc)",
            "Max Sensitivity", "File (max sens)",
            "Min Sensitivity", "File (min sens)",
            "Max Specificity", "File (max spec)",
            "Min Specificity", "File (min spec)"
        ]

        for (chain, metric), group in df.groupby(["chain", "metric"]):
            max_acc_row = group.loc[group['test_acc'].idxmax()]
            min_acc_row = group.loc[group['test_acc'].idxmin()]
            max_auc_row = group.loc[group['test_auc'].idxmax()]
            min_auc_row = group.loc[group['test_auc'].idxmin()]
            max_sens_row = group.loc[group['sensitivity'].idxmax()]
            min_sens_row = group.loc[group['sensitivity'].idxmin()]
            max_spec_row = group.loc[group['specificity'].idxmax()]
            min_spec_row = group.loc[group['specificity'].idxmin()]

            extra_table.add_row([
                chain.upper(),
                metric,
                f"{max_acc_row['test_acc']:.4f}", max_acc_row['test_file'],
                f"{min_acc_row['test_acc']:.4f}", min_acc_row['test_file'],
                f"{max_auc_row['test_auc']:.4f}", max_auc_row['test_file'],
                f"{min_auc_row['test_auc']:.4f}", min_auc_row['test_file'],
                f"{max_sens_row['sensitivity']:.4f}", max_sens_row['test_file'],
                f"{min_sens_row['sensitivity']:.4f}", min_sens_row['test_file'],
                f"{max_spec_row['specificity']:.4f}", max_spec_row['test_file'],
                f"{min_spec_row['specificity']:.4f}", min_spec_row['test_file']
            ])

            extra_data.append({
                "chain": chain.upper(),
                "metric": metric,
                "max_acc": max_acc_row['test_acc'],
                "file_max_acc": max_acc_row['test_file'],
                "min_acc": min_acc_row['test_acc'],
                "file_min_acc": min_acc_row['test_file'],
                "max_auc": max_auc_row['test_auc'],
                "file_max_auc": max_auc_row['test_file'],
                "min_auc": min_auc_row['test_auc'],
                "file_min_auc": min_auc_row['test_file'],
                "max_sens": max_sens_row['sensitivity'],
                "file_max_sens": max_sens_row['test_file'],
                "min_sens": min_sens_row['sensitivity'],
                "file_min_sens": min_sens_row['test_file'],
                "max_spec": max_spec_row['specificity'],
                "file_max_spec": max_spec_row['test_file'],
                "min_spec": min_spec_row['specificity'],
                "file_min_spec": min_spec_row['test_file']
            })

        print(extra_table)

    extra_df = pd.DataFrame(extra_data) if extra_data else None
    return grouped, extra_df


def save_results(df, summary_df, prefix, extra_df=None):
    df.to_csv(f"{prefix}_detailed_results.csv", index=False)
    summary_df.to_csv(f"{prefix}_summary.csv")

    if extra_df is not None:
        extra_df.to_csv(f"{prefix}_extremes.csv", index=False)

    print(f"\nZapisano do: {prefix}_detailed_results.csv, {prefix}_summary.csv", end="")
    if extra_df is not None:
        print(f", {prefix}_extremes.csv")
    else:
        print()
    print("=" * 80)


def run_full_test_suite(RandomForest_withMHC, RandomForest_withoutMHC, k, tree_increment, max_features, bootstrap):
    # TESTY NA WIELU PLIKACH
    # df_tests = run_tests(RandomForest_withMHC, RandomForest_withoutMHC, group_by_column="test_file", k_mode=False, tree_increment = tree_increment, max_features = max_features, bootstrap = bootstrap)
    # summary_tests, extra_tests = display_results(df_tests, group_by_column="test_file", k_mode = False, tree_increment = tree_increment, max_features = max_features, bootstrap = bootstrap)
    # save_results(df_tests, summary_tests, prefix="test_files", extra_df=extra_tests)

    # # TESTY NA WARTOŚCIACH K
    # df_k = run_tests(RandomForest_withMHC, RandomForest_withoutMHC, group_by_column="k", k_mode=True, tree_increment = tree_increment, max_features = max_features, bootstrap = bootstrap)
    # summary_k, extra_k = display_results(df_k, group_by_column="k", k_mode=True, tree_increment = tree_increment, max_features = max_features, bootstrap = bootstrap)
    # save_results(df_k, summary_k, prefix="k_values", extra_df=extra_k)

    # TESTY DLA ITERACYJNEGO ZWIĘKSZANIA SIĘ LICZBY DRZEW
    df_n_trees = run_tests(RandomForest_withMHC, RandomForest_withoutMHC, group_by_column="trees_increment", k_mode=False, k = k, tree_increment = tree_increment, max_features = max_features, bootstrap = bootstrap)
    summary_n_trees, extra_n_trees = display_results(df_n_trees, group_by_column="test_file", k_mode=False, tree_increment = tree_increment, max_features = max_features, bootstrap = bootstrap)
    save_results(df_n_trees, summary_n_trees, prefix="trees_increment", extra_df=extra_n_trees)

    # TESTY DLA ZAMIANY WARTOŚCI BOOTSTRAP WZGLĘDEM DOMYŚLNYCH WARTOŚCI Z REPOZYTORIUM
    df_n_trees = run_tests(RandomForest_withMHC, RandomForest_withoutMHC, group_by_column="bootstrap", k_mode=False, k = k, tree_increment = tree_increment, max_features = max_features, bootstrap = bootstrap)
    summary_n_trees, extra_n_trees = display_results(df_n_trees, group_by_column="test_file", k_mode=False, tree_increment = tree_increment, max_features = max_features, bootstrap = bootstrap)
    save_results(df_n_trees, summary_n_trees, prefix="bootstrap", extra_df=extra_n_trees)

