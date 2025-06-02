import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fontTools.misc.cython import returns
from sklearn.ensemble import RandomForestClassifier
# from imblearn.under_sampling import RandomUnderSampler
from argparse import ArgumentParser
import src.modules.processor as Processor

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
import src.modules.model as Model
from iterative_training import *
from test_runner import *

# Argument parsing
parser = ArgumentParser(description="Specifying Input Parameters")
parser.add_argument("-tr", "--trainfile", required=True, help="Specify the full path of the training file with TCR sequences")
parser.add_argument("-te", "--testfile", required=True, help="Specify the full path of the test file with TCR sequences")
parser.add_argument("-c", "--chain", default="ce", help="Specify the chain(s) to use (ce, cem). Default: ce")
parser.add_argument("-m", "--iter_metric",default="standard",help="Specify the metric for iterative training (LOF, GOF). Default: standard")
parser.add_argument("-k", "--kneighbors",default="1",help="Specify the number of neighbors for LOF/GOF (default: 5). Only used if metric is LOF or GOF")
parser.add_argument("-t", "--test",default="no",help="Specify if you want to test the model (yes, no). Default: no")
parser.add_argument("-ti", "--trees_increment", action='store_true', default=False, help="Specify if new trees should appear in next model fit function call. Default: False")
parser.add_argument("-f", "--number_of_features",default=15,help="Specify how many features a tree should have up to 600 for 'ce' chain and 1280 for 'cem' chain. Default: 15")
parser.add_argument("-b", "--bootstrap", action='store_true', default=False, help="Specify if you want to reverse bootstrap value from the default setup. Default: False")
args = parser.parse_args()

chain = args.chain
metric = args.iter_metric
k = int(args.kneighbors)
test = args.test
f = int(args.number_of_features)
bootstrap = args.bootstrap
ti = args.trees_increment

if chain not in ["ce", "cem"]:
    sys.exit("Invalid chain. You can select ce (cdr3b+epitope) or cem (cdr3b+epitope+mhc)")

if (chain == "ce" and f >=600) or (chain == "cem" and f >= 1280):
    sys.exit("Invalid number_of_features. You can select lower than 600 for ce (cdr3b+epitope) or lower than 1280 for cem (cdr3b+epitope+mhc)")

print('Loading and encoding the dataset...')
train_data = pd.read_csv(args.trainfile)
test_data = pd.read_csv(args.testfile)
filename = args.testfile.split('/')[-1]
name_without_ext = filename.replace('.csv', '')


def RandomForest_withoutMHC(train_data, test_data, metric="standard", k=k, max_features = f, bootstrap = bootstrap, tree_increment = ti):
    X_train, y_train = Processor.dataRepresentationDownsamplingWithoutMHCb(train_data)
    X_test = Processor.dataRepresentationBlosum62WithoutMHCb(test_data)
    y_test = test_data[["binder"]]

    print('Training Random Forest without MHC...')
    model = RandomForestClassifier(bootstrap=bootstrap, max_features=max_features, n_estimators=300, n_jobs=-1, random_state=42)
    if metric == "LOF":
        rf_model = iterative_training_with_lof(X_train, y_train, model, n_iterations=5, k=k, tree_increment = tree_increment)
    elif metric == "GOF":
        rf_model = iterative_training_with_gof(X_train, y_train, model, n_iterations=5, k=k, tree_increment = tree_increment)
    else:
        rf_model = model.fit(X_train, np.ravel(y_train))
    # Model.saveByPickle(rf_model, f"./models/rdforestWithoutMHCModel_metric={metric}_k={k}_test_file={name_without_ext}_max_features={max_features}_bootstrap={bootstrap}_tree_increment={ti}.pickle")

    print('Evaluating Random Forest without MHC...')
    y_rf_test_proba = rf_model.predict_proba(X_test)[:, 1]
    y_rf_test_pred = rf_model.predict(X_test)

    df_test_rf = pd.DataFrame({'predict_proba': y_rf_test_proba})
    df_prob_test_rf = pd.concat([test_data.reset_index(drop=True), df_test_rf], axis=1)
    df_prob_test_rf['binder_pred'] = (df_prob_test_rf['predict_proba'] >= 0.5).astype(int)
    df_prob_test_rf.to_csv(f"output_withoutMHC_metric={metric}_k={k}_test_file={name_without_ext}_max_features={max_features}_bootstrap={bootstrap}_tree_increment={ti}.csv", index=False)

    # Obliczenie metryk
    test_acc = accuracy_score(y_test, y_rf_test_pred)
    test_auc = roc_auc_score(y_test, y_rf_test_proba)
    cm = confusion_matrix(y_test, y_rf_test_pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    y_true = df_prob_test_rf['binder']
    y_proba = df_prob_test_rf['predict_proba']
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)

    if test == "no":
        output_filename = f'roc_curve_withoutMHC_metric={metric}_k={k}_test_file={name_without_ext}_max_features={max_features}_bootstrap={bootstrap}_tree_increment={ti}.png'
        plt.savefig(output_filename)
        print(f"ROC curve saved to: {output_filename}")


    return test_acc, test_auc, sensitivity, specificity

def RandomForest_withMHC(train_data, test_data, metric="standard", k=k, max_features = f, bootstrap = bootstrap, tree_increment = ti):
    X_train_mhc, y_train_mhc = Processor.dataRepresentationDownsamplingWithMHCb(train_data)
    X_test_mhc = Processor.dataRepresentationBlosum62WithMHCb(test_data)
    y_test_mhc = test_data[["binder"]]

    print('Training Random Forest with MHC...')
    model = RandomForestClassifier(bootstrap=bootstrap, max_features=max_features, n_jobs=-1, random_state=42)
    if metric == "LOF":
        rf_model_mhc = iterative_training_with_lof(X_train_mhc, y_train_mhc, model, n_iterations=5, k=k, tree_increment = tree_increment)
    elif metric == "GOF":
        rf_model_mhc = iterative_training_with_gof(X_train_mhc, y_train_mhc, model, n_iterations=5, k=k, tree_increment = tree_increment)
    else:
        model = model.set_params(n_estimators=300)
        rf_model_mhc = model.fit(X_train_mhc, np.ravel(y_train_mhc))
    # Model.saveByPickle(rf_model_mhc, f"./models/rdforestWithMHCModel_metric={metric}_k={k}_test_file={name_without_ext}_max_features={max_features}_bootstrap={bootstrap}_tree_increment={ti}.pickle")

    print('Evaluating Random Forest with MHC...')
    y_rf_test_proba_mhc = rf_model_mhc.predict_proba(X_test_mhc)[:, 1]
    y_rf_test_pred_mhc = rf_model_mhc.predict(X_test_mhc)

    df_test_rf_mhc = pd.DataFrame({'predict_proba': y_rf_test_proba_mhc})
    df_prob_test_rf_mhc = pd.concat([test_data.reset_index(drop=True), df_test_rf_mhc], axis=1)
    df_prob_test_rf_mhc['binder_pred'] = (df_prob_test_rf_mhc['predict_proba'] >= 0.5).astype(int)
    df_prob_test_rf_mhc.to_csv(f"output_withMHC_metric={metric}_k={k}_test_file={name_without_ext}_max_features={max_features}_bootstrap={bootstrap}_tree_increment={ti}.csv", index=False)

    # Obliczenie metryk
    test_acc = accuracy_score(y_test_mhc, y_rf_test_pred_mhc)
    test_auc = roc_auc_score(y_test_mhc, y_rf_test_proba_mhc)
    cm = confusion_matrix(y_test_mhc, y_rf_test_pred_mhc)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    y_true = df_prob_test_rf_mhc['binder']
    y_proba = df_prob_test_rf_mhc['predict_proba']
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)

    if test == "no":
        output_filename = f'roc_curve_withMHC_metric={metric}_k={k}_test_file={name_without_ext}_max_features={max_features}_bootstrap={bootstrap}_tree_increment={ti}.png'
        plt.savefig(output_filename)

    return test_acc, test_auc, sensitivity, specificity

if test == "yes":
    run_full_test_suite(RandomForest_withMHC,RandomForest_withoutMHC, k = k, tree_increment = ti, max_features = f, bootstrap = bootstrap)

else:
    if chain == 'ce':
        if bootstrap:
            test_acc, test_auc, sensitivity, specificity = RandomForest_withoutMHC(train_data, test_data, metric=metric, k=k, max_features = f, bootstrap = True)
        else:
            test_acc, test_auc, sensitivity, specificity = RandomForest_withoutMHC(train_data, test_data, metric=metric, k=k, max_features = f, bootstrap = False)
    else:
        if bootstrap:
            test_acc, test_auc, sensitivity, specificity = RandomForest_withMHC(train_data, test_data, metric=metric, k=k, max_features = f, bootstrap = False)
        else:
            test_acc, test_auc, sensitivity, specificity = RandomForest_withMHC(train_data, test_data, metric=metric, k=k, max_features = f, bootstrap = True)

    print('Done!')
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")


