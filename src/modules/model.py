import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import (
    confusion_matrix, accuracy_score, roc_auc_score, roc_curve 
)
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
from iterative_training import *
import src.modules.processor as Processor

def RandomForest_withoutMHC(train_data, test_data, metric="standard", k=1, max_features = 15, bootstrap = False, tree_increment = False, test = "no", name_without_ext = ""):
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
    # saveByPickle(rf_model, f"./models/rdforestWithoutMHCModel_metric={metric}_k={k}_test_file={name_without_ext}_max_features={max_features}_bootstrap={bootstrap}_tree_increment={tree_increment}.pickle")
    test_acc, test_auc, sensitivity, specificity = evaluation(rf_model, X_test, y_test, test_data, test, metric, k, name_without_ext, max_features, bootstrap, tree_increment)
    return test_acc, test_auc, sensitivity, specificity

def RandomForest_withMHC(train_data, test_data, metric="standard", k=1, max_features = 15, bootstrap = False, tree_increment = False, test = "no", name_without_ext = ""):
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
    # m.saveByPickle(rf_model_mhc, f"./models/rdforestWithMHCModel_metric={metric}_k={k}_test_file={name_without_ext}_max_features={max_features}_bootstrap={bootstrap}_tree_increment={tree_increment}.pickle")
    test_acc, test_auc, sensitivity, specificity = evaluation(rf_model_mhc, X_test_mhc, y_test_mhc, test_data, test, metric, k, name_without_ext, max_features, bootstrap, tree_increment)
    return test_acc, test_auc, sensitivity, specificity

def rocAuc(y_true, y_proba, test, metric, k, name_without_ext, max_features, bootstrap, ti):
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

def saveByPickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"{obj} has been saved at {path}.")

def evaluation(rf_model, X_test, y_test, test_data, test, metric, k, name_without_ext, max_features, bootstrap, ti):
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

    rocAuc(df_prob_test_rf['binder'], df_prob_test_rf['predict_proba'], test, metric, k, name_without_ext, max_features, bootstrap, ti)

    return test_acc, test_auc, sensitivity, specificity
