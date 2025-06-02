import sys
import pandas as pd
from argparse import ArgumentParser

import src.modules.model as m
from iterative_training import *
from test_runner import *

# Argument parsing
parser = ArgumentParser(description="Specifying Input Parameters")
parser.add_argument("-tr", "--trainfile", required=True, help="Specify the full path of the training file with TCR sequences")
parser.add_argument("-te", "--testfile", required=True, help="Specify the full path of the test file with TCR sequences")
parser.add_argument("-c", "--chain", default="ce", help="Specify the chain(s) to use (ce, cem). Default: ce")
parser.add_argument("-m", "--iter_metric",default="standard",help="Specify the metric for iterative training (LOF, GOF). Default: standard")
parser.add_argument("-k", "--kneighbors",default="1",help="Specify the number of neighbors for LOF/GOF (default: 5). Only used if metric is LOF or GOF")
parser.add_argument("-t", "--test",default="no",help="Specify what you want to test the model (test_file, max_features, bootstrap, k, trees_increment, no). Default: no")
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

if test != "no":
    run_full_test_suite(m.RandomForest_withMHC,m.RandomForest_withoutMHC, k = k, tree_increment = ti, max_features = f, bootstrap = bootstrap, group_by_column = test)

else:
    if chain == 'ce':
        if bootstrap:
            test_acc, test_auc, sensitivity, specificity = m.RandomForest_withoutMHC(train_data, test_data, metric=metric, k=k, max_features = f, bootstrap = True, test =test, name_without_ext = name_without_ext)
        else:
            test_acc, test_auc, sensitivity, specificity = m.RandomForest_withoutMHC(train_data, test_data, metric=metric, k=k, max_features = f, bootstrap = False, test =test, name_without_ext = name_without_ext)
    else:
        if bootstrap:
            test_acc, test_auc, sensitivity, specificity = m.RandomForest_withMHC(train_data, test_data, metric=metric, k=k, max_features = f, bootstrap = False, test =test, name_without_ext = name_without_ext)
        else:
            test_acc, test_auc, sensitivity, specificity = m.RandomForest_withMHC(train_data, test_data, metric=metric, k=k, max_features = f, bootstrap = True, test =test, name_without_ext = name_without_ext)

    print('Done!')
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
