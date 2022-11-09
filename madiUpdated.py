import sys
import time
from os import mkdir
from src.madi.detectors.integrated_gradients_interpreter import IntegratedGradientsInterpreter
from src.madi.utils import file_utils
from src.madi.datasets.smart_buildings_dataset import SmartBuildingsDataset
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from IPython.display import clear_output
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.dates as mdates
import matplotlib as mpl
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from src.madi.datasets import gaussian_mixture_dataset
from src.madi.detectors.neg_sample_neural_net_detector import NegativeSamplingNeuralNetworkAD
from src.madi.detectors.isolation_forest_detector import IsolationForestAd
from src.madi.detectors.integrated_gradients_interpreter import IntegratedGradientsInterpreter
from src.madi.detectors.one_class_svm import OneClassSVMAd
from src.madi.detectors.neg_sample_random_forest import NegativeSamplingRandomForestAd
from src.madi.utils import evaluation_utils

from pyod.models.ecod import ECOD
from pyod.models.cblof import CBLOF
from pyod.models.iforest import IForest
from pyod.models.pca import PCA

import tensorflow as tf

assert tf.version.VERSION > '2.1.0'

# @title Choose the data set
_RESOURCE_LOCATION = "src.madi.datasets.data"
data_source = "smart_buildings"  # @param ["gaussian_mixture", "smart_buildings"]
ds = None


class InvalidDatasetError(ValueError):
    pass


if data_source == 'smart_buildings':

    data_file = file_utils.PackageResource(
        _RESOURCE_LOCATION, "ahuVavBothOccModeData.csv")
        # _RESOURCE_LOCATION, "ahuVavModelDataLABELED.csv")
    readme_file = file_utils.PackageResource(
        _RESOURCE_LOCATION, "anomaly_detection_sample_1577622599_README.md")
    ds = SmartBuildingsDataset(data_file, readme_file)
    print(ds.description)

else:
    raise InvalidDatasetError("You requested an invalid data set (%s)." % data_source)

print('Randomize the data, and split into training and test sample.')
split_ix = int(len(ds.sample) * 0.8)
training_sample = ds.sample.iloc[:split_ix]
test_sample = ds.sample.iloc[split_ix:]
print("\tTraining sample size: %d" % len(training_sample))
print("\tTest sample size: %d" % len(test_sample))

# training_sample = training_sample.dropna().astype(int)
# test_sample = test_sample.dropna().astype(int)

# training_sample = training_sample.drop('occupancy_status', axis=1)
# test_sample = test_sample.drop('occupancy_status', axis=1)

# @title Reset Anomlay Detectors
ad_dict = {}
log_dir = "logs/nsnn2"  # @param {type:"string"}

# Set up the logging directory.
# !mkdir -p $log_dir

# Neg Sampling Random Forest Parameters
nsrf_params = {}

# Neg Sampling Neural Net Parameters
nsnn_params = {}

if data_source == 'smart_buildings':

    nsrf_params['sample_ratio'] = 21.00
    nsrf_params['sample_delta'] = 0.05
    nsrf_params['num_estimators'] = 150
    nsrf_params['criterion'] = 'gini'
    nsrf_params['max_depth'] = 50
    nsrf_params['min_samples_split'] = 10
    nsrf_params['min_samples_leaf'] = 5
    nsrf_params['min_weight_fraction_leaf'] = 0.06
    nsrf_params['max_features'] = 0.26

    # recheck hypers
    nsnn_params['sample_ratio'] = 25.0
    nsnn_params['sample_delta'] = 0.05
    nsnn_params['batch_size'] = 32
    nsnn_params['steps_per_epoch'] = 16
    nsnn_params['epochs'] = 100
    nsnn_params['dropout'] = 0.85
    nsnn_params['layer_width'] = 150
    nsnn_params['n_hidden_layers'] = 2
else:
    raise InvalidDatasetError("You requested an invalid data set (%s)." % data_source)


# PYOD UNSUPER MODELS
outliers_fraction = 0.05
random_state = 17

ad_dict['iso'] = IForest(contamination=outliers_fraction, random_state=random_state)
# pca sometimes doesnt work. gotta check what causes it
ad_dict['pca'] = PCA(contamination=outliers_fraction, random_state=random_state)
ad_dict['cblof'] = CBLOF(contamination=outliers_fraction, check_estimator=False, random_state=random_state)
# need to fix error with ecod when using ahuVavBothOccModeData
# ad_dict['ecod'] = ECOD(contamination=outliers_fraction)
# MADI SUPER MODELS
# @title Add in Negative Sampling Random Forest (ns-rf)
# takes long
# ad_dict['ns-rf'] = NegativeSamplingRandomForestAd(
#     n_estimators=nsrf_params['num_estimators'],
#     criterion=nsrf_params['criterion'],
#     max_depth=nsrf_params['max_depth'],
#     min_samples_split=nsrf_params['min_samples_split'],
#     min_samples_leaf=nsrf_params['min_samples_leaf'],
#     min_weight_fraction_leaf=nsrf_params['min_weight_fraction_leaf'],
#     max_features=nsrf_params['max_features'],
#     max_leaf_nodes=None,
#     min_impurity_decrease=0.0,
#     # min_impurity_split=None,
#     bootstrap=True,
#     oob_score=False,
#     n_jobs=None,
#     random_state=None,
#     verbose=0,
#     warm_start=False,
#     class_weight=None,
#     sample_delta=nsrf_params['sample_delta'],
#     sample_ratio=nsrf_params['sample_ratio'])

# @title Add in Negative Sampling Neural Net (ns-nn)
ad_dict['ns-nn'] = NegativeSamplingNeuralNetworkAD(
    sample_ratio=nsnn_params['sample_ratio'],
    sample_delta=nsnn_params['sample_delta'],
    batch_size=nsnn_params['batch_size'],
    steps_per_epoch=nsnn_params['steps_per_epoch'],
    epochs=nsnn_params['epochs'],
    dropout=nsnn_params['dropout'],
    layer_width=nsnn_params['layer_width'],
    n_hidden_layers=nsnn_params['n_hidden_layers'],
    log_dir=log_dir)

print('Anomaly Detectors: ', list(ad_dict))

# @title Execute Cross-Fold Validation {output-height:"unlimited"}
number_crossfolds = 1  # @param {type:"integer"}
number_folds = 5  # @param {type:"integer"}

def fold_sample(sample: pd.DataFrame, n_folds: int = 5) -> List[Dict[str, pd.DataFrame]]:
    """Splits a sample into N folds.

    Args:
      sample: training/test sample to be folded.
    """
    sample = shuffle(sample)

    folds = []
    # Split into train and test folds, and assign to list called folds.
    for training_sample_idx, test_sample_idx in KFold(n_splits=5).split(sample):
        test_sample = sample.iloc[test_sample_idx]
        training_sample = sample.iloc[training_sample_idx]
        folds.append({"train": training_sample, "test": test_sample})
    return folds


def plot_auc(ad_results: Dict[str, Dict[str, Dict[str, np.array]]],
             experiment_name: str):
    """Plots the ROC AUC. """

    fig, ax = plt.subplots(figsize=(15, 15))
    start = 0.0
    stop = 1.0
    colors = [cm.jet(x) for x in np.linspace(start, stop, len(ad_results))]

    df_auc = pd.DataFrame()

    lw = 2
    ix = 0
    for ad_id in ad_results:

        fold_results = ad_results[ad_id]
        vfprs = []
        vtprs = []

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 200)
        validation_aucs = []
        for fold_id in fold_results:
            fpr = fold_results[fold_id]['fpr']
            tpr = fold_results[fold_id]['tpr']

            validation_auc_val = auc(fpr, tpr)
            validation_aucs.append(validation_auc_val)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)

        mean_tpr = np.mean(tprs, axis=0)

        mean_tpr[-1] = 1.0
        mean_auc = np.mean(validation_aucs)
        df_auc[ad_id] = [mean_auc]

        plt.plot(mean_fpr, mean_tpr, color=colors[ix], lw=lw,
                 label='%s: %0.1f%% (%d)' % (
                     ad_id, 100.0 * mean_auc, len(fold_results)))
        ix += 1

        std_tpr = np.std(tprs, axis=0)

        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='gray', alpha=0.5,
                         label=None)

    ax.grid(linestyle='-', linewidth='0.5', color='darkgray')
    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC curves for %s' % experiment_name)

    legend = plt.legend(loc='lower right', shadow=False, fontsize='20')
    legend.get_frame().set_facecolor('white')

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)

    for sp in ax.spines:
        ax.spines[sp].set_color("black")

    plt.show()

def handleZeroDivisionError(n, d):
    if d == 0:
        return 'division by 0... no positives predicted'
    else:
        return n / d

def metrics(actual, predicted):

    print(f'accuracy score: {round(accuracy_score(actual, predicted), 2)}')
    cf_mat = confusion_matrix(actual, predicted)
    tp = cf_mat[1][1]
    tn = cf_mat[0][0]
    fn = cf_mat[1][0]
    fp = cf_mat[0][1]

    sensitivity = tp / (tp + fn)
    precision = handleZeroDivisionError(tp, (tp + fp))

    print(f'class 0 accuracy: {round(tn / (tn + fn), 2)}')
    print(f'class 1 accuracy (precision): {round(precision, 2)}')  # TP / TP + FP
    print(f'sensitivity (recall): {round(sensitivity, 2)}')  # TP / TP + FN, high -> can distinguish faults well
    print(f'specificity: {round(tn / (tn + fp), 2)}')  # high -> can distinguish non-faults well

    return sensitivity, precision

anomaly_detectors = sorted(list(ad_dict))
experiment_name = "%s with %s" % (ds.name, ", ".join(anomaly_detectors))

df_results = pd.DataFrame(columns=['ad', 'auc', 'sensitivity', 'precision', 'extime'])
ad_results = {}

for ad in anomaly_detectors:

    if ad not in ad_results:
        ad_results[ad] = {}

    for cx_run in range(number_crossfolds):
        folds = fold_sample(ds.sample.astype(int), n_folds=number_folds)

        for fid in range(number_folds):
            fold = folds[fid]

            # Drop the class label from the training sample, since this is unsupervised.
            training_sample = fold['train'].copy()
            testing_sample = fold['test'].copy()
            X_train = training_sample.drop(columns=['class_label'])
            X_test = testing_sample.drop(columns=['class_label'])
            y_test = testing_sample['class_label']

            start_time = time.time()

            # Train a model in the training split.
            if ad in ['iso', 'ecod', 'pca', 'cblof']:
                # X_train = X_train.astype(int)
                # X_test = X_test.astype(int)
                ad_dict[ad].fit(X_train)
                y_predicted = ad_dict[ad].decision_function(X_test)
                y_predicted_binary = ad_dict[ad].predict(X_test)
            else:
                ad_dict[ad].train_model(x_train=X_train)
                preds = ad_dict[ad].predict(X_test)
                y_predicted = preds['class_prob']
                y_predicted_binary = preds['Anomaly']

            # Predict on the test set.
            # y_predicted = ad_dict[ad].predict(X_test)['class_prob']

            # Compute the AUC on the test set.
            auc_value = evaluation_utils.compute_auc(
                y_actual=y_test, y_predicted=y_predicted)

            sensitivity, precision = metrics(y_test, y_predicted_binary)

            # Compute the ROC curve.
            fpr, tpr, _ = roc_curve(y_test, y_predicted)

            end_time = time.time()
            extime = end_time - start_time
            ad_results[ad]['%03d-%02d' % (cx_run, fid)] = {'fpr': fpr, 'tpr': tpr}
            df_results.loc['%03d-%02d-%s' % (cx_run, fid, ad)] = [ad, auc_value, sensitivity, precision, extime]

            # Refresh the output area.
            clear_output()

            plot_auc(ad_results, experiment_name=experiment_name)

            del training_sample
            del testing_sample

print("Final Results:")
print(df_results)
