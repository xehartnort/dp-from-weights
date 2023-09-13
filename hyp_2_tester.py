import json
import pandas as pd
import numpy as np
import lightgbm as lgbm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from collections import defaultdict

def train_val_test_split(dataX, dataY, random_state=323223, shuffle=True, train_ratio=0.60, val_ratio=0.15, test_ratio=0.25):
    """https://datascience.stackexchange.com/questions/15135/train-test-validation-set-splitting-in-sklearn"""
    assert train_ratio+val_ratio+test_ratio == 1
    x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=1 - train_ratio, random_state=random_state, shuffle=shuffle)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + val_ratio), random_state=random_state, shuffle=shuffle) 
    return x_train, y_train, x_val, y_val, x_test, y_test

def load_json_file(filepath):
    """Retrieve results in the json file."""
    with open(filepath, "r") as json_fp:
        info = json.load(json_fp)
    return info

# Paths to best meta-classifiers params
meta_classifier_fnc_zoo_params = "./fc_all.json"
meta_classifier_cnn_zoo_params = "./cnn.json"

# Paths to CNN-Zoo and FCN-Zoo models parent directory
cnn_zoo_path = "."
fcn_zoo_path = "."

DATASETS = ['mnist', 'fashion_mnist', "svhn_cropped", "cifar10"]

dnn_pd_dataframes = {}
dnn_no_pd_dataframes = {}
cnn_pd_dataframes = {}
cnn_no_pd_dataframes = {}
for i in DATASETS:
    key_name = f"{i}"
    dnn_pd_dataframes[key_name] = pd.read_pickle(f"{fcn_zoo_path}/{i}_models_dataset/processed.pkl")
    dnn_no_pd_dataframes[key_name] = pd.read_pickle(f"{fcn_zoo_path}/{i}_models_dataset_no_PD/processed.pkl")
    cnn_pd_dataframes[key_name] = pd.read_pickle(f"{cnn_zoo_path}/{i}_models_dataset/processed.pkl")
    cnn_no_pd_dataframes[key_name] = pd.read_pickle(f"{cnn_zoo_path}/{i}_models_dataset_no_PD/processed.pkl")

dataframe = pd.read_pickle(f"{fcn_zoo_path}/mnist_models_dataset/processed.pkl")
column_names = dataframe.columns
WEIGHTS_LAYER_1_STATS_COLS = [c for c in column_names if c.startswith("layer_1")]
dnn_data_frames = {}
for pd_dataset, no_pd_dataset in zip(dnn_pd_dataframes, dnn_no_pd_dataframes):
    X1 = dnn_pd_dataframes[pd_dataset]
    X2 = dnn_no_pd_dataframes[no_pd_dataset]
    X = pd.concat([X1,X2], axis=0)
    y = np.concatenate((np.ones(len(X1)), np.zeros(len(X2))))
    dnn_data_frames[pd_dataset] =  (
                X[WEIGHTS_LAYER_1_STATS_COLS],
                y,
            )

dataframe = pd.read_pickle(f"{cnn_zoo_path}/mnist_models_dataset/processed.pkl")
column_names = dataframe.columns
WEIGHTS_LAYER_3_STATS_COLS = [c for c in column_names if c.startswith("layer_3")]
cnn_data_frames = {}
for pd_dataset, no_pd_dataset in zip(cnn_pd_dataframes, cnn_no_pd_dataframes):
    X1 = cnn_pd_dataframes[pd_dataset]
    X2 = cnn_no_pd_dataframes[no_pd_dataset]
    X = pd.concat([X1,X2], axis=0)
    y = np.concatenate((np.ones(len(X1)), np.zeros(len(X2))))
    cnn_data_frames[pd_dataset] =  (
                X[WEIGHTS_LAYER_3_STATS_COLS],
                y,
            )

fc_zoo_params = load_json_file(meta_classifier_fnc_zoo_params)
cnn_zoo_params = load_json_file(meta_classifier_cnn_zoo_params)
fc_zoo_shared_keys = [
    #"_models_dataset_layer_1_weights-stats",
    "_models_dataset_layer_1_weights-stats+metrics",
    ]
cnn_zoo_shared_keys = [
    # "_models_dataset_layer_3_weights-stats",
    "_models_dataset_layer_3_weights-stats+metric_cols",
    ]

values = defaultdict(list)
pp_names = {
    "mnist": "MNIST",
    "fashion_mnist": "\makecell{Fashion\\\MNIST}",
    "svhn_cropped": "\makecell{Grayscale\\\SVHN}",
    "cifar10": "\makecell{Grayscale\\\CIFAR 10}",
}
for dnn in dnn_data_frames:
    x_data, y_data = dnn_data_frames[dnn]
    train_x, train_y, val_x, val_y, test_x, test_y = train_val_test_split(x_data, y_data, shuffle=False)
    params_key = dnn+fc_zoo_shared_keys[0]
    lgbm_model = lgbm.LGBMClassifier(**fc_zoo_params[params_key])
    lgbm_model = lgbm_model.fit(train_x, train_y,
                    eval_set=[(val_x, val_y)],
                    callbacks=[
                        lgbm.early_stopping(250, verbose=0),
                    ],
                    )
    # Fijado un dnn, pasar a cnn
    for cnn in cnn_data_frames:
        x_data, y_data = cnn_data_frames[cnn]
        pred = lgbm_model.predict(x_data)
        b_acc = metrics.accuracy_score(y_data, pred)
        f1 = metrics.f1_score(y_data, pred)
        precision = metrics.precision_score(y_data, pred)
        recall = metrics.recall_score(y_data, pred)
        values[dnn].append(b_acc)

# Print meta-classifier perfomance table
print("from dnn to cnn")
print(";", ";".join([pp_names[i] for i in DATASETS]))
for ds_name in values:
    content = ";".join([str(i) for i in values[ds_name]])
    print(f"{pp_names[ds_name]};{content}")

values = defaultdict(list)
print("--------")
for cnn in cnn_data_frames:
    x_data, y_data = cnn_data_frames[cnn]
    train_x, train_y, val_x, val_y, test_x, test_y = train_val_test_split(x_data, y_data, shuffle=False)
    params_key = cnn+cnn_zoo_shared_keys[0]
    lgbm_model = lgbm.LGBMClassifier(**cnn_zoo_params[params_key])
    lgbm_model = lgbm_model.fit(train_x, train_y,
                    eval_set=[(val_x, val_y)],
                    callbacks=[
                        lgbm.early_stopping(250, verbose=0),
                    ],
                    )

    for dnn in dnn_data_frames:
        x_data, y_data = dnn_data_frames[dnn]
        pred = lgbm_model.predict(x_data)
        b_acc = metrics.accuracy_score(y_data, pred)
        f1 = metrics.f1_score(y_data, pred)
        precision = metrics.precision_score(y_data, pred)
        recall = metrics.recall_score(y_data, pred)
        values[cnn].append(b_acc)

print("from cnn to dnn")

# Print meta-classifier perfomance table
print(";", ";".join([pp_names[i] for i in DATASETS]))
for ds_name in values:
    content = ";".join([str(i) for i in values[ds_name]])
    print(f"{pp_names[ds_name]};{content}")
