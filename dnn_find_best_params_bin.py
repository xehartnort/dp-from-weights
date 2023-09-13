
from __future__ import division

import itertools
import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgbm
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import optuna
import warnings

CONFIGS_PATH_BASE = './'

def train_val_test_split(dataX, dataY, random_state=323223, shuffle=True, train_ratio=0.60, val_ratio=0.15, test_ratio=0.25):
    """https://datascience.stackexchange.com/questions/15135/train-test-validation-set-splitting-in-sklearn"""
    assert train_ratio+val_ratio+test_ratio == 1
    x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=1 - train_ratio, random_state=random_state, shuffle=shuffle)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + val_ratio), random_state=random_state, shuffle=shuffle) 
    return x_train, y_train, x_val, y_val, x_test, y_test

def extract_summary_features(w, qts=None):
    """Extract various statistics from the flat vector w."""
    if qts:
        features = np.percentile(w, qts)
        feat_names = []
        for i in qts:
            if i == 0:
                feat_names.append("min")
            elif i == 100:
                feat_names.append("max")
            else:
                feat_names.append(f"percentile_{i}")
    else:
        features = []
        feat_names = []
    features = np.append(features, [np.std(w), np.mean(w)])
    feat_names += ["std", "mean"]
    return features, feat_names

def extract_per_layer_features(weights, qts=None, layers=(0, 1)):
    """Extract per-layer statistics from the weight vector and concatenate."""
    # Indices of the location of kernels/biases in the flattened vector

    if len(weights) == 100352+128+128*10+10: # fashion mnist, mnist
        all_boundaries = {
            0: [slice(0, 100352), slice(100352, 100352+128)], # kernel, biases
            1: [slice(100352+128, 100352+128+128*10), slice(100352+128+128*10, 100352+128+128*10+10)]}
    else: # else len(weights) == 131072+128+128*10+10
        all_boundaries = {
            0: [slice(0, 131072), slice(131072, 131072+128)], # kernel, biases
            1: [slice(131072+128, 131072+128+128*10), slice(131072+128+128*10, 131072+128+128*10+10)]}

    full_feat_names = []
    full_features = []
    for layer_i in layers:
        for i, s in enumerate(all_boundaries[layer_i]):
            features, feat_names = extract_summary_features(weights[s], qts)
            part = "kernel" if i == 0 else "biases"
            feat_names = [f"layer_{layer_i}_{part}_{name}" for name in feat_names]
            full_features.append(features)
            full_feat_names += feat_names
    return list(np.concatenate(full_features)), full_feat_names

def load_json_file(filepath):
    """Retrieve results in the json file."""
    with open(filepath, "r") as json_fp:
        info = json.load(json_fp)
    return info

def save_json(stuff, filepath):
    """Save stuff in a json file."""
    with open(filepath, "w") as json_fp:
        json.dump(stuff, json_fp)

def combine_data(dataset_dir, binarize=True):
    processed_name = f"{dataset_dir}/processed.pkl"
    processed_name = os.path.join(CONFIGS_PATH_BASE, processed_name)
    try:
        dataframe = pd.read_pickle(processed_name)
        column_names = dataframe.columns
        WEIGHTS_LAYER_0_STATS_COLS = [c for c in column_names if c.startswith("layer_0")]
        WEIGHTS_LAYER_1_STATS_COLS = [c for c in column_names if c.startswith("layer_1")]
        return dataframe, WEIGHTS_LAYER_0_STATS_COLS, WEIGHTS_LAYER_1_STATS_COLS
    except:
        pass

    all_data_dir = os.path.join(CONFIGS_PATH_BASE, dataset_dir)
    str_path = os.path.join(all_data_dir, 'a_exp.json')
    ref_dataframe = pd.read_json(str_path)
    base_cols = {
        # Other
        # 'exp_id': pd.Series(dtype='int'),
        # Metric cols
        'train_loss': pd.Series(dtype='float'),
        'train_accuracy': pd.Series(dtype='float'),
        'test_loss': pd.Series(dtype='float'),
        'test_accuracy': pd.Series(dtype='float'),
        # Hyperparameters
        'activation': pd.Series(dtype='str'),
        'w_init': pd.Series(dtype='str'),    ## NUEVO
        'optimizer': pd.Series(dtype='str'),    ## NUEVO
        # 'batchsize': pd.Series(dtype='int'),
        'learning_rate': pd.Series(dtype='float'),
        'l2_clip': pd.Series(dtype='float'),
        # DP hyperparameters
        'steps': pd.Series(dtype='int'),
        'sample_size': pd.Series(dtype='int'),
        # 'noise_mult': pd.Series(dtype='float'),
        'eps': pd.Series(dtype='float'),
    }
    # Add variables for layers weights stats
    tmp_weight_stats_names = []
    for i, j in itertools.product(range(2), range(14)):
        k = f'layer_{i}_weight_stat_{j}'
        base_cols[k] = pd.Series(dtype='float')
        tmp_weight_stats_names.append(k)
    dataframe = pd.DataFrame(base_cols)
    pathlist = Path(all_data_dir).rglob('*.json')
    for path in pathlist:
        # because path is object not string
        str_path = str(path)
        if not str_path.endswith("a_exp.json"):
            exp_file = load_json_file(str_path)
            # get exp_id
            str_path = str_path.split("/")[-1] # get last item in path
            str_path = str_path[:-5] # remove .json end
            exp_id = str_path.split("-")[-1] # get exp id
            exp_id = int(exp_id)
            ref_row = ref_dataframe.loc[ref_dataframe['exp_id'] == exp_id]

            new_row = []
            # Metric cols
            new_row += [
                list(exp_file['train_loss'].values())[-1],
                list(exp_file['train_accuracy'].values())[-1],
                list(exp_file['test_loss'].values())[-1],
                list(exp_file['test_accuracy'].values())[-1]
            ]
            # Hyperparameters
            new_row += [
                # exp_file['config.batchsize'],
                ref_row.activation.to_numpy()[0],
                ref_row.w_init.to_numpy()[0],
                ref_row.optimizer.to_numpy()[0],
                float(ref_row.learning_rate.iloc[0]),
                float(ref_row.l2_clip.iloc[0]),
                # DP hyperparams
                exp_file['steps'],
                exp_file['sample_size'],
                # exp_file['config.noise_mult'],
            ]
            new_row += [exp_file['eps']] if 'eps' in exp_file else [None]
            # Stats from raw weights
            flat_w = exp_file['flat_weights']
            feats, feat_names = extract_per_layer_features(flat_w, qts=(0, 25, 50, 75, 100), layers=(0,))
            new_row += feats
            WEIGHTS_LAYER_0_STATS_COLS = feat_names
            feats, feat_names= extract_per_layer_features(flat_w, qts=(0, 25, 50, 75, 100), layers=(1,))
            WEIGHTS_LAYER_1_STATS_COLS = feat_names
            new_row += feats
            dataframe.loc[len(dataframe.index)] = new_row
    # Rename weight stats columns
    rename_dict = dict(zip(tmp_weight_stats_names, WEIGHTS_LAYER_0_STATS_COLS+WEIGHTS_LAYER_1_STATS_COLS))
    dataframe.rename(columns=rename_dict, inplace=True)
    # Binarize categorical features
    CATEGORICAL_CONFIG_PARAMS = ['w_init', 'activation', 'optimizer']
    CATEGORICAL_CONFIG_PARAMS_PREFIX = ['winit', 'act', 'opt']
    if binarize:
        dataframe = pd.get_dummies(
            dataframe,
            columns=CATEGORICAL_CONFIG_PARAMS,
            prefix=CATEGORICAL_CONFIG_PARAMS_PREFIX)
    else:
        # Make the categorical features have pandas type "category"
        # Then LGBM can use those as categorical
        dataframe.is_copy = False
        for col in CATEGORICAL_CONFIG_PARAMS:
            dataframe[col] = dataframe[col].astype('category')
    # Store processed dataframe for faster loading time
    dataframe.to_pickle(processed_name)
    return dataframe, WEIGHTS_LAYER_0_STATS_COLS, WEIGHTS_LAYER_1_STATS_COLS

# Add variables for layers weights stats
HYPERPARAMS_COLS = ['sample_size', 'steps', 'learning_rate', 'activation', "w_init", "optimizer"]
METRIC_COLS = ['train_loss', 'train_accuracy', 'test_loss', 'test_accuracy']


# +
PD_CIFAR10 = "cifar10_models_dataset"
PD_SVHN = "svhn_cropped_models_dataset"
PD_FASHION_MNIST = "fashion_mnist_models_dataset"
PD_MNIST = "mnist_models_dataset"
pd_ds = [PD_CIFAR10, PD_SVHN, PD_CIFAR10, PD_MNIST, PD_FASHION_MNIST]

NO_PD_CIFAR10 = f"{PD_CIFAR10}_no_PD"
NO_PD_SVHN = f"{PD_SVHN}_no_PD"
NO_PD_FASHION_MNIST = f"{PD_FASHION_MNIST}_no_PD"
NO_PD_MNIST = f"{PD_MNIST}_no_PD"
no_pd_ds = [NO_PD_CIFAR10, NO_PD_SVHN, NO_PD_MNIST, NO_PD_FASHION_MNIST]

no_pd_data_frames ={}
for i in tqdm(no_pd_ds):
    data, WEIGHTS_LAYER_0_STATS_COLS, WEIGHTS_LAYER_1_STATS_COLS = combine_data(dataset_dir=i, binarize=False)
    no_pd_data_frames[i] = data

pd_data_frames = {}
for i in tqdm(pd_ds):
    data, WEIGHTS_LAYER_0_STATS_COLS, WEIGHTS_LAYER_1_STATS_COLS = combine_data(dataset_dir=i, binarize=False)
    pd_data_frames[i] = data

# -

data_frames = {}
for pd_dataset, no_pd_dataset in zip(pd_ds, no_pd_ds):
    X1 = pd_data_frames[pd_dataset]
    X2 = no_pd_data_frames[no_pd_dataset]
    X = pd.concat([X1,X2], axis=0)
    # X.fillna(99999, inplace=True)
    y = np.concatenate((np.ones(len(X1)), np.zeros(len(X2))))
    # assert X.isna().sum().sum() == 0
    data_frames[pd_dataset] = {
        'dataset_types': {
            'metrics': (X[METRIC_COLS], y),
            'hyperparams': (X[HYPERPARAMS_COLS], y),
            'hyperparams+metrics': (X[HYPERPARAMS_COLS + METRIC_COLS], y),
            'layer_0_weights-stats': (
                X[WEIGHTS_LAYER_0_STATS_COLS],
                y,
            ),
            'layer_1_weights-stats': (
                X[WEIGHTS_LAYER_1_STATS_COLS],
                y,
            ),
            'layer_1_weights-stats+metrics': (
                X[WEIGHTS_LAYER_1_STATS_COLS + METRIC_COLS],
                y,
            ),
            'layer_1_weights-stats+hyperparams': (
                X[WEIGHTS_LAYER_1_STATS_COLS + HYPERPARAMS_COLS],
                y,
            ),
            'layer_1_weights-stats+metrics+hyperparams': (
                X[WEIGHTS_LAYER_1_STATS_COLS + METRIC_COLS + HYPERPARAMS_COLS],
                y,
            ),
            'all_weights-stats': (
                X[WEIGHTS_LAYER_0_STATS_COLS + WEIGHTS_LAYER_1_STATS_COLS],
                y,
            )
        }
    }

warnings.filterwarnings("ignore", category=UserWarning)

early_stopping_rounds = 250

def objective(trial, X, y):
    param_grid = {
        "device_type": 'cpu',
        "objective": "binary",
        "num_leaves": trial.suggest_int("num_leaves", 20, 1e4),
        "max_depth": trial.suggest_int("max_depth", 5, 15),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 1e-1, log=True), # Small learning rate
        "max_bin": trial.suggest_int('max_bin', 2**6-1, 2**8-1), # Large max_bin
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100), # l1 reg, lambda_l1
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 5),  # l2 reg
        "subsample": trial.suggest_float("subsample", 0.1, 1),
        "subsample_freq" : 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree",  0.7, 1), # fraction of features to use
        "n_estimators": 1500,
        # Other fixed parameters
        "first_metric_only": True,
        "extra_trees": True, # use extremely randomized trees, can sped up and reduce overfit
        'verbose': -1
    }
    lgb_train = lgbm.Dataset(X, y)
    result = lgbm.cv(params=param_grid, 
            train_set=lgb_train, 
            # metrics=eval_metric,
            stratified=True,
            nfold=3,
            callbacks=[
                        lgbm.early_stopping(early_stopping_rounds, verbose=0)
                    ],
            )
    return result['binary_logloss-mean'][-1]


# -
num_trials = 500
best_params = {}
best_metrics = {}
for i in pd_ds:
    for k in tqdm(data_frames[i]['dataset_types']):
        x_data, y_data = data_frames[i]['dataset_types'][k]
        train_x, train_y, val_x, val_y, _, _ = train_val_test_split(x_data, y_data, shuffle=False)
        all_train_x = pd.concat((train_x, val_x), axis=0)
        all_train_y = np.concatenate((train_y, val_y), axis=0)
        study = optuna.create_study(direction="minimize", study_name=f"LGBM {i} {k}")
        func = lambda trial: objective(trial, all_train_x, all_train_y)
        study.optimize(func, n_trials=num_trials)
        params_key = f"{i}_{k}" # datasetName_datasetType
        best_params[params_key] = study.best_params
        best_metrics[params_key] = study.best_value
        print(f"\tBest value (log-loss): {study.best_value:.5f} for {params_key}")
        print(f"\tBest params: {study.best_params}")

# Store best_parms
store_name = 'fcn.json'
save_json(best_params, store_name)