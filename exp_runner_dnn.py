from sultan.api import Sultan
import numpy as np
import json
import os
import argparse
import tensorflow_privacy as tf_privacy
from joblib import Parallel, delayed
import pandas as pd


def save(stuff, filepath):
    """Save stuff in a json file."""
    with open(filepath, "w") as json_fp:
        json.dump(stuff, json_fp)


def countdown(bs, nm, size, epochs):
    return tf_privacy.compute_dp_sgd_privacy(
            n=size,
            batch_size=bs,
            noise_multiplier=nm,
            epochs=epochs,
            delta=1e-5,
        )[0]


def restore(filepath):
    """Retrieve results in a json file."""
    with open(filepath, "r") as json_fp:
        info = json.load(json_fp)
    return info

# Training settings
parser = argparse.ArgumentParser(
    description="Train a fixed number of models with multiple configurations",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-n",
    "--num-exp",
    type=int,
    default=10_000,
    help="Number of experiments",
)
parser.add_argument(
    "-dp",
    "--diff-priv",
    type=int,
    default=1,
    help="Whether to enable or no dp",
)
parser.add_argument(
    "-ds",
    "--dataset",
    type=str,
    default='spiral',
    help="Name of the dataset to train the models on: spiral, aniso_blobs, noisy_gaussian_quantiles",
)

if __name__ == "__main__":
    EPOCHS = 5
    np.random.seed(0)
    args = parser.parse_args()
    # Number of experiments to run
    num_exp = args.num_exp
    ds = args.dataset
    # Workdir
    wd = f"./{ds}_models_dataset/"
    if args.diff_priv == 0:
        wd = f"./{ds}_models_dataset_no_PD/"
    os.makedirs(wd, exist_ok=True)  # succeeds even if directory exists.
    # File that stores the current state of the experimentation
    exp_file = "a_exp.json" 
    filepath = f"{wd}{exp_file}"
    # Try to load the exp state, otherwise create a new one
    try:
        exp_info = restore(filepath)
        print("Restoring from file: ", filepath)
    except:
        # Dataset size, the same for mnist, cifar10, cifar100 and fashion_mnist
        if (ds == "mnist") or (ds == "fashion_mnist"):
            ds_size = 60_000
        elif ds == "svhn_cropped":
            ds_size = 73_257
        else: # cifar10
            ds_size = 50_000
        # L2 norm clip for gradient updates
        l2_range = np.linspace(start=0.1, stop=1.5, num=100)
        # Batch size values
        bs_range = np.arange(start=2**5, stop=2**11, step=1)
        # Noise multiplier values
        nm_range = np.linspace(start=1e-3, stop=1.5, num=int(num_exp))
        # Learning rate values
        lr_range = np.linspace(start=0.001, stop=0.1, num=num_exp)
        # Activation function
        activation = ["relu", "tanh"]
        # Weight initialization scheme
        w_init = ["glorot_normal", "RandomNormal", "TruncatedNormal", "orthogonal", "he_normal"]
        # Optimizers available in tf privacy
        optimizer = ["adam", "sgd"]
        # Fraction of the available training data to use
        train_fraction = np.arange(start=0.3, stop=1, step=0.05)
        # Different init for each exp
        init_std = np.linspace(start=0.1, stop=0.5, num=num_exp)
        # Number of epochs per experiment
        e_range = [EPOCHS]
        # PD related params
        # Change bs, nm and train_size to create a more stable eps distribution
        tmp_bs = np.random.choice(bs_range, size=int(num_exp*2.5))
        tmp_nm = np.random.choice(nm_range, size=int(num_exp*2.5))
        tmp_train_size = np.random.choice(ds_size*train_fraction, size=int(num_exp*2.5))
        tmp_epochs = e_range*int(num_exp*2.5)
        # Compute eps distribution
        eps = Parallel(n_jobs=-1)(delayed(countdown)(*args) for args in zip(tmp_bs, tmp_nm, tmp_train_size, tmp_epochs))
        
        dataframe = pd.DataFrame({
            'batchsize': tmp_bs,
            'noise_mult': tmp_nm,
            'train_fraction': tmp_train_size/ds_size,
            'eps': eps,
            })
        max_eps = 10
        dataframe = dataframe.loc[dataframe.eps<max_eps]
        # Downsample to have 10_000 items
        indx = np.arange(len(dataframe))
        indx_to_remove = np.random.choice(indx[(dataframe.eps>=1) & (dataframe.eps<=5)], replace=False, size=len(dataframe)-num_exp)
        remaining = list(set(indx) - set(indx_to_remove))
        dataframe = dataframe.iloc[remaining]
        # Save results
        bs_range = dataframe.batchsize.tolist()
        nm_range = dataframe.noise_mult.tolist()
        train_fraction = dataframe.train_fraction.tolist()
        if args.diff_priv == 0:
            # Noise multiplier values
            nm_range = [0]
            # L2 norm clip for gradient updates
            l2_range = np.concatenate((l2_range, np.zeros((len(l2_range),))))

        print("Restoring failed, creating new config", filepath)
        # Build exp_info
        np.random.seed(0)
        exp_info = {
            "num_exp": num_exp,
            "last_run": 0,
            "ds": ds,
            "epochs": np.random.choice(e_range, size=num_exp).tolist(),
            "batchsize": np.random.choice(bs_range, size=num_exp).tolist(),
            "noise_mult": np.random.choice(nm_range, size=num_exp).tolist(),
            "learning_rate": np.random.choice(lr_range, size=num_exp).tolist(),
            "l2_clip": np.random.choice(l2_range, size=num_exp).tolist(),
            "activation": np.random.choice(activation, size=num_exp).tolist(),
            "w_init": np.random.choice(w_init, size=num_exp).tolist(),
            "optimizer": np.random.choice(optimizer, size=num_exp).tolist(),
            "init_std": np.random.choice(init_std, size=num_exp).tolist(),
            "train_fraction": np.random.choice(train_fraction, size=num_exp).tolist(),
            "exp_id": np.arange(num_exp).tolist()
        }
        exp_info["train_size"] = list(np.array(exp_info["train_fraction"]) * ds_size)

        # Save exp info
        save(exp_info, filepath)

    # Continue if possible with experimentation
    save_every = 1
    for i in range(exp_info["last_run"], exp_info["num_exp"]):
        arg_str = (
            f"--epochs {int(exp_info['epochs'][i])}"
            f" --batchsize {int(exp_info['batchsize'][i])}"
            f" --noise_mult {exp_info['noise_mult'][i]}"
            f" --learning_rate {exp_info['learning_rate'][i]}"
            f" --l2_clip {exp_info['l2_clip'][i]}"
            f" --activation {exp_info['activation'][i]}"
            f" --w_init {exp_info['w_init'][i]}"
            f" --optimizer {exp_info['optimizer'][i]}"
            f" --init_std {exp_info['init_std'][i]}"
            f" --train_fraction {exp_info['train_fraction'][i]}"
            f" --exp_id {exp_info['exp_id'][i]}"
            f" --dataset {exp_info['ds']}"
            f" --workdir {wd}"
            " --num_layers 1 --num_units 128 --dnn_architecture fcn"
        )

        arg_str = arg_str.split(" ")
        cmd = "train_network.py"
        with Sultan.load() as s:
            cmd_str = cmd+" ".join(arg_str)
            s.python3(cmd, *arg_str).run()

        # process is still running
        if i % save_every == 0:
            exp_info["last_run"] = i + 1
            save(exp_info, filepath)
