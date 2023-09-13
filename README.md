# dp-from-weights
A repository to reproduce the experiments of the paper: Deep Learning models with privacy: certificating Differential Privacy by recognizing its imprint in model weights.

To reproduce the experiments you need to follow the following steps:

## 1. Generate the Zoos

We recomend running this step using the following docker container: `xehartnort/tfgpu`

FCN-Zoo:

```python
python exp_runner_dnn.py -ds <dataset> -dp 0 # Generate 10.000 models trained on <dataset> without DP
python exp_runner_dnn.py -ds <dataset> -dp 1 # Generate 10.000 models trained on <dataset> with DP
```

CNN-Zoo:

```python
python exp_runner_cnn.py -ds <dataset> -dp 0 # Generate 10.000 models trained on <dataset> without DP
python exp_runner_cnn.py -ds <dataset> -dp 1 # Generate 10.000 models trained on <dataset> with DP
```

Beware that in both cases the datasets will be stored in `./<dataset>_models_dataset_no_PD /` (without DP) and in `./<dataset>_models_dataset/` (with DP)

where `<dataset>` can be: `mnist`, `fashion_mnist`, `svhn_cropped`, `cifar10`

## 1. Alternative, download the Zoos and unzip them

Given that generating the model Zoos take a lot of time, even on multiple GPUs, we provide links to download each Zoo individually. Remember to extract them.

## 2. Train the meta-classifiers and obtain hypothesis I results

We recomend running this step using the following docker container: `xehartnort/lightgbm-gpu`

The following generates the meta-classifiers for the Zoos, stores the best configurations in `fcn.json` and `cnn.json`, and, respectively:

FCN-Zoo:

```python
python dnn_find_best_params_bin.py
```

CNN-Zoo:

```python
python cnn_find_best_params_bin.py
```

## 3. Obtain hypothesis II results

We recomend running this step using the following docker container: `xehartnort/lightgbm-gpu`

First, fill in the correct values for the variables:

```python
# Paths to best meta-classifiers params
meta_classifier_fnc_zoo_params = "./fcn.json"
meta_classifier_cnn_zoo_params = "./cnn.json"

# Paths to CNN-Zoo and FCN-Zoo models parent directory
cnn_zoo_path = "."
fcn_zoo_path = "."
```
in file `hyp_2_tester.py`.

Then, run the following command:

```python
python hyp_2_tester.py
```