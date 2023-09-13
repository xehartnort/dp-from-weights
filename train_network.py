"""Train DNN of a specified architecture on a specified data set."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tqdm import tqdm

import json
import os
import sys
import time

from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.io import gfile
from tensorflow_privacy.privacy.keras_models.dp_keras_model import DPSequential
import tensorflow_datasets as tfds
import tensorflow_privacy as tf_privacy

FLAGS = flags.FLAGS
CNN_KERNEL_SIZE = 3

flags.DEFINE_integer("num_layers", 3, "Number of layers in the network.")
flags.DEFINE_integer("num_units", 16, "Number of units in a dense layer.")
flags.DEFINE_float(
    "train_fraction",
    1.0,
    "How much of the dataset to use for" "training [as fraction]: eg. 0.15, 0.5, 1.0",
)
flags.DEFINE_integer("random_seed", 42, "Random seed.")
flags.DEFINE_integer("cnn_stride", 2, "Stride of the CNN")
flags.DEFINE_float("dropout", 0.0, "Dropout Rate")
flags.DEFINE_float("l2reg", 0.0, "L2 regularization strength")
flags.DEFINE_float("init_std", 0.05, "Standard deviation of the initializer.")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
flags.DEFINE_string("optimizer", "sgd", "Optimizer algorithm: sgd / adam / momentum.")
flags.DEFINE_string(
    "activation", "relu", "Nonlinear activation: relu / tanh / sigmoind / selu."
)
flags.DEFINE_string(
    "w_init",
    "he_normal",
    "Initialization for weights. " "see tf.keras.initializers for options",
)
flags.DEFINE_string(
    "b_init",
    "zero",
    "Initialization for biases." "see tf.keras.initializers for options",
)
flags.DEFINE_boolean("grayscale", True, "Convert input images to grayscale.")
flags.DEFINE_boolean("augment_traindata", False, "Augmenting Training data.")
flags.DEFINE_boolean("reduce_learningrate", False, "Reduce LR towards end of training.")
flags.DEFINE_string("dataset", "mnist", "Name of the dataset compatible " "with TFDS.")
flags.DEFINE_string(
    "dnn_architecture", "cnn", "Architecture of the DNN [fc, cnn, cnnbn]"
)
flags.DEFINE_string(
    "workdir",
    "./dnn_science_workdir",
    "Base working directory for storing" "checkpoints, summaries, etc.",
)
flags.DEFINE_integer("verbose", 0, "Verbosity")
flags.DEFINE_bool("use_tpu", False, "Whether running on TPU or not.")
flags.DEFINE_string(
    "master", "local", 'Name of the TensorFlow master to use. "local" for GPU.'
)
flags.DEFINE_string(
    "tpu_job_name",
    "tpu_worker",
    "Name of the TPU worker job. This is required when having multiple TPU "
    "worker jobs.",
)

# New parameters we want to change
flags.DEFINE_float("noise_mult", 1.0, "Calibrate noise.")
flags.DEFINE_float("l2_clip", None, "Maximum norm to clip gradients.")
flags.DEFINE_integer(
    "batchsize", 512, "Size of the mini-batch."
)  # 16, 32, 64, 128, 256
flags.DEFINE_integer("epochs", 18, "How many epochs to train for")
flags.DEFINE_integer("exp_id", 0, "Experiment id")
flags.DEFINE_integer(
    "epochs_between_checkpoints",
    6,
    "How many epochs to train between creating checkpoints",
)


def store_results(info_dict, filepath):
    """Save results in the json file."""
    with gfile.GFile(filepath, "w") as json_fp:
        json.dump(info_dict, json_fp)


def restore_results(filepath):
    """Retrieve results in the json file."""
    with gfile.GFile(filepath, "r") as json_fp:
        info = json.load(json_fp)
    return info


def _preprocess_batch(batch, normalize, to_grayscale, augment=False):
    """Preprocessing function for each batch of data."""
    image = tf.cast(batch["image"], tf.float32)
    image /= 255.0

    if augment:
        shape = image.shape
        image = tf.image.resize_with_crop_or_pad(image, shape[1] + 2, shape[2] + 2)
        image = tf.image.random_crop(image, size=shape)

        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_hue(image, 0.08)
        image = tf.image.random_saturation(image, 0.6, 1.6)
        image = tf.image.random_brightness(image, 0.05)
        image = tf.image.random_contrast(image, 0.7, 1.3)

    if normalize:
        min_out = -1.0
        max_out = 1.0
        image = min_out + image * (max_out - min_out)
    if to_grayscale:
        image = tf.math.reduce_mean(image, axis=-1, keepdims=True)
    return image, batch["label"]


def get_dataset(
    dataset,
    batchsize,
    to_grayscale=True,
    train_fraction=1.0,
    shuffle_buffer=1024,
    random_seed=None,
    normalize=True,
    augment=False,
):
    """Load and preprocess the dataset.

    Args:
      dataset: The dataset name. Either 'toy' or a TFDS dataset
      batchsize: the desired batch size
      to_grayscale: if True, all images will be converted into grayscale
      train_fraction: what fraction of the overall training set should we use
      shuffle_buffer: size of the shuffle.buffer for tf.data.Dataset.shuffle
      random_seed: random seed for shuffling operations
      normalize: whether to normalize the data into [-1, 1]
      augment: use data augmentation on the training set.

    Returns:
      tuple (training_dataset, test_dataset, info), where info is a dictionary
      with some relevant information about the dataset.
    """
    if dataset not in [
        "circles",
        "moons",
        "spiral",
        "blobs",
        "aniso_blobs",
        "blobs_varied_variances",
        "noisy_gaussian_quantiles"
    ]:
        data_tr, ds_info = tfds.load(dataset, split="train", with_info=True)
        effective_train_size = ds_info.splits["train"].num_examples

        if train_fraction < 1.0:
            effective_train_size = int(effective_train_size * train_fraction)
            data_tr = data_tr.shuffle(shuffle_buffer, seed=random_seed)
            data_tr = data_tr.take(effective_train_size)

        fn_tr = lambda b: _preprocess_batch(b, normalize, to_grayscale, augment)
        data_tr = data_tr.shuffle(shuffle_buffer, seed=random_seed)
        data_tr = data_tr.batch(batchsize, drop_remainder=True)
        data_tr = data_tr.map(fn_tr, tf.data.experimental.AUTOTUNE)
        data_tr = data_tr.prefetch(tf.data.experimental.AUTOTUNE)

        fn_te = lambda b: _preprocess_batch(b, normalize, to_grayscale, False)
        data_te = tfds.load(dataset, split="test")
        data_te = data_te.batch(batchsize)
        data_te = data_te.map(fn_te, tf.data.experimental.AUTOTUNE)
        data_te = data_te.prefetch(tf.data.experimental.AUTOTUNE)

        dataset_info = {
            "num_classes": ds_info.features["label"].num_classes,
            "data_shape": ds_info.features["image"].shape,
            "train_num_examples": effective_train_size,
        }
    else:
        savepath = f"./synthetic_datasets/{dataset}.npy"
        with open(savepath, "rb") as f:
            X_train, X_test, y_train, y_test = np.load(f, allow_pickle=True)
        effective_train_size = len(X_train)
        data_tr = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        if train_fraction < 1.0:
            effective_train_size = int(effective_train_size * train_fraction)
            data_tr = data_tr.shuffle(shuffle_buffer, seed=random_seed)
            data_tr = data_tr.take(effective_train_size)
        # No preprocessing, just batching
        data_tr = data_tr.batch(batchsize, drop_remainder=True)
        data_tr = data_tr.prefetch(tf.data.experimental.AUTOTUNE)

        # No preprocessing, just batching
        data_te = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        data_te = data_te.batch(batchsize)
        data_te = data_te.prefetch(tf.data.experimental.AUTOTUNE)

        dataset_info = {
            "num_classes": len(np.unique(y_test)),
            "data_shape": X_train[0].shape,
            "train_num_examples": effective_train_size,
        }
    dataset_info["name"] = dataset
    return data_tr, data_te, dataset_info


def build_cnn(
    n_layers,
    n_hidden,
    n_outputs,
    dropout_rate,
    activation,
    stride,
    w_regularizer,
    w_init,
    b_init,
    use_batchnorm,
    dp=False,
    l2_clip=None,
    noise_mult=None,
    num_microbatches=None,
):
    """Convolutional deep neural network."""
    if FLAGS.dataset in ["mnist", "fashion_mnist"]:
        layers = [tf.keras.layers.InputLayer(input_shape=(28,28,1))]
    elif FLAGS.dataset in ["cifar10", "svhn_cropped"]:
        layers = [tf.keras.layers.InputLayer(input_shape=(32,32,1))]
    else:
        layers = []
    for _ in range(n_layers):
        layers.append(
            tf.keras.layers.Conv2D(
                n_hidden,
                kernel_size=CNN_KERNEL_SIZE,
                strides=stride,
                activation=activation,
                kernel_regularizer=w_regularizer,
                kernel_initializer=w_init,
                bias_initializer=b_init,
            )
        )
        if dropout_rate > 0.0:
            layers.append(tf.keras.layers.Dropout(dropout_rate))
        if use_batchnorm:
            layers.append(tf.keras.layers.BatchNormalization())
    layers.append(tf.keras.layers.GlobalAveragePooling2D())
    layers.append(
        tf.keras.layers.Dense(
            n_outputs,
            kernel_regularizer=w_regularizer,
            kernel_initializer=w_init,
            bias_initializer=b_init,
        )
    )
    if dp:
        model = DPSequential(
            l2_norm_clip=l2_clip,
            noise_multiplier=noise_mult,
            num_microbatches=num_microbatches,
            layers=layers,
        )
    else:
        model = tf.keras.Sequential(layers=layers)

    return model


def build_fcn(
    n_layers,
    n_hidden,
    n_outputs,
    dropout_rate,
    activation,
    w_regularizer,
    w_init,
    b_init,
    use_batchnorm,
    dp=False,
    l2_clip=None,
    noise_mult=None,
    num_microbatches=None,
):
    """Fully Connected deep neural network."""
    model = tf.keras.Sequential()
    if FLAGS.dataset in ["mnist", "fashion_mnist"]:
        layers = [tf.keras.layers.InputLayer(input_shape=(28,28,1))]
    elif FLAGS.dataset in ["cifar10", "svhn_cropped"]:
        layers = [tf.keras.layers.InputLayer(input_shape=(32,32,1))]
    else:
        layers = []
    layers.append(tf.keras.layers.Flatten())
    for _ in range(n_layers):
        layers.append(
            tf.keras.layers.Dense(
                n_hidden,
                activation=activation,
                kernel_regularizer=w_regularizer,
                kernel_initializer=w_init,
                bias_initializer=b_init,
            )
        )
        if dropout_rate > 0.0:
            layers.append(tf.keras.layers.Dropout(dropout_rate))
        if use_batchnorm:
            layers.append(tf.keras.layers.BatchNormalization())

    layers.append(
        tf.keras.layers.Dense(
            n_outputs,
            kernel_regularizer=w_regularizer,
            kernel_initializer=w_init,
            bias_initializer=b_init,
        )
    )
    if dp:
        model = DPSequential(
            l2_norm_clip=l2_clip,
            noise_multiplier=noise_mult,
            num_microbatches=num_microbatches,
            layers=layers,
        )
    else:
        model = tf.keras.Sequential(layers=layers)

    return model


def eval_model(model, data_tr, data_te, info, cur_epoch):
    """Runs Model Evaluation."""
    # get training set metrics in eval-mode (no dropout etc.)
    metrics_te = model.evaluate(data_te, verbose=0)
    res_te = dict(zip(model.metrics_names, metrics_te))
    metrics_tr = model.evaluate(data_tr, verbose=0)
    res_tr = dict(zip(model.metrics_names, metrics_tr))
    metrics = {
        "train_accuracy": res_tr["accuracy"],
        "train_loss": res_tr["loss"],
        "test_accuracy": res_te["accuracy"],
        "test_loss": res_te["loss"],
    }
    for k in metrics:
        info[k][cur_epoch] = float(metrics[k])
    metrics["epoch"] = cur_epoch  # so it's included in the logging output
    print(metrics)
    flat_weights = np.concatenate(model.get_weights(), axis=None)  # completely flat
    info[
        "flat_weights"
    ] = (
        flat_weights.tolist()
    )



def run(
    workdir,
    data,
    architecture,
    n_layers,
    n_hiddens,
    activation,
    dropout_rate,
    l2_penalty,
    w_init_name,
    b_init_name,
    optimizer_name,
    learning_rate,
    n_epochs,
    epochs_between_checkpoints,
    init_stddev,
    cnn_stride,
    reduce_learningrate=False,
    verbosity=0,
):
    """Runs the whole training procedure."""
    data_tr, data_te, dataset_info = data
    n_outputs = dataset_info["num_classes"]

    kargs = {
        'learning_rate': learning_rate,
        'clipnorm': FLAGS.l2_clip
    }
    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(**kargs)
    else:
        optimizer = tf.keras.optimizers.SGD(**kargs)

    # else:
    #     kargs = {
    #         "l2_norm_clip": FLAGS.l2_clip,
    #         "noise_multiplier": FLAGS.noise_mult,
    #         "num_microbatches": FLAGS.batchsize,
    #         "learning_rate": learning_rate,
    #         "gradient_accumulation_steps": FLAGS.batchsize, # required to be careful with memory, but it makes the whole process slower
    #     }
    #     if optimizer_name == "adam":
    #         optimizer = tf_privacy.DPKerasAdamOptimizer(**kargs)
    #     else:
    #         optimizer = tf_privacy.DPKerasSGDOptimizer(**kargs)
    # optimizer_name += "_dp"
    w_init = tf.keras.initializers.get(w_init_name)
    if w_init_name.lower() in ["truncatednormal", "randomnormal"]:
        w_init.stddev = init_stddev
    b_init = tf.keras.initializers.get(b_init_name)
    if b_init_name.lower() in ["truncatednormal", "randomnormal"]:
        b_init.stddev = init_stddev
    w_reg = tf.keras.regularizers.l2(l2_penalty) if l2_penalty > 0 else None

    if architecture == "cnn" or architecture == "cnnbn":
        model = build_cnn(
            n_layers,
            n_hiddens,
            n_outputs,
            dropout_rate,
            activation,
            cnn_stride,
            w_reg,
            w_init,
            b_init,
            architecture == "cnnbn",
            dp=FLAGS.noise_mult>0,
            noise_mult=FLAGS.noise_mult,
            l2_clip=FLAGS.l2_clip,
            num_microbatches=1
        )
    elif architecture == "fcn":
        model = build_fcn(
            n_layers,
            n_hiddens,
            n_outputs,
            dropout_rate,
            activation,
            w_reg,
            w_init,
            b_init,
            False,
            dp=FLAGS.noise_mult>0,
            noise_mult=FLAGS.noise_mult,
            l2_clip=FLAGS.l2_clip,
            num_microbatches=1
        )
    else:
        assert False, "Unknown architecture: " % architecture

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.losses.Reduction.NONE
        ),
        metrics=["accuracy", "mse", "sparse_categorical_crossentropy"],
    )

    # force the model to set input shapes and init weights
    for x, _ in data_tr:
        model.predict(x)
        if verbosity:
            model.summary()
        break


    info = {
        "steps": 0,
        "start_time": time.time(),
        "train_loss": {},
        "train_accuracy": {},
        "test_loss": {},
        "test_accuracy": {},
        }

    starting_epoch = len(info["train_loss"])
    cur_epoch = starting_epoch
    info["train_size"] = dataset_info["train_num_examples"]
    info["sample_size"] = FLAGS.batchsize / dataset_info["train_num_examples"]
    from math import ceil
    info["steps"] = int(ceil(FLAGS.epochs * dataset_info["train_num_examples"] / FLAGS.batchsize))

    if FLAGS.noise_mult > 0:
        info["eps"] = tf_privacy.compute_dp_sgd_privacy(
            n=dataset_info["train_num_examples"],
            batch_size=FLAGS.batchsize,
            noise_multiplier=FLAGS.noise_mult,
            epochs=FLAGS.epochs,
            delta=1e-5,
        )[0]
    for cur_epoch in tqdm(range(cur_epoch, n_epochs)):
        if reduce_learningrate and cur_epoch == n_epochs - (n_epochs // 10):
            optimizer.learning_rate = learning_rate / 10
        elif reduce_learningrate and cur_epoch == n_epochs - 2:
            optimizer.learning_rate = learning_rate / 100

        try:
            model.fit(data_tr, epochs=1, verbose=verbosity)
            dt = time.time() - info["start_time"]
            logging.info("epoch %d (%3.2fs)", cur_epoch, dt)
            info["status"] = "Sucess"
        except tf.errors.InvalidArgumentError as e:
            # We got NaN in the loss, most likely gradients resulted in NaNs
            logging.info(str(e))
            info["status"] = "NaN"
            logging.info("Stop training because NaNs encountered")
            break

    # Save only sane models
    if info["status"] != "NaN":
        info["time elapsed"] = time.time() - info["start_time"]

        eval_model(model, data_tr, data_te, info, cur_epoch + 1)
        store_results(info, os.path.join(workdir, f"model-{FLAGS.exp_id}.json"))

    # we don't need the temporary checkpoints anymore
    # gfile.rmtree(os.path.join(workdir, "temporary-ckpt"))
    # gfile.remove(os.path.join(workdir, ".intermediate-results.json"))


def main(unused_argv):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            if FLAGS.dnn_architecture != "cnn":
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120*2)])
        except RuntimeError as e:
            print(e)
    workdir = FLAGS.workdir

    if not gfile.isdir(workdir):
        gfile.makedirs(workdir)

    tf.random.set_seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)
    data = get_dataset(
        FLAGS.dataset,
        FLAGS.batchsize,
        to_grayscale=FLAGS.grayscale,
        train_fraction=FLAGS.train_fraction,
        random_seed=FLAGS.random_seed,
        augment=FLAGS.augment_traindata,
    )

    run(
        workdir,
        data,
        architecture=FLAGS.dnn_architecture,
        n_layers=FLAGS.num_layers,
        n_hiddens=FLAGS.num_units,
        activation=FLAGS.activation,
        dropout_rate=FLAGS.dropout,
        l2_penalty=FLAGS.l2reg,
        w_init_name=FLAGS.w_init,
        b_init_name=FLAGS.b_init,
        optimizer_name=FLAGS.optimizer,
        learning_rate=FLAGS.learning_rate,
        n_epochs=FLAGS.epochs,
        epochs_between_checkpoints=FLAGS.epochs_between_checkpoints,
        init_stddev=FLAGS.init_std,
        cnn_stride=FLAGS.cnn_stride,
        reduce_learningrate=FLAGS.reduce_learningrate,
        verbosity=FLAGS.verbose,
    )


if __name__ == "__main__":
    tf.enable_v2_behavior()
    app.run(main)
