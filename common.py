"""Валидация моделей
"""
import numpy as np
from keras.datasets import cifar10  # subroutines for fetching the CIFAR-10 dataset
from keras.utils import np_utils  # utilities for one-hot encoding of ground truth values

BATCH_SIZE = 32  # in each iteration, we consider 32 training examples at once
NUM_EPOCHS = 20  # we iterate 20 times over the entire training set


def load_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()  # fetch CIFAR-10 data
    num_classes = np.unique(y_train).shape[0]  # there are 10 image classes

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    max_x_values = np.max(X_train)
    X_train /= max_x_values  # Normalise data to [0, 1] range
    X_test /= max_x_values  # Normalise data to [0, 1] range

    Y_train = np_utils.to_categorical(y_train, num_classes)  # One-hot encode the labels
    Y_test = np_utils.to_categorical(y_test, num_classes)  # One-hot encode the labels
    return X_train, Y_train, X_test, Y_test


def validation(model, x_train, y_train, x_test, y_test,
               batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=2):
    if verbose > 0:
        print("Model: ")
        loss, acc = model.evaluate(x_test, y_test, verbose=0)  # Evaluate the trained model on the test set!
        print(loss, acc)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=verbose,
              validation_split=0.1)

    loss, acc = model.evaluate(x_test, y_test, verbose=0)  # Evaluate the trained model on the test set!
    return loss, acc


def compare_models(models, x_train, y_train, x_test, y_test, start_from=0):
    res = []
    for i, model in enumerate(models):
        if i < start_from:
            continue
        loss, acc = validation(model, x_train, y_train, x_test, y_test, verbose=0)
        print("{:4d}   {:.6f}   {:.6f}   {}".format(i, loss, acc, model.name))
        res.append((loss, acc, model.name))
    print('')
    return res
