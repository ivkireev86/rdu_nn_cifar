"""Валидация моделей
"""
import keras
import numpy as np
from keras.datasets import cifar10  # subroutines for fetching the CIFAR-10 dataset
from keras.preprocessing.image import ImageDataGenerator
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


def vgg_train(model, x_train, y_train, x_test, y_test,
              batch_size=128,
              maxepoches=250,
              learning_rate=0.1,
              lr_drop=20,
              ):
    def lr_scheduler(epoch):
        return learning_rate * (0.5 ** (epoch // lr_drop))

    reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

    # data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    historytemp = model.fit_generator(datagen.flow(x_train, y_train,
                                                   batch_size=batch_size),
                                      steps_per_epoch=x_train.shape[0] // batch_size,
                                      epochs=maxepoches,
                                      validation_data=(x_test, y_test), callbacks=[reduce_lr], verbose=2)


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
