"""Разные модели
"""
from collections import OrderedDict

from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.models import Model, Sequential  # basic class for specifying and training a neural network


def auto_naming(func):
    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)

        name = str(type(self))
        for k, v in sorted(kwargs.items()):
            name += " {}={}".format(k, v)
        self.name = name

    return wrapper


class BaseModel(Model):
    @auto_naming
    def __init__(self, height, width, depth, num_classes,
                 conv_depth_1=32, conv_depth_2=64, kernel_size=3, pool_size=2,
                 drop_prob_1=0.25, drop_prob_2=0.5, hidden_size=512):
        inp = Input(shape=(height, width, depth))
        # Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
        conv_1 = Conv2D(conv_depth_1, kernel_size, padding='same', activation='relu')(inp)
        conv_2 = Conv2D(conv_depth_1, kernel_size, padding='same', activation='relu')(conv_1)
        pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
        drop_1 = Dropout(drop_prob_1)(pool_1)
        # Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
        conv_3 = Conv2D(conv_depth_2, kernel_size, padding='same', activation='relu')(drop_1)
        conv_4 = Conv2D(conv_depth_2, kernel_size, padding='same', activation='relu')(conv_3)
        pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
        drop_2 = Dropout(drop_prob_1)(pool_2)
        # Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax
        flat = Flatten()(drop_2)
        hidden = Dense(hidden_size, activation='relu')(flat)
        drop_3 = Dropout(drop_prob_2)(hidden)
        out = Dense(num_classes, activation='softmax')(drop_3)

        super().__init__(inputs=inp, outputs=out)  # To define a model, just specify its input and output layers
        self.compile(loss='categorical_crossentropy',  # using the cross-entropy loss function
                     optimizer='adam',  # using the Adam optimiser
                     metrics=['accuracy'])  # reporting the accuracy


class SmallModel(Sequential):
    @auto_naming
    def __init__(self, height, width, depth, num_classes, hidden_size):
        out_layers = [
            Flatten(input_shape=(height, width, depth)),
            Dense(hidden_size, activation='relu'),
            Dense(num_classes, activation='softmax')
        ]

        super().__init__(out_layers)
        self.compile(loss='categorical_crossentropy',  # using the cross-entropy loss function
                     optimizer='adam',  # using the Adam optimiser
                     metrics=['accuracy'])  # reporting the accuracy


class VggLikeModel(Model):
    @auto_naming
    def __init__(self, height, width, depth, num_classes,
                 conv_params, dense_size, dense_dropout_rate):
        inp_layer = Input(shape=(height, width, depth))
        out_layer = inp_layer
        for conv_param in conv_params:
            out_layer = self.get_conv(out_layer, **conv_param)

        out_layer = Flatten(input_shape=(height, width, depth))(out_layer)
        if dense_size:
            for d_size in dense_size:
                out_layer = Dense(d_size, activation='relu')(out_layer)
                out_layer = Dropout(rate=dense_dropout_rate)(out_layer)
        out_layer = Dense(num_classes, activation='softmax')(out_layer)

        super().__init__(inputs=inp_layer, outputs=out_layer)
        self.compile(loss='categorical_crossentropy',  # using the cross-entropy loss function
                     optimizer='adam',  # using the Adam optimiser
                     metrics=['accuracy'])  # reporting the accuracy

    @staticmethod
    def get_conv(inp_layer, conv_count=2, filters=16, kernel_size=3, activation=None, dropout_rate=0.25):
        out_layer = inp_layer
        for _ in range(conv_count):
            out_layer = Conv2D(filters=filters, kernel_size=kernel_size, activation=activation)(out_layer)
        out_layer = MaxPooling2D(padding='same')(out_layer)
        out_layer = Dropout(rate=dropout_rate)(out_layer)
        return out_layer


def get_all_models(height, width, depth, num_classes):
    return [
        BaseModel(height, width, depth, num_classes),
        SmallModel(height, width, depth, num_classes, hidden_size=512),
        SmallModel(height, width, depth, num_classes, hidden_size=128),
        VggLikeModel(height, width, depth, num_classes, conv_params=[
            {'conv_count': 1, 'filters': 16, 'activation': None, 'dropout_rate': 0.25},
        ], dense_size=[128], dense_dropout_rate=0.5),
        VggLikeModel(height, width, depth, num_classes, conv_params=[
            {'conv_count': 2, 'filters': 16, 'activation': None, 'dropout_rate': 0.25},
        ], dense_size=[128], dense_dropout_rate=0.5),
        VggLikeModel(height, width, depth, num_classes, conv_params=[
            {'conv_count': 2, 'filters': 32, 'activation': None, 'dropout_rate': 0.25},
        ], dense_size=[128], dense_dropout_rate=0.5),
        VggLikeModel(height, width, depth, num_classes, conv_params=[
            {'conv_count': 3, 'filters': 16, 'activation': None, 'dropout_rate': 0.25},
        ], dense_size=[128], dense_dropout_rate=0.5),
        VggLikeModel(height, width, depth, num_classes, conv_params=[
            {'conv_count': 2, 'filters': 16, 'activation': None, 'dropout_rate': 0.50},
        ], dense_size=[128], dense_dropout_rate=0.5),
        VggLikeModel(height, width, depth, num_classes, conv_params=[
            {'conv_count': 2, 'filters': 16, 'activation': None, 'dropout_rate': 0.25},
        ], dense_size=[512], dense_dropout_rate=0.5),
        VggLikeModel(height, width, depth, num_classes, conv_params=[
            {'conv_count': 2, 'filters': 16, 'activation': None, 'dropout_rate': 0.25},
        ], dense_size=[], dense_dropout_rate=0.5),
        VggLikeModel(height, width, depth, num_classes, conv_params=[
            {'conv_count': 2, 'filters': 16, 'activation': None, 'dropout_rate': 0.25},
        ], dense_size=[256], dense_dropout_rate=0.5),
        VggLikeModel(height, width, depth, num_classes, conv_params=[
            {'conv_count': 2, 'filters': 16, 'activation': None, 'dropout_rate': 0.25},
        ], dense_size=[256, 32], dense_dropout_rate=0.5),
    ]


if __name__ == '__main__':
    models = get_all_models(32, 32, 3, 10)
    for model in models:
        print(model.name)
    print('test ok')
