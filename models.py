"""Разные модели
"""

from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Activation
from keras.models import Model, Sequential  # basic class for specifying and training a neural network
from keras import regularizers

from model_bag.core import auto_naming
from model_bag.simple import BaseModel, SmallModel
from model_bag.vgg import VggLikeModel, Vgg16


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
        ###
        VggLikeModel(height, width, depth, num_classes, conv_params=[
            {'conv_count': 2, 'filters': 16, 'activation': None, 'dropout_rate': 0.4},  # 32-16
        ], dense_size=[128], dense_dropout_rate=0.5),
        VggLikeModel(height, width, depth, num_classes, conv_params=[
            {'conv_count': 2, 'filters': 16, 'activation': None, 'dropout_rate': 0.4},  # 32-16-8
            {'conv_count': 2, 'filters': 32, 'activation': None, 'dropout_rate': 0.4},
        ], dense_size=[128], dense_dropout_rate=0.5),
        VggLikeModel(height, width, depth, num_classes, conv_params=[
            {'conv_count': 2, 'filters': 16, 'activation': None, 'dropout_rate': 0.4},  # 32-16-8-4
            {'conv_count': 2, 'filters': 32, 'activation': None, 'dropout_rate': 0.4},
            {'conv_count': 2, 'filters': 64, 'activation': None, 'dropout_rate': 0.4},
        ], dense_size=[128], dense_dropout_rate=0.5),
        VggLikeModel(height, width, depth, num_classes, conv_params=[
            {'conv_count': 2, 'filters': 16, 'activation': None, 'dropout_rate': 0.4},  # 32-16-8-4
            {'conv_count': 2, 'filters': 32, 'activation': None, 'dropout_rate': 0.4},
            {'conv_count': 2, 'filters': 64, 'activation': None, 'dropout_rate': 0.4},
        ], dense_size=[512], dense_dropout_rate=0.5),
        VggLikeModel(height, width, depth, num_classes, conv_params=[
            {'conv_count': 2, 'filters': 16, 'activation': None, 'dropout_rate': 0.4},  # 32-16-8-4-2
            {'conv_count': 2, 'filters': 32, 'activation': None, 'dropout_rate': 0.4},
            {'conv_count': 2, 'filters': 64, 'activation': None, 'dropout_rate': 0.4},
            {'conv_count': 1, 'filters': 96, 'activation': None, 'dropout_rate': 0.6},
        ], dense_size=[128], dense_dropout_rate=0.5),
        ###
        VggLikeModel(height, width, depth, num_classes, conv_params=[
            {'conv_count': 2, 'filters': 16, 'activation': 'tanh', 'dropout_rate': 0.4},  # 32-16
        ], dense_size=[128], dense_dropout_rate=0.5),
        VggLikeModel(height, width, depth, num_classes, conv_params=[
            {'conv_count': 2, 'filters': 16, 'activation': 'tanh', 'dropout_rate': 0.4},  # 32-16-8
            {'conv_count': 2, 'filters': 32, 'activation': 'tanh', 'dropout_rate': 0.4},
        ], dense_size=[128], dense_dropout_rate=0.5),
        VggLikeModel(height, width, depth, num_classes, conv_params=[
            {'conv_count': 2, 'filters': 16, 'activation': 'tanh', 'dropout_rate': 0.4},  # 32-16-8-4
            {'conv_count': 2, 'filters': 32, 'activation': 'tanh', 'dropout_rate': 0.4},
            {'conv_count': 2, 'filters': 64, 'activation': 'tanh', 'dropout_rate': 0.4},
        ], dense_size=[128], dense_dropout_rate=0.5),
        VggLikeModel(height, width, depth, num_classes, conv_params=[  # 20
            {'conv_count': 2, 'filters': 16, 'activation': 'tanh', 'dropout_rate': 0.4},  # 32-16-8-4
            {'conv_count': 2, 'filters': 32, 'activation': 'tanh', 'dropout_rate': 0.4},
            {'conv_count': 2, 'filters': 64, 'activation': 'tanh', 'dropout_rate': 0.4},
        ], dense_size=[512], dense_dropout_rate=0.5),
        VggLikeModel(height, width, depth, num_classes, conv_params=[
            {'conv_count': 2, 'filters': 16, 'activation': 'tanh', 'dropout_rate': 0.4},  # 32-16-8-4-2
            {'conv_count': 2, 'filters': 32, 'activation': 'tanh', 'dropout_rate': 0.4},
            {'conv_count': 2, 'filters': 64, 'activation': 'tanh', 'dropout_rate': 0.4},
            {'conv_count': 1, 'filters': 96, 'activation': 'tanh', 'dropout_rate': 0.6},
        ], dense_size=[128], dense_dropout_rate=0.5),
        ### 20 +
        VggLikeModel(height, width, depth, num_classes, conv_params=[
            {'conv_count': 3, 'filters': 16, 'activation': 'tanh', 'dropout_rate': 0.4},  # 32-16-8-4
            {'conv_count': 3, 'filters': 32, 'activation': 'tanh', 'dropout_rate': 0.4},
            {'conv_count': 3, 'filters': 64, 'activation': 'tanh', 'dropout_rate': 0.4},
        ], dense_size=[512], dense_dropout_rate=0.5),
        VggLikeModel(height, width, depth, num_classes, conv_params=[
            {'conv_count': 2, 'filters': 32, 'activation': 'tanh', 'dropout_rate': 0.4},  # 32-16-8-4
            {'conv_count': 2, 'filters': 64, 'activation': 'tanh', 'dropout_rate': 0.4},
            {'conv_count': 2, 'filters': 128, 'activation': 'tanh', 'dropout_rate': 0.4},
        ], dense_size=[512], dense_dropout_rate=0.5),
        VggLikeModel(height, width, depth, num_classes, conv_params=[
            {'conv_count': 2, 'filters': 16, 'activation': 'relu', 'dropout_rate': 0.4},  # 32-16-8-4
            {'conv_count': 2, 'filters': 32, 'activation': 'relu', 'dropout_rate': 0.4},
            {'conv_count': 2, 'filters': 64, 'activation': 'relu', 'dropout_rate': 0.4},
        ], dense_size=[512], dense_dropout_rate=0.5),
        VggLikeModel(height, width, depth, num_classes, conv_params=[
            {'conv_count': 2, 'filters': 16, 'activation': 'tanh', 'dropout_rate': 0.7},  # 32-16-8-4
            {'conv_count': 2, 'filters': 32, 'activation': 'tanh', 'dropout_rate': 0.7},
            {'conv_count': 2, 'filters': 64, 'activation': 'tanh', 'dropout_rate': 0.7},
        ], dense_size=[512], dense_dropout_rate=0.5),
        VggLikeModel(height, width, depth, num_classes, conv_params=[
            {'conv_count': 2, 'filters': 16, 'activation': 'tanh', 'dropout_rate': 0.4},  # 32-16-8-4
            {'conv_count': 2, 'filters': 32, 'activation': 'tanh', 'dropout_rate': 0.4},
            {'conv_count': 2, 'filters': 64, 'activation': 'tanh', 'dropout_rate': 0.4},
        ], dense_size=[512, 128], dense_dropout_rate=0.5),
        VggLikeModel(height, width, depth, num_classes, conv_params=[
            {'conv_count': 2, 'filters': 16, 'activation': 'tanh', 'dropout_rate': 0.4},  # 32-16-8-4
            {'conv_count': 2, 'filters': 32, 'activation': 'tanh', 'dropout_rate': 0.4},
            {'conv_count': 2, 'filters': 64, 'activation': 'tanh', 'dropout_rate': 0.4},
        ], dense_size=[512+256], dense_dropout_rate=0.5),

        ### 20 + 'kernel_regularizer': regularizers.l2(0.1)
        VggLikeModel(height, width, depth, num_classes, conv_params=[  # 20 + kernel_regularizer
            {'conv_count': 2, 'filters': 16, 'activation': 'tanh', 'dropout_rate': 0.4, 'kernel_regularizer': regularizers.l2(0.001)},  # 32-16-8-4
            {'conv_count': 2, 'filters': 32, 'activation': 'tanh', 'dropout_rate': 0.4, 'kernel_regularizer': regularizers.l2(0.001)},
            {'conv_count': 2, 'filters': 64, 'activation': 'tanh', 'dropout_rate': 0.4, 'kernel_regularizer': regularizers.l2(0.001)},
        ], dense_size=[512], dense_dropout_rate=0.5),
        VggLikeModel(height, width, depth, num_classes, conv_params=[
            {'conv_count': 3, 'filters': 16, 'activation': 'tanh', 'dropout_rate': 0.4, 'kernel_regularizer': regularizers.l2(0.001)},  # 32-16-8-4
            {'conv_count': 3, 'filters': 32, 'activation': 'tanh', 'dropout_rate': 0.4, 'kernel_regularizer': regularizers.l2(0.001)},
            {'conv_count': 3, 'filters': 64, 'activation': 'tanh', 'dropout_rate': 0.4, 'kernel_regularizer': regularizers.l2(0.001)},
        ], dense_size=[512], dense_dropout_rate=0.5),
        VggLikeModel(height, width, depth, num_classes, conv_params=[
            {'conv_count': 2, 'filters': 32, 'activation': 'tanh', 'dropout_rate': 0.4, 'kernel_regularizer': regularizers.l2(0.001)},  # 32-16-8-4
            {'conv_count': 2, 'filters': 64, 'activation': 'tanh', 'dropout_rate': 0.4, 'kernel_regularizer': regularizers.l2(0.001)},
            {'conv_count': 2, 'filters': 128, 'activation': 'tanh', 'dropout_rate': 0.4, 'kernel_regularizer': regularizers.l2(0.001)},
        ], dense_size=[512], dense_dropout_rate=0.5),
        VggLikeModel(height, width, depth, num_classes, conv_params=[
            {'conv_count': 2, 'filters': 16, 'activation': 'relu', 'dropout_rate': 0.4, 'kernel_regularizer': regularizers.l2(0.001)},  # 32-16-8-4
            {'conv_count': 2, 'filters': 32, 'activation': 'relu', 'dropout_rate': 0.4, 'kernel_regularizer': regularizers.l2(0.001)},
            {'conv_count': 2, 'filters': 64, 'activation': 'relu', 'dropout_rate': 0.4, 'kernel_regularizer': regularizers.l2(0.001)},
        ], dense_size=[512], dense_dropout_rate=0.5),
        VggLikeModel(height, width, depth, num_classes, conv_params=[
            {'conv_count': 2, 'filters': 16, 'activation': 'tanh', 'dropout_rate': 0.7, 'kernel_regularizer': regularizers.l2(0.001)},  # 32-16-8-4
            {'conv_count': 2, 'filters': 32, 'activation': 'tanh', 'dropout_rate': 0.7, 'kernel_regularizer': regularizers.l2(0.001)},
            {'conv_count': 2, 'filters': 64, 'activation': 'tanh', 'dropout_rate': 0.7, 'kernel_regularizer': regularizers.l2(0.001)},
        ], dense_size=[512], dense_dropout_rate=0.5),
        VggLikeModel(height, width, depth, num_classes, conv_params=[
            {'conv_count': 2, 'filters': 16, 'activation': 'tanh', 'dropout_rate': 0.4, 'kernel_regularizer': regularizers.l2(0.001)},  # 32-16-8-4
            {'conv_count': 2, 'filters': 32, 'activation': 'tanh', 'dropout_rate': 0.4, 'kernel_regularizer': regularizers.l2(0.001)},
            {'conv_count': 2, 'filters': 64, 'activation': 'tanh', 'dropout_rate': 0.4, 'kernel_regularizer': regularizers.l2(0.001)},
        ], dense_size=[512, 128], dense_dropout_rate=0.5),
        VggLikeModel(height, width, depth, num_classes, conv_params=[
            {'conv_count': 2, 'filters': 16, 'activation': 'tanh', 'dropout_rate': 0.4, 'kernel_regularizer': regularizers.l2(0.001)},  # 32-16-8-4
            {'conv_count': 2, 'filters': 32, 'activation': 'tanh', 'dropout_rate': 0.4, 'kernel_regularizer': regularizers.l2(0.001)},
            {'conv_count': 2, 'filters': 64, 'activation': 'tanh', 'dropout_rate': 0.4, 'kernel_regularizer': regularizers.l2(0.001)},
        ], dense_size=[512+256], dense_dropout_rate=0.5),
        Vgg16(height, width, depth, num_classes),
    ]


if __name__ == '__main__':
    models = get_all_models(32, 32, 3, 10)
    for i, model in enumerate(models):
        print(i, model.name)
    print('test ok')
