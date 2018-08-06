"""Разные модели
"""
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.models import Model, Sequential # basic class for specifying and training a neural network


class BaseModel(Model):
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
    def __init__(self, height, width, depth, num_classes, hidden_size):
        inp_layers = [Input(shape=(height, width, depth))]
        out_layers = [
            Flatten(),
            Dense(hidden_size, activation='relu'),
            Dense(num_classes, activation='softmax')
        ]

        super().__init__(inp_layers + out_layers)
        self.compile(loss='categorical_crossentropy',  # using the cross-entropy loss function
                     optimizer='adam',  # using the Adam optimiser
                     metrics=['accuracy'])  # reporting the accuracy
