from keras import Model, Input, Sequential, regularizers
from keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D, Activation, BatchNormalization
from keras import optimizers

from model_bag.core import auto_naming


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
    def get_conv(inp_layer, conv_count=2, filters=16, kernel_size=3, activation=None, dropout_rate=0.25,
                 kernel_regularizer=None):
        out_layer = inp_layer
        for _ in range(conv_count):
            out_layer = Conv2D(filters=filters, kernel_size=kernel_size, padding='same',
                               kernel_regularizer=kernel_regularizer)(out_layer)
        out_layer = MaxPooling2D(padding='same')(out_layer)
        if activation:
            out_layer = Activation(activation)(out_layer)
        out_layer = Dropout(rate=dropout_rate)(out_layer)
        return out_layer


class Vgg16(Sequential):
    @auto_naming
    def __init__(self, height, width, depth, num_classes):
        super().__init__()

        weight_decay = 0.0005

        self.add(Conv2D(64, (3, 3), padding='same',
                        input_shape=(height, width, depth), kernel_regularizer=regularizers.l2(weight_decay)))
        self.add(Activation('relu'))
        self.add(BatchNormalization())
        self.add(Dropout(0.3))

        self.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.add(Activation('relu'))
        self.add(BatchNormalization())

        self.add(MaxPooling2D(pool_size=(2, 2)))

        self.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.add(Activation('relu'))
        self.add(BatchNormalization())
        self.add(Dropout(0.4))

        self.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.add(Activation('relu'))
        self.add(BatchNormalization())

        self.add(MaxPooling2D(pool_size=(2, 2)))

        self.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.add(Activation('relu'))
        self.add(BatchNormalization())
        self.add(Dropout(0.4))

        self.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.add(Activation('relu'))
        self.add(BatchNormalization())
        self.add(Dropout(0.4))

        self.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.add(Activation('relu'))
        self.add(BatchNormalization())

        self.add(MaxPooling2D(pool_size=(2, 2)))

        self.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.add(Activation('relu'))
        self.add(BatchNormalization())
        self.add(Dropout(0.4))

        self.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.add(Activation('relu'))
        self.add(BatchNormalization())
        self.add(Dropout(0.4))

        self.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.add(Activation('relu'))
        self.add(BatchNormalization())

        self.add(MaxPooling2D(pool_size=(2, 2)))

        self.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.add(Activation('relu'))
        self.add(BatchNormalization())
        self.add(Dropout(0.4))

        self.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.add(Activation('relu'))
        self.add(BatchNormalization())
        self.add(Dropout(0.4))

        self.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.add(Activation('relu'))
        self.add(BatchNormalization())

        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Dropout(0.5))

        self.add(Flatten())
        self.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
        self.add(Activation('relu'))
        self.add(BatchNormalization())

        self.add(Dropout(0.5))
        self.add(Dense(num_classes))
        self.add(Activation('softmax'))

        lr_decay = 1e-6
        learning_rate = 0.1
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        self.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

