from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.regularizers import l2

class AlexNet:

    @staticmethod
    def build(height, width, channels, classes, reg=0.0002):

        input_shape = (height, width, channels)
        channel_dim = -1
        if K.image_data_format() == "channels_first":
            input_shape = (channels, height, width)
            channel_dim = 1

        model = Sequential()

        model.add(Conv2D(96, (11, 11), strides=(4, 4), padding="same", input_shape=input_shape, kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (5, 5), padding="same", kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(384, (3, 3), padding="same", kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(Conv2D(384, (3, 3), padding="same", kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(Conv2D(256, (3, 3), padding="same", kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())

        model.add(Dense(4096, kernel_regularizer=l2(reg), activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(4096, kernel_regularizer=l2(reg), activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(classes, kernel_regularizer=l2(reg), activation="softmax"))
        return model
