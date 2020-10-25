import json
import matplotlib
matplotlib.use("Agg")
import os
from config import config
from core.callbacks import TrainingMonitor
from core.io import HDF5Reader
from core.nn import AlexNet
from core.preprocessing import MeanSubtraction
from core.preprocessing import RandomCrop
from core.preprocessing import Resize
from core.preprocessing import ToArray
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def main():

    # Define preprocessors
    resize = Resize(227, 227)
    random_crop = RandomCrop(227, 227)
    means = json.loads(open(config.MEAN_PATH).read())
    mean_subtraction = MeanSubtraction(means["R"], means["G"], means["B"])

    # Define image augmentation
    augmentation = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    # Data generator from HDF5 files
    train_gen = HDF5Reader(config.TRAIN_PATH,
        batch_size=128,
        preprocessors=[random_crop, mean_subtraction, ToArray()],
        augmentation=augmentation,
    )

    valid_gen = HDF5Reader(config.VALID_PATH,
        batch_size=128,
        preprocessors=[resize, mean_subtraction, ToArray()],
    )

    # Training monitor
    fig_path = os.path.sep.join([config.OUTPUT_PATH, f"{os.getpid()}.png"])
    callbacks = [TrainingMonitor(fig_path)]

    # Compile model
    model = AlexNet.build(227, 227, 3, classes=2)
    optimizer = Adam(lr=1e-3)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=["accuracy"])

    # Train model
    model.fit_generator(
        train_gen.generator(),
        epochs=75,
        max_queue_size=10,
        steps_per_epoch=train_gen.num_images // 128,
        validation_data=valid_gen.generator(),
        validation_steps=valid_gen.num_images // 128,
        callbacks=callbacks,
        verbose=1
    )

    model.save(config.MODEL_PATH, overwrite=True)

    train_gen.close()
    valid_gen.close()


if __name__ == '__main__':
    main()
