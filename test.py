import json
import numpy as np
import tqdm
from config import config
from core.evaluate import CropAveraging
from core.io import HDF5Reader
from core.preprocessing import MeanSubtraction
from core.preprocessing import ToArray
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model

def main():


    # Load model
    model = load_model(config.MODEL_PATH)

    means = json.loads(open(config.MEAN_PATH).read())
    mean_subtraction = MeanSubtraction(means["R"], means["G"], means["B"])
    test_gen = HDF5Reader(config.TEST_PATH, batch_size=64, preprocessors=[mean_subtraction])

    sampler = CropAveraging(227, 227)
    predictions = []
    for batch_images, batch_labels in tqdm.tqdm(test_gen.generator(epochs=1)):
        for image in batch_images:
            # Apply crop averaging to increase model accuracy
            samples = sampler.preprocess(image)
            samples = np.array([ToArray().preprocess(sample) for sample in samples], dtype="float32")
            # Make predictions to the samples and take average
            preds = model.predict(samples)
            predictions.append(preds.mean(axis=0))
    predictions = np.array(predictions)
    report = classification_report(test_gen.db["labels"], predictions.argmax(axis=1), target_names=["cat", "dog"])
    print(report)


if __name__ == '__main__':
    main()
