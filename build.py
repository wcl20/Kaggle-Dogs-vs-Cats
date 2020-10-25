import cv2
import json
import numpy as np
import os
import tqdm
from config import config
from imutils import paths
from core.io import HDF5Writer
from core.preprocessing import ResizeWithAspectRatio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def main():

    img_paths = list(paths.list_images(config.IMAGES_PATH))

    label_encoder = LabelEncoder()
    labels = [os.path.basename(img_path).split(".")[0] for img_path in img_paths]
    labels = label_encoder.fit_transform(labels)

    train_paths, test_paths, y_train, y_test = train_test_split(img_paths, labels, test_size=config.NUM_TEST_IMAGES, stratify=labels, random_state=42)
    train_paths, valid_paths, y_train, y_valid = train_test_split(train_paths, y_train, test_size=config.NUM_VALID_IMAGES, stratify=y_train, random_state=42)

    datasets = [
        ("train", train_paths, y_train, config.TRAIN_PATH),
        ("valid", valid_paths, y_valid, config.VALID_PATH),
        ("test", test_paths, y_test, config.TEST_PATH),
    ]

    preprocessor = ResizeWithAspectRatio(256, 256)
    R, G, B = [], [], []

    for type, img_paths, labels, output_path in datasets:
        print(f"[INFO] Building {output_path} ...")
        print(f"[INFO] Number of images: {len(img_paths)}. Labels: {labels.shape}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        dataset = HDF5Writer(output_path, (len(img_paths), 256, 256, 3))
        for img_path, label in tqdm.tqdm(zip(img_paths, labels)):
            image = cv2.imread(img_path)
            image = preprocessor.preprocess(image)
            if type == "train":
                # Get mean value of each channel
                b, g, r = cv2.mean(image)[:3]
                B.append(b)
                G.append(g)
                R.append(r)
            dataset.add([image], [label])
    dataset.close()

    print(f"[INFO] Saving means to {config.MEAN_PATH}")
    os.makedirs(config.OUTPUT_PATH, exist_ok=True)
    file = open(config.MEAN_PATH, "w")
    file.write(json.dumps({ "R": np.mean(R), "G": np.mean(G), "B": np.mean(B) }))
    file.close()

if __name__ == '__main__':
    main()
