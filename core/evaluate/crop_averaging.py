import cv2
import numpy as np

class CropAveraging:

    def __init__(self, height, width, flip=True, interpolation=cv2.INTER_AREA):
        self.height = height
        self.width = width
        self.flip = flip
        self.interpolation = interpolation

    def preprocess(self, image):

        img_height, img_width = image.shape[:2]
        coords = [
            # Top left corner
            [0, 0, self.width, self.height],
            # Top right corner
            [img_width - self.width, 0, img_width, self.height],
            # Bottom right corner
            [img_width - self.width, img_height - self.height, img_width, img_height],
            # Bottom left corner
            [0, img_height - self.height, self.width, img_height]
        ]

        offset_w = int(0.5 * (img_width - self.width))
        offset_h = int(0.5 * (img_height - self.height))
        coords.append([offset_w, offset_h, img_width - offset_w, img_height - offset_h])

        images = []
        for x1, y1, x2, y2 in coords:
            roi = image[y1:y2, x1:x2]
            roi = cv2.resize(roi, (self.width, self.height), interpolation=self.interpolation)
            images.append(roi)

        if self.flip:
            # Horizontal Flip
            images.extend([cv2.flip(roi, 1) for roi in images])

        return np.array(images)
