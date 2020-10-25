from sklearn.feature_extraction.image import extract_patches_2d

class RandomCrop:

    def __init__(self, height, width):
        self.dim = (height, width)

    def preprocess(self, image):
        return extract_patches_2d(image, self.dim, max_patches=1)[0]
