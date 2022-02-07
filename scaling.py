import cv2
import numpy as np
import torch

class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, common_size=1024):
        self.common_size = common_size;

    def __call__(self, sample, common_size=1024):
        common_size = self.common_size;
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = common_size / height
            resized_height = common_size
            resized_width = int(width * scale)
        else:
            scale = common_size / width
            resized_height = int(height * scale)
            resized_width = common_size

        image = cv2.resize(image, (resized_width, resized_height))
        #print("W * H :", resized_width, " * ", resized_height)
        new_image = np.zeros((common_size, common_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}