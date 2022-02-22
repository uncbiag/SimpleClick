from pysnic.algorithms.snic import snic
from pkg_resources import resource_stream
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import skimage.color
from skimage.segmentation import mark_boundaries
from pysnic.algorithms.snic import snic
import os
from PIL import Image
import sys


args = sys.argv
start, end = int(args[1]), int(args[2])

image_folder = '/playpen-raid2/qinliu/data/OAI/train/image'
images = os.listdir(image_folder)
images.sort()
images = images[start:end]

output_folder = '/playpen-raid2/qinliu/data/OAI/train/annotations'

for idx, image in enumerate(images):
    print(idx, image)

    # load image
    image_path = os.path.join(image_folder, image)
    color_image = np.array(Image.open(image_path))

    if len(color_image.shape) != 3:
        color_image = np.stack([color_image, color_image, color_image], axis=2)

    lab_image = skimage.color.rgb2lab(color_image).tolist()
    number_of_pixels = color_image.shape[0] * color_image.shape[1]

    # SNIC parameters
    target_number_of_segments = 50
    compactness = 10.00

    segmentation, _, centroids = snic(lab_image, target_number_of_segments, compactness)
    actual_number_of_segments = len(centroids)

    segmentation_npy = np.array(segmentation).astype(np.int32)
    img = Image.fromarray(segmentation_npy)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(os.path.join(output_folder, image))