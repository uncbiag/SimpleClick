from isegm.data.datasets.PASCAL import PASCAL
import cv2
import matplotlib.pyplot as plt

def show_sample(sample):
    plt.imshow(sample.image)
    plt.show()
    plt.imshow(sample._encoded_masks)
    plt.show()
    print("done")

dataset = PASCAL(dataset_path="/home/gyt/gyt/dataset/data/pascal_person_part", split='train')

a_sample = dataset.get_sample(0)
show_sample(a_sample)

