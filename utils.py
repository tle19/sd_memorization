import numpy as np
from PIL import Image
import argparse


def euclidean_func(x, y):
    image1 = Image.open(x)
    image2 = Image.open(y)

    vec1 = np.array(image1).flatten()
    vec2 = np.array(image2).flatten()

    distance = np.linalg.norm(vec1 - vec2)
    print(distance)

    return distance

def manhattan_func(x, y):
    image1 = Image.open(x)
    image2 = Image.open(y)

    vec1 = np.array(image1).flatten()
    vec2 = np.array(image2).flatten()

    distance = np.linalg.norm(vec1 - vec2)
    print(distance)

    return distance