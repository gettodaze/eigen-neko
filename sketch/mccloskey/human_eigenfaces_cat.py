# From https://towardsdatascience.com/eigenfaces-recovering-humans-from-ghosts-17606c328184
# %%
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import typing as tp
from core import utils
import typing as tp
import math
import itertools as it
from typing_extensions import Annotated
from PIL import Image, ImageOps
import numpy as np
from matplotlib import pyplot as plt
from sklearn import decomposition

from core import utils

N_SAMPLES = 100
# %%
# """It helps visualising the portraits from the dataset."""
def plot_portraits(images, titles, shape, n_row, n_col):
    plt.figure(figsize=(2.2 * n_col, 2.2 * n_row))
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.20)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape(shape))
        plt.title(titles[i])
        plt.xticks(())
        plt.yticks(())


def gen_image_arrays(files: tp.Iterable[utils.AnnotatedImage], output_shape=(64, 64)):
    for file in files:
        image = Image.open(file.image)
        grayscale = ImageOps.grayscale(image)
        min_pt, max_pt = utils.Point.to_min_max(file.points)

        yield np.array(
            image.crop((min_pt.x, min_pt.y, max_pt.x, max_pt.y)).resize(output_shape)
        )


files = list(utils.Paths.gen_files())
names = [f.image.stem for f in files[:N_SAMPLES]]
cropped_images = list(gen_image_arrays(files[:N_SAMPLES]))
shape = cropped_images[0].shape
X_train = np.array([im.flatten() for im in cropped_images])
plot_portraits(X_train, names, shape, n_row=4, n_col=4)

# %%
def pca(X, n_pc):
    n_samples, n_features = X.shape
    mean = np.mean(X, axis=0)
    centered_data = X - mean
    U, S, V = np.linalg.svd(centered_data)
    components = V[:n_pc]
    projected = U[:, :n_pc] * S[:n_pc]

    return projected, components, mean, centered_data


def expand_components(arr: np.ndarray):
    ret = arr
    ret -= ret.min()
    ret /= ret.max()
    return ret


n_components = 50
P, C, M, Y = pca(X_train, n_pc=n_components)
eigenfaces = np.array([expand_components(arr) for arr in C])
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_portraits(eigenfaces, eigenface_titles, shape, 4, 4)

# %%
def reconstruction(Y, C, M, shape, image_index):
    n_samples, n_features = Y.shape
    weights = np.dot(Y, C.T)
    centered_vector = np.dot(weights[image_index, :], C)
    recovered_image = (M + centered_vector).reshape(shape)
    return recovered_image


recovered_images = [reconstruction(Y, C, M, shape, i) for i in range(N_SAMPLES)]
plot_portraits(recovered_images, names, shape, n_row=4, n_col=4)

# %%
