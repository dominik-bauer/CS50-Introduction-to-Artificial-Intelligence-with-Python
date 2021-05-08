import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

# Some of my own imports
from collections import Counter
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def get_crop_tuple(n_crop_total):
    if n_crop_total < 0:
        raise ValueError
    a = b = n_crop_total // 2
    b = b + n_crop_total % 2
    return a, b


def center_crop_to_square(input_array):

    w, h, _ = input_array.shape
    crop_top = crop_bottom = crop_left = crop_right = 0

    if w < h:
        d = h - w
        crop_top, crop_bottom = get_crop_tuple(d)
    elif w > h:
        d = w - h
        crop_left, crop_right = get_crop_tuple(d)

    return input_array[crop_left:w-crop_right, crop_top:h-crop_bottom]


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    files = []
    images = []
    labels = []

    # remove trailing back/slashes
    if data_dir[-1] in ["\\", "/"]:
        data_dir = data_dir[:-1]

    # Loop all subdirectories
    for sd in [f for f in os.scandir(data_dir) if f.is_dir()]:

        # check if the current directory is a number
        if not sd.name.isdigit():
            continue

        # get all files in subdirectory and save paths and label in lists
        for fn in [f.path for f in os.scandir(sd) if f.is_file()]:
            files.append(fn)
            labels.append(int(sd.name))

    # Print statistics
    print("\nFound a total of {} files and {} labels.\n".format(len(files), len(set(labels))))
    print("Found {} images within {}".format(len(labels), data_dir))
    for lab in sorted(set(labels)):
        print("Label {}: {} images".format(lab, Counter(labels)[lab]))
    print()

    # Start loading images as per specification
    n_total = len(files)
    for n, (filepath, label) in enumerate(zip(files, labels), 1):

        # Open the picture in CV
        img = cv2.imread(filepath)

        # possible variant is to crop to square first in order to avoid skewed pictures
        # img = center_crop_to_square(img)

        # Finally resize to given dimensions
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

        # And add image to designated list
        if not img.shape == (IMG_WIDTH, IMG_HEIGHT, 3):
            raise Exception("The Image is of incorrect shape")
        images.append(img)

        # Checking the progress
        if (not n == 0 and not n % 1000) or n == n_total:
            print("{} images processed ({:.1%})".format(n, n/n_total))

    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    shp = IMG_WIDTH, IMG_HEIGHT, 3

    model = tf.keras.Sequential([
        Conv2D(filters=20, kernel_size=(3, 3), input_shape=shp, activation="tanh"),
        MaxPooling2D(pool_size=2),
        Conv2D(filters=16, kernel_size=(3, 3), activation="tanh"),
        MaxPooling2D(pool_size=2),
        Flatten(),
        Dense(units=100, activation="sigmoid"),
        Dropout(rate=0.5),
        Dense(units=43, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()
