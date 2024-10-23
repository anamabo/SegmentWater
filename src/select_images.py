"""
IMPORTANT: Run this script in debug mode to visualize the contours of each mask.
This script is used to MANUALLY select the images used in fine-tuning Paligemma.
"""


import logging
import os
import random

import cv2
import numpy as np
from matplotlib import pyplot as plt

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
random.seed(123)


def get_contours_coordinates(ccontours) -> dict:
    reshaped_cnts = [cnt.reshape(len(cnt), 2) for cnt in ccontours]

    contours_coords = dict()
    for n, contour in enumerate(reshaped_cnts):
        flatten_cnt = contour.flatten()
        xvals = [
            flatten_cnt[x] for x in range(0, len(flatten_cnt), 2)
        ]  # even=x
        yvals = [
            flatten_cnt[y] for y in range(1, len(flatten_cnt), 2)
        ]  # odd=y
        contours_coords[n] = (xvals, yvals)
    return contours_coords


def plot_image_and_contours(image_arr, mask_arr, contour):
    cnt_dict = get_contours_coordinates(contour)
    fig, ax = plt.subplots(1, 2, figsize=(5, 5))
    ax[0].imshow(image_arr)
    ax[1].imshow(mask_arr)
    for _, (x, y) in cnt_dict.items():
        ax[1].plot(x, y, "r-")
    plt.show()


if __name__ == "__main__":
    data_path = "./data"  # Path to the data folder. (Change if needed).
    masks_folder_name = (
        "Masks_cleaned"  # Folder containing the masks. (Change if needed).
    )
    images_folder_name = (
        "Images_cleaned"  # folder containing the images. (Change if needed).
    )

    # Code
    mask_path = os.path.join(data_path, masks_folder_name)
    image_path = os.path.join(data_path, images_folder_name)
    threshold = 150

    with open(os.path.join(data_path, "selected_images.txt"), "r") as file:
        masks_names = file.read().splitlines()

    for mask_name in masks_names:
        print("mask_name:", mask_name)
        image = cv2.imread(os.path.join(image_path, mask_name))
        mask = cv2.imread(os.path.join(mask_path, mask_name))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        if np.unique(mask).shape[0] == 1 and np.unique(mask)[0] == 0:
            print("image", mask_name, "has no water")

        else:
            # make the mask binary
            _, mask_binary = cv2.threshold(
                mask, thresh=threshold, maxval=255, type=cv2.THRESH_BINARY
            )

            # Get the contours of the mask
            # tuple(ndarray(cnt points, 1, 2),...)
            contours, _ = cv2.findContours(
                mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Reduce the number of points in the contours
            # This is needed to reduce the tokens  for Paligemma
            approximated_contours = tuple()
            for cnt in contours:
                perimeter = cv2.arcLength(cnt, closed=True)
                approx = cv2.approxPolyDP(cnt, 0.001 * perimeter, closed=True)
                approximated_contours += (approx,)

            plot_image_and_contours(image, mask, approximated_contours)
            print("done!")
