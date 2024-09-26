import cv2
import glob
import json
import os
import numpy as np
import logging
import shutil
import random


logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
random.seed(123)


def get_padded_bbox(contour, image_height: int, image_width: int, pad: int = 4):
    x1, y1, w, h = cv2.boundingRect(contour)
    x2, y2 = x1 + w, y1 + h
    new_bbox = np.array([y1 / image_height, x1 / image_width, y2 / image_height, x2 / image_width])
    new_bbox *= 1024  # needed for paliGemma
    new_bbox = new_bbox.astype(int)
    paligemma_bbox = np.char.zfill(new_bbox.astype(str), width=pad)  # pad with zeros to the left
    return paligemma_bbox


def format_padded_bbox(bbox):
    return ''.join([f'<loc{element}>' for element in bbox])


def get_padded_contour(contour, image_height: int, image_width: int):
    npoints = contour.shape[0]
    reshaped_contour = contour.reshape(npoints, 2)

    # The multiplication is needed for Paligemma
    new_cnt = [(coords[1] / image_height * 1024, coords[0] / image_width * 1024) for coords in reshaped_contour]
    new_cnt = np.array(new_cnt)
    new_cnt = new_cnt.astype(int).flatten()
    paligemma_cnt = np.char.zfill(new_cnt.astype(str), width=3)
    return paligemma_cnt


def format_padded_contour(contour):
    return ''.join([f'<seg{element}>' for element in contour])


def create_output_for_paligemma(
        mask_path: str, mask_name: str, threshold: int, cclass: str, prefix: str
) -> dict:
    """ Given an image, it creates a dict with the output for paligemma.
     IMPORTANT: This function assumes the same filename for both images and masks."""

    mask = cv2.imread(os.path.join(mask_path, mask_name))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    im_height, im_width = mask.shape

    if np.unique(mask).shape[0] == 1 and np.unique(mask)[0] == 0:
        # If the mask has no water, return an empty suffix
        final_output = {"image": mask_name, "prefix": prefix, "suffix": " "}

    else:
        # make the mask binary
        _, mask_binary = cv2.threshold(mask, thresh=threshold, maxval=255, type=cv2.THRESH_BINARY)

        # Get the contours of the mask
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # get the output for paligemma
        paligemma_output = []
        for counter, contour in enumerate(contours):
            padded_bbox = get_padded_bbox(contour, image_height=im_height, image_width=im_width)
            line_bbox = format_padded_bbox(padded_bbox)

            padded_contour = get_padded_contour(contour, image_height=im_height, image_width=im_width)
            line_contour = format_padded_contour(padded_contour)

            paligemma_output.append(line_bbox + " " + line_contour + " " + cclass)

        paligemma_output = "; ".join(paligemma_output)
        final_output = {"image": mask_name, "prefix": prefix, "suffix": paligemma_output}

    return final_output


if __name__ == "__main__":
    # parameters
    data_path = "./data"
    masks_folder_name = "Masks_cleaned"
    images_folder_name = "Images_cleaned"
    output_folder_name = "water_bodies"
    threshold = 150
    prefix = "Segment water"
    class_in_file = "water"
    train_fraction = 0.85

    # Code
    mask_path = os.path.join(data_path, masks_folder_name)
    image_path = os.path.join(data_path, images_folder_name)
    output_path = os.path.join(data_path, output_folder_name)

    masks = glob.glob(os.path.join(mask_path, "*jpg"))
    masks_names = [name.split("/")[-1] for name in masks]

    test_fraction = 0.05  # just a small fraction for testing
    validation_fraction = 1 - train_fraction - test_fraction

    nsamples = len(masks_names)
    train_nsamples = int(nsamples * train_fraction)
    validation_nsamples = int(nsamples * validation_fraction)

    # select the images for each set
    images_train_set = random.sample(masks_names, train_nsamples)
    remaining = list(set(masks_names) - set(images_train_set))
    images_validation_set = random.sample(remaining, validation_nsamples)
    images_test_set = list(set(remaining) - set(images_validation_set))

    # check that the datasets are disjoint
    assert set(images_train_set).intersection(images_validation_set) == set()
    assert set(images_validation_set).intersection(images_test_set) == set()

    assert len(images_train_set) + len(images_validation_set) + len(images_test_set) == len(masks_names)

    # create the Paligemma output for each dataset
    dataset_names = ["train", "validation", "test"]
    dataset_images = [images_train_set, images_validation_set, images_test_set]

    for dataset, list_images in zip(dataset_names, dataset_images):
        logging.info(f"Analyzing {len(list_images)} images in the {dataset} dataset...")
        paligemma_list = []
        for image_name in list_images:
            output_line = create_output_for_paligemma(
                mask_path=mask_path,
                mask_name=image_name,
                threshold=threshold,
                cclass=class_in_file,
                prefix=prefix,
            )
            paligemma_list.append(output_line)

        logging.info(f"Writing the output to {os.path.join(data_path, dataset+'.json')}...")
        with open(os.path.join(data_path, dataset+'.json'), 'w', encoding='utf-8') as json_file:
            for item in paligemma_list:
                # Convert the string to JSON format and write it to the file
                json.dump(item, json_file)
                # Write a newline character
                json_file.write('\n')

    # finally, create the folder that will be used in Collab and move the images and json files
    logging.info("Creating the folder for usage in Collab and moving info there...")
    os.makedirs(output_path, exist_ok=True)
    shutil.copytree(image_path, output_path, dirs_exist_ok=True)
    for dataset in dataset_names:
        shutil.move(os.path.join(data_path, dataset + ".json"),
                    os.path.join(output_path,  dataset + ".json")
                    )

    logging.info("Done!")
