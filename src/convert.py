import glob
import json
import logging
import os
import random
import shutil

import click
import cv2
import numpy as np

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
random.seed(123)


def get_padded_bbox(
    contour, image_height: int, image_width: int, pad: int = 4
):
    x1, y1, w, h = cv2.boundingRect(contour)
    x2, y2 = x1 + w, y1 + h
    new_bbox = np.array(
        [
            y1 / image_height,
            x1 / image_width,
            y2 / image_height,
            x2 / image_width,
        ]
    )
    new_bbox *= 1024  # needed for paliGemma
    new_bbox = new_bbox.astype(int)
    # pad with zeros to the left
    paligemma_bbox = np.char.zfill(new_bbox.astype(str), width=pad)
    return paligemma_bbox


def format_padded_bbox(bbox):
    return "".join([f"<loc{element}>" for element in bbox])


def get_padded_contour(contour, image_height: int, image_width: int):
    npoints = contour.shape[0]
    reshaped_contour = contour.reshape(npoints, 2)

    # The multiplication is needed for Paligemma
    # For segmentation, we need coords = y,x
    new_cnt = [
        (coords[1] / image_height * 1024, coords[0] / image_width * 1024)
        for coords in reshaped_contour
    ]
    new_cnt = np.array(new_cnt)
    new_cnt = new_cnt.astype(int).flatten()
    paligemma_cnt = np.char.zfill(new_cnt.astype(str), width=3)
    return paligemma_cnt


def format_padded_contour(contour):
    return "".join([f"<seg{element}>" for element in contour])


def create_output_for_paligemma(
    mask_path: str, mask_name: str, threshold: int, cclass: str, prefix: str
) -> dict:
    """Given an image, it creates a dict with the output for paligemma.
    IMPORTANT: This function assumes the same filename for both images and masks."""

    mask = cv2.imread(os.path.join(mask_path, mask_name))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    im_height, im_width = mask.shape

    if np.unique(mask).shape[0] == 1 and np.unique(mask)[0] == 0:
        # If the mask has no water, return an empty suffix
        final_output = {"image": mask_name, "prefix": prefix, "suffix": " "}

    else:
        # make the mask binary
        _, mask_binary = cv2.threshold(
            mask, thresh=threshold, maxval=255, type=cv2.THRESH_BINARY
        )

        # Get the contours of the mask
        contours, _ = cv2.findContours(
            mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # get the output for paligemma
        paligemma_output = []
        for counter, contour in enumerate(contours):
            padded_bbox = get_padded_bbox(
                contour, image_height=im_height, image_width=im_width
            )
            line_bbox = format_padded_bbox(padded_bbox)

            padded_contour = get_padded_contour(
                contour, image_height=im_height, image_width=im_width
            )
            line_contour = format_padded_contour(padded_contour)

            paligemma_output.append(
                line_bbox + " " + line_contour + " " + cclass
            )

        paligemma_output = "; ".join(paligemma_output)
        final_output = {
            "image": mask_name,
            "prefix": prefix,
            "suffix": paligemma_output,
        }

    return final_output


@click.command()
@click.option(
    "--data_path",
    required=True,
    type=str,
    help="The absolute path to the data folder.",
)
@click.option(
    "--masks_folder_name",
    required=True,
    type=str,
    help="The name of the folder with the corrected masks.",
)
@click.option(
    "--images_folder_name",
    required=True,
    type=str,
    help="The name of the folder with the corrected images.",
)
@click.option(
    "--output_folder_name",
    default="water_bodies",
    type=str,
    help="The name of the folder with the output for Paligemma.",
)
@click.option(
    "--threshold",
    default=150,
    type=int,
    help="Threshold for the binary mask. Values larger then this will be tagged as water (255, which is white)",
)
@click.option(
    "--prefix",
    default="Segment water",
    type=str,
    help="The prefix field in the output for Paligemma.",
)
@click.option(
    "--class_in_file",
    default="water",
    type=str,
    help="The class to be segmented.",
)
@click.option(
    "--train_fraction",
    default=0.85,
    type=float,
    help="The fraction of the data to be used for training.",
)
def main(
    data_path,
    masks_folder_name,
    images_folder_name,
    output_folder_name,
    threshold,
    prefix,
    class_in_file,
    train_fraction,
):
    # Code
    mask_path = os.path.join(data_path, masks_folder_name)
    image_path = os.path.join(data_path, images_folder_name)
    output_path = os.path.join(data_path, output_folder_name)

    os.makedirs(output_path, exist_ok=True)

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

    total_files = (
        len(images_train_set)
        + len(images_validation_set)
        + len(images_test_set)
    )
    assert total_files == len(masks_names)

    # create the Paligemma output for each dataset
    dataset_names = ["train", "validation", "test"]
    dataset_images = [images_train_set, images_validation_set, images_test_set]

    for dataset, list_images in zip(dataset_names, dataset_images):
        logging.info(
            f"Copy {len(list_images)} images to the {dataset} dataset."
        )

        output_filename = dataset + ".jsonl"
        full_out_path = os.path.join(output_path, output_filename)

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

        logging.info(f"Writing the results to {full_out_path}.")
        with open(full_out_path, "w", encoding="utf-8") as file:
            for item in paligemma_list:
                json.dump(item, file)
                file.write("\n")

    # finally, copy the images to the output folder
    shutil.copytree(image_path, output_path, dirs_exist_ok=True)
    logging.info("Done!")


if __name__ == "__main__":
    main()
