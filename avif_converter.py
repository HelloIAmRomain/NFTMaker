# Program that converts AVIF to CSV files in a folder

import os
import sys
import csv
from PIL import Image
import pillow_avif
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def convert_to_csv(folder_path, output_path):
    """
    Converts all AVIF files in a csv file
    each row is an image
    """
    list_images = os.listdir(folder_path)
    list_pixels = []
    for image in tqdm(list_images):
        if image.endswith(".avif"):
            img = Image.open(folder_path + image)
            img = img.convert('RGB')
            # Convert to numpy array
            img = np.array(img)
            # Convert to 1D array
            img = img.flatten()
            list_pixels.append(img)
    # Convert to numpy array
    list_pixels = np.array(list_pixels)
    shape = list_pixels.shape
    # Save array
    np.savetxt(output_path, list_pixels, delimiter=",")
    print("Saved to csv")
    return shape


def convert_to_png(folder_path, output_path):
    """
    Converts all AVIF files in a folder to PNG files
    """
    list_images = os.listdir(folder_path)
    for image in tqdm(list_images):
        if image.endswith(".avif"):
            img = Image.open(folder_path + image)
            img = img.convert('RGB')
            img.save(output_path + image[:-5] + ".png")
    print("Saved to png")


def show_image(avif_file):
    """Shows an AVIF image"""
    print("Showing image")
    img = Image.open(avif_file)
    img = img.convert('RGB')
    print("Image size: ", img.size)
    plt.imshow(img)
    plt.show()


def show_image_from_csv(csv_file, image_size: tuple, image_number: int):
    """Shows an image from a csv file 
    dimension is n_images x image_size.product()
    """
    print("Showing image from csv")
    img = np.loadtxt(csv_file, delimiter=",")
    n_images = img.shape[0]
    img_list = img.reshape(n_images, image_size[0], image_size[1], image_size[2])
    plt.imshow(img_list[image_number])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", '-p', help="Path to the folder with the avif files", default="bored_ape_yacht_club_nfts_on_rarible__buy__sell_and_trade___rarible/")
    parser.add_argument("--output_path", '-o', help="Path to the output csv file", default="png_images/")
    args = parser.parse_args()
    shape = convert_to_png(args.folder_path, args.output_path)
    print("Image converted to png")
    print("Shape: ", shape)
    print("Done")
    sys.exit(0)
