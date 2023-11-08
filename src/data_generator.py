import pydicom
from PIL import Image, ImageDraw, ImageFont
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

from pdb import set_trace as pause


def data_generator(original_image_array):

    dataset_list = []
    sample_info = []

    downsamples = [2022, 1024, 512]
    no_of_downsamples = len(downsamples) # (2022, 2022) , (1011, 1011) ,(505, 505)

    for downsample_idx in range(no_of_downsamples):

        resized_image_array = cv2.resize(original_image_array, (downsamples[downsample_idx], downsamples[downsample_idx]))
        # print("Preparing DICOM image with shape: ", resized_image_array.shape)

        for text_size in [10, 20, 30]:
            for angle in [0, 45, 90, 180, 270]:

                # Add text to the current image
                image = Image.fromarray(resized_image_array)
                image = image.rotate(angle)
                draw = ImageDraw.Draw(image)
                text = "just a sample text"
                font = ImageFont.truetype("../Bembo.ttf", text_size)
                text_color = 80
                text_position = (300, 300)
                draw.text(text_position, text, fill=text_color, font=font)
                image=image.rotate(-angle)

                generated_image = np.array(image)

                # Plot the generated image
                # plt.figure(figsize=(6, 6))
                # plt.imshow(generated_image, cmap=plt.cm.bone, clim=(original_image_array.min(), original_image_array.max()))
                # plt.axis('off')
                # plt.show()

                dataset_list.append(generated_image)
                sample_info.append({'shape': generated_image.shape, 'text_size': text_size, 'angle': angle})

    return dataset_list, sample_info


