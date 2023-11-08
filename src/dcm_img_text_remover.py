import os, logging

## Warning supression
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

## Warning supression
tf.get_logger().setLevel(logging.ERROR)


import numpy as np
import cv2
import math
from time import time
import keras_ocr ## Documentation: https://keras-ocr.readthedocs.io
import pytesseract
from presidio_image_redactor import ImageRedactorEngine, DicomImageRedactorEngine, bbox
from data_generator import data_generator
import pandas as pd
import tensorflow as tf
import pydicom
from PIL import Image

from matplotlib import pyplot as plt

import visuals
import rw

from pdb import set_trace as pause


def ndarray_size(arr: np.ndarray) -> int:
    return arr.itemsize*arr.size

def basic_preprocessing(img, downscale, toint8 = True, multichannel = True) -> np.ndarray:
    '''
        Description:
            Main preprocessing. It is imperative that the image is converted to (1) uint8 and in (2) RGB in order for keras_ocr's detector to properly function.

        Args:
            downscale. Bool.

        Returns:
            out_image. Its shape is (H, W) if `multichannel` is set to `False`, otherwise its shape is (H, W, 3).
    '''

    if downscale:
        ## Downscale
        downscale_dimensionality = 1024
        new_shape = (min([downscale_dimensionality, img.shape[0]]), min([downscale_dimensionality, img.shape[1]]))
        img = cv2.resize(img, (new_shape[1], new_shape[0]))
        print('Detection input downscaled to (%d, %d)'%(new_shape[0], new_shape[1]))

    if toint8:
        img = (255.0 * (img / np.max(img))).astype(np.uint8)

    if (multichannel) and (len(img.shape) == 2):
        img = np.stack(3*[img], axis = -1)

    return img

def text_remover(img, bboxes: np.ndarray, initial_array_shape, downscaled_array_shape):
    '''
        Args:
            bboxes. Shape (n_bboxes, 4, 2), where 4 is the number of vertices for each box and 2 are the plane coordinates. The vertices inside the bboxes array should be sorted in a way that corresponds to a geometrically counter-clockwise order. For example given a non-rotated (0 degree) bounding box with index 0, the following rule applies
                bboxes[0, 0, :] -> upper left vertex
                bboxes[0, 1, :] -> lower left vertex
                bboxes[0, 2, :] -> lower right vertex
                bboxes[0, 3, :] -> upper right vertex
    '''

    # img = img.max() - img ## In case i want to invert the input

    reducted_region_color = np.mean(img)

    multiplicative_mask = np.ones(downscaled_array_shape, dtype = np.uint8)
    additive_mask = np.zeros(initial_array_shape, dtype = np.uint8)

    ## Concecutive embeddings of bounding boxes
    for bbox in bboxes:

        x0, y0 = bbox[0, 0:(1+1)]
        x1, y1 = bbox[1, 0:(1+1)]
        x2, y2 = bbox[2, 0:(1+1)]
        x3, y3 = bbox[3, 0:(1+1)]

        rectangle = np.array\
        (
            [
                [
                    [x0, y0],
                    [x1, y1],
                    [x2, y2],
                    [x3, y3]
                ]
            ],
            dtype = np.int32 ## Must remain this way. Otherwise, cv2.fillPoly will throw an error.
        )

        ## Filled rectangle
        cv2.fillPoly(multiplicative_mask, rectangle, 0)

    ## When multiplied with image, bounding box pixels will be replaced with 0
    multiplicative_mask = cv2.resize(multiplicative_mask, (initial_array_shape[1], initial_array_shape[0]), interpolation = cv2.INTER_NEAREST)

    ## When added after multiplication, bounding box pixels will be replaced with 255
    additive_mask = reducted_region_color * (multiplicative_mask == 0)

    # print(mask.shape)
    # plt.imshow(mask)
    # plt.show()

    img_ = img.copy()
    img_ = img_ * multiplicative_mask + additive_mask

    return img_

## ! Pipelines: Begin

def presidio():

    t0 = time()

    rw_obj = rw.rw_1_dcm(filename = 'pos2.dcm')

    ## Get DICOM
    dcm = rw_obj.parse_file()

    ## Extract image data from dicom files
    ## Scalar data type -> uint16
    raw_img_uint16_grayscale = dcm.pixel_array

    ## Secondary information about the DICOM file
    print('Input DICOM file information')
    print('Input image shape: ', raw_img_uint16_grayscale.shape)
    print('Modality: ', dcm.Modality)
    print('Physical region: ', dcm.BodyPartExamined, end = 2 * '\n')

    # raw_img_uint16_grayscale = np.array(Image.fromarray(raw_img_uint16_grayscale).rotate(-150))

    img_prep = basic_preprocessing(img = raw_img_uint16_grayscale, downscale = False, toint8 = True, multichannel = False)

    initial_array_shape = raw_img_uint16_grayscale.shape
    downscaled_array_shape = img_prep.shape

    t1 = time()

    engine = ImageRedactorEngine()

    # Redact
    cleaned_img = np.asarray(engine.redact(Image.fromarray(img_prep), fill = 0))
    cleaned_img = cv2.resize(cleaned_img, (initial_array_shape[1], initial_array_shape[0]), interpolation = cv2.INTER_NEAREST)

    removal_period = time() - t1

    vis_obj = visuals.DetectionVisuals(fig_title = 'Based on presidio', n_axes = 2)
    vis_obj.build_plt(imgs = [raw_img_uint16_grayscale, cleaned_img], removal_period = removal_period)

    ## Update the DICOM image data with the modified image
    dcm.PixelData = cleaned_img.tobytes()

    ## Update the DICOM image data with the modified image
    dcm.PixelData = img_prep.tobytes()

    rw_obj.store_fig(figure = vis_obj.fig)

    ## Save modified DICOM
    rw_obj.save_file(dcm = dcm)

    total_period = time() - t0

    return removal_period, total_period

def pytesseract_dicom_image_text_remover():

    rw_obj = rw.rw_1_dcm(filename = 'pos2.dcm')

    ## Get DICOM
    dcm = rw_obj.parse_file()

    ## Extract image data from dicom files
    ## Scalar data type -> uint16
    raw_img_uint16_grayscale = dcm.pixel_array

    print('Image shape: ', raw_img_uint16_grayscale.shape)

    pytesseract_pred = pytesseract.image_to_data(raw_img_uint16_grayscale, output_type = pytesseract.Output.DICT)
    n_bboxes = len(pytesseract_pred['text'])

    bboxes = []
    for bbox_idx in range(n_bboxes):

        (bbox_x, bbox_y, bbox_w, bbox_h) = \
        (
            pytesseract_pred['left'][bbox_idx],
            pytesseract_pred['top'][bbox_idx],
            pytesseract_pred['width'][bbox_idx],
            pytesseract_pred['height'][bbox_idx]
        )

        if (bbox_x, bbox_y, bbox_w, bbox_h) != (0, 0, raw_img_uint16_grayscale.shape[0], raw_img_uint16_grayscale.shape[1]):

            ## Change coordinate system to 4 points
            bboxes.append([[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y]])

    bboxes = np.array(bboxes)

    cleaned_img = text_remover(img = raw_img_uint16_grayscale, bboxes = bboxes)

    ## Visualization
    contour_display = keras_ocr.tools.drawBoxes\
    (
        image = basic_preprocessing(img = raw_img_uint16_grayscale, downscale = True),
        boxes = bboxes,
        thickness = 5
    )
    visuals.DetectionVisuals(fig_title = 'Pipeline: PyTesseract text detector').display([raw_img_uint16_grayscale, contour_display, cleaned_img])

    ## Update the DICOM image data with the modified image
    dcm.PixelData = cleaned_img.tobytes()

    ## Save modified DICOM
    rw_obj.save_file(dcm = dcm)

def keras_ocr_dicom_image_text_remover():

    def prep_det_keras_ocr(img):

        img_prep = basic_preprocessing(img = img, downscale = False)
        bboxes = det_keras_ocr(img_prep)

        return img_prep, bboxes

    def det_keras_ocr(img):

        pipeline = keras_ocr.detection.Detector()

        ## Returns a ndarray with shape (n_bboxes, 4, 2) where 4 is the number of points for each box, 2 are the plane coordinates.
        bboxes = pipeline.detect([img], detection_threshold = .0)[0]

        return bboxes

    t0 = time()

    rw_obj = rw.rw_1_dcm(filename = 'pos2.dcm')

    ## Get DICOM
    dcm = rw_obj.parse_file()

    ## Extract image data from dicom files
    ## Scalar data type -> uint16
    raw_img_uint16_grayscale = dcm.pixel_array

    ## Secondary information about the DICOM file
    print('Input DICOM file information')
    print('Image shape: ', raw_img_uint16_grayscale.shape)
    print('Modality: ', dcm.Modality)
    print('Physical region: ', dcm.BodyPartExamined, end = 2 * '\n')

    # raw_img_uint16_grayscale = np.array(Image.fromarray(raw_img_uint16_grayscale).rotate(-150))

    print('Input image shape: ', raw_img_uint16_grayscale.shape)

    t1 = time()

    raw_img_uint8_grayscale, bboxes = prep_det_keras_ocr(img = raw_img_uint16_grayscale)

    removal_period = time() - t1

    initial_array_shape = raw_img_uint16_grayscale.shape
    downscaled_array_shape = raw_img_uint8_grayscale.shape[:-1]

    if np.size(bboxes) != 0:

        cleaned_img = text_remover\
        (
            img = raw_img_uint16_grayscale,
            bboxes = bboxes,
            initial_array_shape = initial_array_shape,
            downscaled_array_shape = downscaled_array_shape
        )

        ## Contour
        contour_display = keras_ocr.tools.drawBoxes\
        (
            image = raw_img_uint8_grayscale,
            boxes = bboxes,
            thickness = 5
        )
        vis_obj = visuals.DetectionVisuals(fig_title = 'Based on keras-ocr')
        vis_obj.build_plt(imgs = [raw_img_uint16_grayscale, contour_display, cleaned_img], removal_period = removal_period)

        rw_obj.store_fig(figure = vis_obj.fig)

        ## Update the DICOM image data with the modified image
        dcm.PixelData = cleaned_img.tobytes()

    else:

        print('No text detected.')

    ## Save modified DICOM
    rw_obj.save_file(dcm = dcm)

    total_period = time() - t0

    return removal_period, total_period

def keras_ocr_dicom_image_generator_text_remover():

    def prep_det_keras_ocr(img):

        ## Warning: For this particular function calling, downscale MUST REMAIN FALSE.
        img_prep = basic_preprocessing(img = img, downscale = False)
        bboxes = det_keras_ocr(img_prep)

        return img_prep, bboxes

    def det_keras_ocr(img):

        pipeline = keras_ocr.detection.Detector()

        ## Returns a ndarray with shape (n_bboxes, 4, 2) where 4 is the number of points for each box, 2 are the plane coordinates.
        bboxes = pipeline.detect([img])[0]

        return bboxes


    rw_obj = rw.rw_1_dcm('pos2.dcm')

    ## Get DICOM
    dcm = rw_obj.parse_file()

    ## Extract image data from dicom files
    ## Scalar data type -> uint16
    original_image_array = dcm.pixel_array

    dataset_list, sample_info = data_generator(original_image_array = original_image_array)

    for sample_idx in range(len(dataset_list)):

        t0 = time()

        raw_img_uint16_grayscale = dataset_list[sample_idx]

        print('Image shape: ', raw_img_uint16_grayscale.shape)

        raw_img_uint8_grayscale, bboxes = prep_det_keras_ocr(img = raw_img_uint16_grayscale)

        initial_array_shape = raw_img_uint16_grayscale.shape
        downscaled_array_shape = raw_img_uint8_grayscale.shape[:-1]

        cleaned_img = text_remover\
        (
            img = raw_img_uint16_grayscale,
            bboxes = bboxes,
            initial_array_shape = initial_array_shape,
            downscaled_array_shape = downscaled_array_shape
        )

        sample_info[sample_idx]['removal_period'] = time() - t0

        contour_display = keras_ocr.tools.drawBoxes\
        (
            image = raw_img_uint8_grayscale,
            boxes = bboxes,
            thickness = 5
        )
        vis_obj = visuals.DetectionVisuals(fig_title = 'Pipeline: CRAFT text detector')
        vis_obj.build_plt\
        (
            imgs = [raw_img_uint16_grayscale, contour_display, cleaned_img],
            removal_period = sample_info[sample_idx]['removal_period']
        )
        print\
        (
            'Saving as \'res%dx%d_txtsize%d_angle%d.jpg\''%\
            (
                sample_info[sample_idx]['shape'][0],
                sample_info[sample_idx]['shape'][1],
                sample_info[sample_idx]['text_size'],
                sample_info[sample_idx]['angle']
            ),
            end = 2*'\n'
        )
        vis_obj.store_fig\
        (
            fp = '../gen_sample_out/res%dx%d_txtsize%d_angle%d.jpg'%\
            (
                sample_info[sample_idx]['shape'][0],
                sample_info[sample_idx]['shape'][1],
                sample_info[sample_idx]['text_size'],
                sample_info[sample_idx]['angle']
            )
        )

        ## Update the DICOM image data with the modified image
        # dcm.PixelData = cleaned_img.tobytes()

        ## Save modified DICOM
        # rw_obj.save_file(dcm = dcm)

    df_out = pd.DataFrame.from_dict(sample_info)
    df_out.to_csv('../gen_sample_out/metrics.csv')

    total_period = time() - t0

    return -1, -1

## ! Pipelines: End







