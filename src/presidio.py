import glob
from pathlib import Path
import matplotlib.pyplot as plt
import pydicom
from time import time
from presidio_image_redactor import DicomImageRedactorEngine
import os


def compare_dicom_images(
    instance_original: pydicom.dataset.FileDataset,
    instance_redacted: pydicom.dataset.FileDataset,
    figsize: tuple = (9, 4)
) -> None:
    """
    Display the DICOM pixel arrays of both original and redacted as images.

    Args:
        instance_original (pydicom.dataset.FileDataset): A single DICOM instance (with text PHI).
        instance_redacted (pydicom.dataset.FileDataset): A single DICOM instance (redacted PHI).
        figsize (tuple): Figure size in inches (width, height).
    """

    _, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].imshow(instance_original.pixel_array, cmap="gray")
    ax[0].set_title('Original')
    ax[0].axis('off')
    ax[1].imshow(instance_redacted.pixel_array, cmap="gray")
    ax[1].set_title('Redacted')
    ax[1].axis('off')
    # plt.show()
    plt.savefig('../dataset/clean/0_ORIGINAL.png', dpi=1200)

t0 = time()

engine = DicomImageRedactorEngine()

# Load in and process your DICOM file as needed
dicom_instance = pydicom.dcmread('../dataset/raw/0_ORIGINAL.dcm')

# Redact
redacted_dicom_instance = engine.redact(dicom_instance, fill="background")

t1 = time()
print('Detection time: %.3f'%(t1-t0))

compare_dicom_images(dicom_instance, redacted_dicom_instance)

