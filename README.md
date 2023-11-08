# Dicom Image De-Identifier

This is a simple DICOM image de-identifier. Its primary function is to receive a DICOM file, extract its payload (one or more images), apply image detection to each corresponding image and then remove the corresponding detected areas from that image.

## Installation

Execute in a terminal
```
python3 -m pip install -r requirements.txt
```

## Results

![](https://raw.githubusercontent.com/fl0wxr/DICOMImageDeIdentifier/master/fig1.png)

Resulting text detection and removal of text from respective bounding boxes based on the CRAFT detector of Keras OCR.

## Available Image Detector/OCR Engines

- [Keras OCR](https://keras-ocr.readthedocs.io/en/latest/)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [Presidio](https://microsoft.github.io/presidio/image-redactor/)

## Text Removal

### Pipeline

To select an image text removal pipeline, open `main.py`, in the line
```
PIPELINE = <PIPELINE_FUNCTION>
```
replace `<PIPELINE_FUNCTION>` with one of the functions between the lines
```
## ! Pipelines: Begin

...

## ! Pipelines: End
```
that can be found inside `dcm_img_text_remover.py`.

### Input File

To select the input DICOM file (with name e.g. `pos2.dcm`), first place it inside `./dataset/raw` and specify its path inside `dcm_img_text_remover.py` in the beginning of the pipeline's function (e.g. `rw_obj = rw.rw_1_dcm(filename = 'pos2.dcm')`) navigate inside `./src` and apply
```
python3 main.py
```

### Outputs

You can find the output DICOM file along with its prediction plot on the path `./dataset/clean` with the corresponding filename as its input. If the plot is unwanted, it can be disabled by commenting out the lines
```
vis_obj = visuals.DetectionVisuals(...)
```
```
vis_obj.build_plt(...)
```
```
rw_obj.store_fig(...)
```
from the associated pipeline function inside `dcm_img_text_remover.py`.

### Keras OCR | Parallel Computation

Depending if your machine supports CUDA, you may enable GPU support to run this software by adjusting the corresponding `GPU` variable inside `main.py`. Hence to enable parallel computation (Requires CUDA!) set it to `True`, otherwise set it to `False`.

## Technical Description

### Pipelines

#### Keras OCR

![](https://raw.githubusercontent.com/fl0wxr/DICOMImageDeIdentifier/master/fig0.png)

A high abstraction of the dicom image text removal pipeline based on keras-ocr. In this demonstrative example, the DICOM file contains exactly one image defined as $I$. The output image $I'$ is the cleaned image, based on the trained CRAFT model's estimation of bounding box locations in the image.
