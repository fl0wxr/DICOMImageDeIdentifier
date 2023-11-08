'''
    Description:
        Read/write related functionality.
'''


import pydicom as dicom
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

from pdb import set_trace as pause


class rw_1_dcm:
    '''
        Description:
            Can read and write one file on disk.
    '''

    def __init__(self, filename):

        self.data_dp = '../dataset/'
        self.filename = filename
        self.raw_data_fp = self.data_dp + 'raw/' + self.filename
        self.clean_data_fp = self.data_dp + 'clean/' + self.filename

    def parse_file(self):

        return dicom.dcmread(self.raw_data_fp)

    def save_file(self, dcm):

        dcm.save_as(self.clean_data_fp)

    def store_fig(self, figure):
        fp = '..' + ''.join(self.clean_data_fp.split('.')[:-1])
        plt.savefig(fp, dpi = 1200)
        plt.close()
        figure.clear()

class rw_2_dcm:
    '''
        Description:
            Can read and write multiple files on disk.
    '''

    def __init__(self):

        self.data_dp = '../dataset/'
        self.raw_data_dp = self.data_dp + 'raw/generated_samples/'
        self.clean_data_dp = self.data_dp + 'clean/generated_samples/'

    def parse_file(self, ImgSize, TextSize):

        dcm = dicom.dcmread(self.raw_data_dp + 'sample_ImgSize_%d_TextSize_%d.dcm'%(ImgSize, TextSize))

        return dcm

    def save_dcm(self, ImgSize, TextSize):

        dcm.save(self.raw_data_dp + 'sample_ImgSize_%d_TextSize_%d.dcm'%(ImgSize, TextSize))

    def store_fig(self, figure):
        fp = '..' + ''.join(self.clean_data_fp.split('.')[:-1])
        plt.savefig(fp, dpi = 1200)
        plt.close()
        figure.clear()

class rw_3:
    '''
        Description:
            Can read and write one file on disk.
    '''

    def __init__(self):

        self.data_dp = '../dataset/'
        self.raw_data_fp = self.data_dp + 'misc/Untitled.jpeg'
        self.clean_data_fp = self.data_dp + 'misc/Untitled_out.jpg'

    def parse_file(self):

        return np.array(Image.open(self.raw_data_fp))

    def save_file(self, img):

        im = Image.fromarray(img)
        im.save(self.clean_data_fp)

