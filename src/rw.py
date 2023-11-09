'''
    Description:
        Read/write related functionality.
'''


import pydicom as dicom
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import shutil
import os
from glob import glob

from pdb import set_trace as pause


class rw_1_dcm:
    '''
        Description:
            Can read and write one file on disk.
    '''

    def __init__(self, filename):

        self.filename = filename
        self.raw_data_fp = '../dataset/raw/' + self.filename
        self.clean_data_fp = '../dataset/clean/' + self.filename

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
            Can read and write multiple files on a directory. Given a directory path (e.g. '../dataset/raw/direc'), it
            1. Copies the directory structure along with all non-DICOM files inside '../dataset/clean'.
            2. Recursively searches all DICOM files inside the input directory.
            3. Parses all found DICOM files from inside the input directory, and pastes them in the respective paths of the output directory after a potential modification.
    '''

    def __init__(self, dp: str):
        '''
            Args:
                dp: Directory path of the input directory.
        '''

        self.modifiable_file_extension_names = \
        [
            'dcm',
            'dcM',
            'dCm',
            'dCM',
            'Dcm',
            'DcM',
            'DCm',
            'DCM'
        ]

        self.SAFETY_SWITCH = False
        if not self.SAFETY_SWITCH:
            print('W: Safety switch is off. Output directory can now be deleted.')

        if dp[-1] != '/': dp = dp + '/'
        self.raw_data_dp = dp
        self.clean_data_dp = '../dataset/clean/' + self.raw_data_dp.split('/')[-2] + '/'
        self.copy_dir_structure()
        self.raw_dicom_paths = self.generate_dicom_paths(data_dp = self.raw_data_dp)

        self.n_dicom_files = len(self.dicom_paths)

        self.DICOM_IDX = -1

    def __next__(self):

        self.DICOM_IDX += 1

    def copy_dir_structure(self):
        '''
            Description:
                Generates an empty replica of the input directory. The replica is placed inside '../dataset/clean/'. It also includes all non-DICOM files.
        '''

        ## Rule where for an existing filesystem path, if the path corresponds to a DICOM file path it is added to the output's list.
        def ffilter(dir, all_files):
            filtered_files = []
            for f in all_files:
                if ( f.split('.')[-1] in self.modifiable_file_extension_names ) and ( os.path.isfile(os.path.join(dir, f)) ):
                    filtered_files.append(f)

            return filtered_files

        if os.path.exists(self.clean_data_dp):
            print('W: Output directory already exists. Proceeding to delete it.')
            self.rm_out_dir()

        ## Recursively parses all directory paths and copies structure to the clean directory.
        print('Created a new output directory.')
        shutil.copytree\
        (
            src = self.raw_data_dp,
            dst = self.clean_data_dp,
            ignore = ffilter
        )

    def rm_out_dir(self):
        '''
            Description:
                This removes the output directory.
        '''

        if self.SAFETY_SWITCH:
            print('E: Safety switch is on, hence output directory will not be deleted.\nAborting.')
            exit()
        else:
            shutil.rmtree(self.clean_data_dp)
            print('W: Output directory deleted.')

    def generate_dicom_paths(self, data_dp):

        dicom_paths = []
        for extension_name in self.modifiable_file_extension_names:
            dicom_paths += \
            (
                glob\
                (
                    pathname = data_dp + '**/*.' + extension_name,
                    recursive = True
                )
            )

        return dicom_paths

    def parse_file(self):

        dcm = dicom.dcmread()

        return dcm

    def save_dcm(self, ):

        dcm.save(self.raw_data_dp)