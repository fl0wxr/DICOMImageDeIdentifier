import dcm_img_text_remover
from pdb import set_trace as pause
import numpy as np


def time_format(seconds: float) -> str:

    if seconds is not None:
        seconds = int(seconds)
        d = seconds // (3600 * 24)
        h = seconds // 3600 % 24
        m = seconds % 3600 // 60
        s = seconds % 3600 % 60
        if d > 0:
            return '{:02d}:{:02d}:{:02d}:{:02d}'.format(d, h, m, s)
        elif h > 0:
            return '00:{:02d}:{:02d}:{:02d}'.format(h, m, s)
        elif m > 0:
            return '00:00:{:02d}:{:02d}'.format(m, s)
        elif s >= 0:
            return '00:00:00:{:02d}'.format(s)
    return '-'

def convert_one_file(PIPELINE, filename, n_reps = 1):

    total_periods = []
    for iterations in range(n_reps):

        removal_period, total_period = PIPELINE(filename = filename)

        if total_period == -1:
            break

        print('Total time for current run: %.3f'%total_period)
        total_periods.append(total_period)

    if (n_reps != 1) and (total_period != -1):

        removal_period = np.array(removal_period)
        print('Avg removal time: %.3f'%np.mean(removal_period))
        print('Max removal time: %.3f'%np.max(removal_period))
        print('Min removal time: %.3f'%np.min(removal_period))

        total_periods = np.array(total_periods)
        print('Avg total time: %.3f'%np.mean(total_periods))
        print('Max total time: %.3f'%np.max(total_periods))
        print('Min total time: %.3f'%np.min(total_periods))

def convert_multiple_files(PIPELINE, DP):

    total_period = PIPELINE(DP = DP)
    print('Conversion completed successfully')
    print('Total time needed in DD:HH:MM:SS format\n%s'%(time_format(total_period)))

if __name__ == '__main__':

    ## ! Initial parameters: Begin

    GPU = True ## Set to True if you want to invoke NVIDIA GPU
    # PIPELINE = dcm_img_text_remover.keras_ocr_dicom_image_text_remover
    # IN_PATH = 'pos1.dcm'
    PIPELINE = dcm_img_text_remover.MassConversion
    IN_PATH = '../dataset/raw/dummy_directory'

    ## ! Initial parameters: End

    if PIPELINE in [dcm_img_text_remover.keras_ocr_dicom_image_text_remover, dcm_img_text_remover.keras_ocr_dicom_image_generator_text_remover, dcm_img_text_remover.MassConversion]:
        import tensorflow as tf
        if not GPU:
            tf.config.set_visible_devices([], 'GPU')
            print('[DISABLED] PARALLEL COMPUTATION\n\n---')
        elif tf.config.list_physical_devices('GPU')[0][1] == 'GPU':
            print('[ENABLED] PARALLEL COMPUTATION\n\n---')

    if PIPELINE in [dcm_img_text_remover.MassConversion]:
        convert_multiple_files(PIPELINE = PIPELINE, DP = IN_PATH)
    elif PIPELINE in [dcm_img_text_remover.keras_ocr_dicom_image_text_remover, dcm_img_text_remover.keras_ocr_dicom_image_generator_text_remover]:
        convert_one_file(PIPELINE = PIPELINE, filename = IN_PATH, n_reps = 1)