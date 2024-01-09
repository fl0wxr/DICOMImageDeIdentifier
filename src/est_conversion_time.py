from datetime import timedelta
from datetime import datetime
import pandas as pd

from pdb import set_trace as pause


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


## ! keras-ocr | CPU: Begin

## Estimated time required for 1 pixel to be processed
t_1024_1024 = 0.8
t_1 = t_1024_1024 / (1024**2)

## Average number of DICOM image pixels for given modality after downscale
m_mg = 1024**2
m_ct = 528.59*518.79
m_mr = 370.52 * 398.56
m_ot = 1024**2
m_pt = 219.14 * 220.01
m_us = 775.91 * 1024

## Number of DICOM files
n_mg = 169661
n_ct = 1454337
n_mr = 1752555
n_ot = 6868
n_pt = 450023
n_us = 534

## Estimated required time in seconds
t_mg = t_1 * m_mg * n_mg
t_ct = t_1 * m_ct * n_ct
t_mr = t_1 * m_mr * n_mr
t_ot = t_1 * m_ot * n_ot
t_pt = t_1 * m_pt * n_pt
t_us = t_1 * m_us * n_us
t_total = t_mg + t_ct + t_mr + t_ot + t_pt + t_us

# ## Conversion to HH:MM format
# t_mg = timedelta(seconds = t_mg) #'%d:%d:%d'%()
# t_ct = timedelta(seconds = t_ct)
# t_mr = timedelta(seconds = t_mr)
# t_ot = timedelta(seconds = t_ot)
# t_pt = timedelta(seconds = t_pt)
# t_us = timedelta(seconds = t_us)

# ## Format DD:HH:MM
# t_mg = '%d:%d:%d'%(t_mg.days, t_mg.seconds // 3600, t_mg.seconds // 60)
# t_ct = '%d:%d:%d'%(t_ct.days, t_ct.seconds // 3600, t_ct.seconds // 60)
# t_mr = '%d:%d:%d'%(t_mr.days, t_mr.seconds // 3600, t_mr.seconds // 60)
# t_ot = '%d:%d:%d'%(t_ot.days, t_ot.seconds // 3600, t_ot.seconds // 60)
# t_pt = '%d:%d:%d'%(t_pt.days, t_pt.seconds // 3600, t_pt.seconds // 60)
# t_us = '%d:%d:%d'%(t_us.days, t_us.seconds // 3600, t_us.seconds // 60)

print('mg', time_format(t_mg))
print('ct', time_format(t_ct))
print('mr', time_format(t_mr))
print('ot', time_format(t_ot))
print('pt', time_format(t_pt))
print('us', time_format(t_us))
print('total', time_format(t_total))


## ! keras-ocr | CPU: End