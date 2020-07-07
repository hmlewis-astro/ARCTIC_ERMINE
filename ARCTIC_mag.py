"""
ARCTIC_mag.py
Hannah Lewis
hlewis@virginia.edu
2020
    
Automatic reduction pipeline for transit photometry with the Astrophysical Research Consortium Imaging Camera (ARCTIC) at Apache Point Observatory (APO).
    
to use:
python ARCTIC_mag.py path/to/your/data
    
OR place ARCTIC_mag.py in your folder with data and run with no argument:

python ARCTIC_mag.py
    
Converts raw photometry to instrumental magnitudes.
"""

import os
import re
import sys
import warnings

import numpy as np

import glob

import pandas as pd

import astropy.io.fits as pyfits
from astropy.io import ascii

# ignore overwriting reduced files warnings in case you need to rerun
warnings.filterwarnings('ignore', message='Overwriting existing file')

# ignore overflow errors
warnings.filterwarnings('ignore', message='overflow encountered in sinh')


"""
Find reduced data
"""

# take directory from user or assume current directory
if len(sys.argv) > 1:
    direc = sys.argv[1]
else:
    direc = '.'

cals_direc = os.path.join(direc, 'reduced', 'cals')
reduced_direc = os.path.join(direc, 'reduced', 'data')
results_direc = os.path.join(reduced_direc, 'results')

# directories for reduced images
if not os.path.exists(cals_direc):
    print('   > Reduced cals directory does not exist! Run ARCTIC_imagered.py first.')
if not os.path.exists(reduced_direc):
    print('   > Reduced data directory does not exist! Run ARCTIC_imagered.py first.')
if not os.path.exists(results_direc):
    print('   > Results directory does not exist! Run ARCTIC_phot.py first.')

"""
Find results
"""

import matplotlib
matplotlib.use('agg')

from astropy.stats import sigma_clipped_stats
from astropy.time import Time
from astropy.visualization import SqrtStretch, SinhStretch, MinMaxInterval, PercentileInterval, ZScaleInterval
from astropy.visualization.mpl_normalize import ImageNormalize

import csv

import datetime

import matplotlib.pyplot as plt

import photutils as pt
from photutils import DAOStarFinder, find_peaks, aperture_photometry, CircularAperture

from progress.bar import ChargingBar

from scipy import stats
import scipy.signal
import scipy.optimize as optimize

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

files = glob.glob(os.path.join(reduced_direc, "*.fits"))
mag_files = glob.glob(os.path.join(results_direc, "*.mag"))

print('\n >>> Deriving magnitudes and creating result files...')

results = []
radec = os.path.join(reduced_direc,'radec.txt')
with open(radec, 'r') as df:
    for row in df:
        r, d = row.split()
        results.append({'ra':r, 'dec':d})

ref = range(len(results))

bar = ChargingBar('   > ', max=len(ref))

for i in ref:
    
    for ff,mname in enumerate(mag_files):
        hdul = pyfits.open(files[ff])
        filt = hdul[0].header['FILTER']
        sdss_filt = filt.strip('SDSS # 2')
        UTC = hdul[0].header['DATE-OBS']
        UTC = Time(UTC, format='isot', scale='utc')
        AIRMASS = hdul[0].header['AIRMASS']
        EXPTIME = hdul[0].header['EXPTIME']
        EPADU = hdul[0].header['GTGAIN11']


        if i == 0:
            results_file = 'star_exopl_'+str(sdss_filt)+'.result'
            result = open(os.path.join(results_direc,results_file),'a')
        else:
            results_file = 'star_ref_'+str(i)+'_'+str(sdss_filt)+'.result'
            result = open(os.path.join(results_direc,results_file),'a')
        
        mfile = open(mname,'r')
        for j,line in enumerate(mfile):
            if line.startswith('#') or line.startswith('id'):
                continue
            if line.startswith('aperture_area') or line.startswith('annulus_area'):
                splist = line.split('\t')
                if line.startswith('aperture_area'):
                    aperture_area = float(splist[1])
                if line.startswith('annulus_area'):
                    annulus_area = float(splist[1])
            if (j-4) == i:
                splist = line.split('\t')
                id = int(splist[0])
                xval = float(splist[1])
                yval = float(splist[2])
                aperture_sum = float(splist[3])
                sky_flux = float(splist[4])

        ID = i
        FLUX = aperture_sum
        EPADU = hdul[0].header['GTGAIN11']
        AREA = aperture_area
        STDDEV = sky_flux
        NSKY = annulus_area

        _, new_mname = os.path.split(mname)
        new_mname = os.path.splitext(new_mname)[0]
        
        result.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(new_mname, ID, UTC.mjd, AIRMASS, EXPTIME, FLUX, EPADU, AREA, STDDEV, NSKY))

    bar.next()

bar.finish()

