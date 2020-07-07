"""
ARCTIC_transit.py
Hannah Lewis
hlewis@virginia.edu
2020
    
Automatic reduction pipeline for transit photometry with the Astrophysical Research Consortium Imaging Camera (ARCTIC) at Apache Point Observatory (APO).
    
to use:
python ARCTIC_transit.py path/to/your/data
    
OR place ARCTIC_transit.py in your folder with data and run with no argument:

python ARCTIC_transit.py
    
Makes various plots of the relative magnitudes, flux versus time, airmass.
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

#from progress.bar import ChargingBar

from scipy import stats
import scipy.signal
import scipy.optimize as optimize

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

matplotlib.rcParams.update({'xtick.labelsize': 14})
matplotlib.rcParams.update({'ytick.labelsize': 14})

sdss_array = ['u', 'g', 'r', 'i', 'z']
color_array = ['royalblue', 'mediumseagreen', 'gold', 'darkorange', 'tomato']

files = glob.glob(os.path.join(results_direc, "*.result"))

print('\n >>> Starting plots...')

filters = []
filts = os.path.join(reduced_direc,'filts.txt')
with open(filts, 'r') as df:
    for row in df:
        filters.append(row.strip('\n'))

#bar = ChargingBar('   > ', max=len(filters))

#flux
fig_all_flux_transit = plt.figure()
ax_all_flux_transit = fig_all_flux_transit.add_subplot(1, 1, 1)

#magnitude
fig_all_magn_transit = plt.figure()
ax_all_magn_transit = fig_all_magn_transit.add_subplot(1, 1, 1)

flux_star = []
flux_ref = []

magn_star = []
magn_ref = []

for i,filt in enumerate(filters):
    result_filter = "*"+str(filt)+".result"
    files = glob.glob(os.path.join(results_direc, result_filter))
    
    #time vs airmass
    fig_time_airmass = plt.figure()
    ax_time_airmass = fig_time_airmass.add_subplot(1, 1, 1)
    
    #flux
    fig_flux_airmass = plt.figure()
    ax_flux_airmass = fig_flux_airmass.add_subplot(1, 1, 1)
    
    fig_flux_time = plt.figure()
    ax_flux_time = fig_flux_time.add_subplot(1, 1, 1)
    
    fig_flux_transit = plt.figure()
    ax_flux_transit = fig_flux_transit.add_subplot(1, 1, 1)
    
    #magnitude
    fig_magn_airmass = plt.figure()
    ax_magn_airmass = fig_magn_airmass.add_subplot(1, 1, 1)
    
    fig_magn_time = plt.figure()
    ax_magn_time = fig_magn_time.add_subplot(1, 1, 1)
    
    fig_magn_transit = plt.figure()
    ax_magn_transit = fig_magn_transit.add_subplot(1, 1, 1)
    
    for ff,rname in enumerate(files):
        
        with open(rname) as f:
            lines = f.readlines()
            image = [line.split()[0] for line in lines]
            id = [line.split()[1] for line in lines]
            otime_mjd = [float(line.split()[2]) for line in lines]
            xairmass = [float(line.split()[3]) for line in lines]
            exptime = [float(line.split()[4]) for line in lines]
            flux = [float(line.split()[5]) for line in lines]
            epadu = [float(line.split()[6]) for line in lines]
            area = [float(line.split()[7]) for line in lines]
            stdev = [float(line.split()[8]) for line in lines]
            nsky = [float(line.split()[9]) for line in lines]
            
            
            bkg_mean = [st / ns for st,ns in zip(stdev,nsky)]
            bkg_sum = [bkg * ar for bkg,ar in zip(bkg_mean,area)]
            final_flux = [fl - bkg for fl,bkg in zip(flux,bkg_sum)]
            magn = 25.0 - 2.5*np.log10(final_flux) + 2.5*np.log10(exptime)
            
            #time vs airmass
            ax_time_airmass.scatter(otime_mjd, xairmass, c=color_array[int(id[0])])
            
            #flux
            ax_flux_airmass.scatter(xairmass, final_flux, c=color_array[int(id[0])])
            ax_flux_time.scatter(otime_mjd, final_flux, c=color_array[int(id[0])])
                    
            #magnitude
            ax_magn_airmass.scatter(xairmass, magn, c=color_array[int(id[0])])
            ax_magn_time.scatter(otime_mjd, magn, c=color_array[int(id[0])])

            if id[0] == str(0):
                flux_star = np.asarray(final_flux)
                magn_star = np.asarray(magn)

            if filt == 'r':
                if id[0] == str(2):
                    flux_ref = np.asarray(final_flux)
                    magn_ref = np.asarray(magn)
            if filt == 'i':
                if id[0]  == str(1):
                    flux_ref = np.asarray(final_flux)
                    magn_ref = np.asarray(magn)

    if np.max(flux_star) < np.max(flux_ref):
        factor = 1.0
    else:
        factor = -1.0

    rel_flux = factor*(flux_star - flux_ref)/flux_ref + 1.0
    rel_magn = (magn_star / magn_ref)**factor

    def remove_outliers(x,y,sigma=3.):
    
        x = np.array(x)
        y = np.array(y)
    
        med_y = np.median(y)
        std_y = np.std(y)
    
        keep = np.abs(y - med_y)<sigma*std_y
    
        return x[keep], y[keep]

    mjd_keep_flux, flux_keep = remove_outliers(otime_mjd, rel_flux, sigma=2.0)
    mjd_keep_magn, magn_keep = remove_outliers(otime_mjd, rel_magn, sigma=2.0)

    out_of_transit_flux = np.append(flux_keep[0:10], flux_keep[-10:])
    out_of_transit_mjd_flux = np.append(mjd_keep_flux[0:10], mjd_keep_flux[-10:])

    out_of_transit_magn = np.append(magn_keep[0:10], magn_keep[-10:])
    out_of_transit_mjd_magn = np.append(mjd_keep_magn[0:10], mjd_keep_magn[-10:])

    c_flux = np.polyfit(out_of_transit_mjd_flux, out_of_transit_flux, 2)
    c_magn = np.polyfit(out_of_transit_mjd_magn, out_of_transit_magn, 2)

    fit_flux = flux_keep / (c_flux[2] + c_flux[1]*mjd_keep_flux + c_flux[0]*mjd_keep_flux**2.0)
    fit_magn = magn_keep / (c_magn[2] + c_magn[1]*mjd_keep_magn + c_magn[0]*mjd_keep_magn**2.0)

    ax_all_flux_transit.scatter(mjd_keep_flux, fit_flux - 0.30*i, c=color_array[i], label="SDSS "+str(filt)+"'")
    ax_all_flux_transit.axhline(y=1.00 - 0.30*i, c='k', ls='--')

    ax_all_magn_transit.scatter(mjd_keep_magn, fit_magn - 0.030*i, c=color_array[i], label="SDSS "+str(filt)+"'")
    ax_all_magn_transit.axhline(y=1.00 - 0.030*i, c='k', ls='--')

    #time vs airmass
    ax_time_airmass.set_xlabel('MJD')
    ax_time_airmass.set_ylabel('Airmass')
    ax_time_airmass.invert_yaxis()

    #flux
    ax_flux_airmass.set_xlabel('Airmass')
    ax_flux_airmass.set_ylabel('Instrumental Flux')
    #ax_flux_airmass.invert_yaxis()

    ax_flux_time.set_xlabel('MJD')
    ax_flux_time.set_ylabel('Instrumental Flux')
    #ax_flux_time.invert_yaxis()

    #magnitude
    ax_magn_airmass.set_xlabel('Airmass')
    ax_magn_airmass.set_ylabel('Instrumental Magnitude')
    ax_magn_airmass.invert_yaxis()

    ax_magn_time.set_xlabel('MJD')
    ax_magn_time.set_ylabel('Instrumental Magnitude')
    ax_magn_time.invert_yaxis()


    plt.tight_layout()
    fig_time_airmass.savefig(os.path.join(results_direc,'time_x_'+str(filt)+'.png'),dpi=300)
    fig_flux_airmass.savefig(os.path.join(results_direc,'flux_x_'+str(filt)+'.png'),dpi=300)
    fig_flux_time.savefig(os.path.join(results_direc,'flux_time_'+str(filt)+'.png'),dpi=300)
    fig_magn_airmass.savefig(os.path.join(results_direc,'magn_x_'+str(filt)+'.png'),dpi=300)
    fig_magn_time.savefig(os.path.join(results_direc,'magn_time_'+str(filt)+'.png'),dpi=300)

#bar.next()

ax_all_flux_transit.set_xlabel('MJD', fontsize=20)
ax_all_flux_transit.set_ylabel('Relative Flux', fontsize=20)
ax_all_flux_transit.legend(fontsize=12)
fig_all_flux_transit.tight_layout()
fig_all_flux_transit.savefig(os.path.join(results_direc,'transits_flux.png'),dpi=300)

ax_all_magn_transit.set_xlabel('MJD',fontsize=20)
ax_all_magn_transit.set_ylabel('Relative Magnitude',fontsize=20)
ax_all_magn_transit.legend(fontsize=12)
fig_all_magn_transit.tight_layout()
fig_all_magn_transit.savefig(os.path.join(results_direc,'transits_magn.png'),dpi=300)


#bar.finish()
