"""
ARCTIC_phot.py
Hannah Lewis
hlewis@virginia.edu
2020
    
Automatic reduction pipeline for transit photometry with the Astrophysical Research Consortium Imaging Camera (ARCTIC) at Apache Point Observatory (APO).
    
to use:
python ARCTIC_phot.py path/to/your/data
    
OR place ARCTIC_phot.py in your folder with data and run with no argument:

python ARCTIC_phot.py
    
Performs aperture photometry on science images in the /reduced/data/ directory for multiple filters.
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
from astropy.coordinates import FK5, SkyCoord
from astropy.wcs import WCS
import astropy.units as u

# ignore overwriting reduced files warnings in case you need to rerun
warnings.filterwarnings('ignore', message='Overwriting existing file')

# ignore overflow errors
warnings.filterwarnings('ignore', message='overflow encountered in sinh')

# ignore everything
warnings.filterwarnings('ignore')


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
    os.makedirs(results_direc)

"""
Find sources
"""

import matplotlib
matplotlib.use('agg')

from astropy.stats import sigma_clipped_stats, mad_std
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

print('\n >>> Starting daofind...')

bar = ChargingBar('   > ', max=len(files))

for ff,fname in enumerate(files):
    hdul = pyfits.open(fname)
    header = hdul[0].header
    wcs = WCS(header)
    filt = hdul[0].header['FILTER']
    image = hdul[0].data
    
    results = []
    radec = os.path.join(reduced_direc,'radec.txt')
    with open(radec, 'r') as df:
        for row in df:
            r, d = row.split()
            results.append({'ra':r, 'dec':d})

    fwhm = 18.
    source_snr = 2.
    
    #mean, median, std = sigma_clipped_stats(image, sigma=3., iters=10)
    #daofind = DAOStarFinder(threshold=source_snr*std, fwhm=fwhm, exclude_border=True)
    #sources = daofind(image - median)

    bkg_sigma = mad_std(image)
    daofind = DAOStarFinder(fwhm=fwhm, threshold=source_snr*bkg_sigma)
    sources = daofind(image)

    for star in results:
        star_coord = SkyCoord(star['ra'], star['dec'], unit=(u.hourangle, u.deg))
        xy = SkyCoord.to_pixel(star_coord, wcs=wcs, origin=1)
        x = xy[0].item(0) - 7.0
        y = xy[1].item(0) - 7.0
        for source in sources:
            if(source['xcentroid']-15 < x < source['xcentroid']+15) and source['ycentroid']-15 < y < source['ycentroid']+15:
                star['x'] = x
                star['y'] = y
                star['peak'] = source['peak']
    results = pd.DataFrame(results)

    ref0 = (results['x'][0], results['y'][0])
    ref1 = (results['x'][1], results['y'][1])

    refs = [ref0, ref1]

    plot_apertures = CircularAperture(refs, r=37.)

    _, new_fname = os.path.split(fname)
    new_fname = os.path.splitext(new_fname)[0]

    if str(new_fname)[-1:] == '5':
        norm = ImageNormalize(image, interval=ZScaleInterval(), stretch=SinhStretch())
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(image, cmap='Greys', origin='lower', norm=norm)
        plot_apertures.plot(color='r', lw=1.0, alpha=0.5)
        fig.colorbar(im, label='Counts')
        plt.tight_layout()
        plt.savefig(os.path.join(results_direc,str(new_fname)+'.coor.png'))
        plt.close()
        plt.close()
        plt.close()

    radii = np.arange(1.0,60.0,1.0)
    for r in refs:
        if np.isnan(r).any():
            print('Make sure you remove the file!', fname)
            break
        else:
            apertures = [pt.CircularAperture(refs, r=r) for r in radii]

            phot_table = pt.aperture_photometry(image, apertures)

            if str(new_fname)[-1:] == '5':
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
        
                for s in range(len(results['x'])):
                    aperture_sum = []
                    for j, r in enumerate(radii):
                        col = 'aperture_sum_'+str(j)
                        aperture_sum.append(-2.5*np.log10(phot_table[col][s]))
                    ax.scatter(radii, aperture_sum/np.min(aperture_sum) - 1.0)

                #plt.axvline(x=ap_radii,linestyle='--',linewidth=1.0,c='k')
                plt.axhline(y=0.0, linestyle='--', linewidth=1.0, c='k')
                plt.xlabel('Aperture Radius (pixels)')
                plt.ylabel(r'$\Delta$ Magnitude')
                plt.tight_layout()
                plt.savefig(os.path.join(results_direc,str(new_fname)+'.cog.png'))
                plt.close()
                plt.close()
                plt.close()

            """
            Get magnitudes of those sources
            """

            new_fname_mag = str(new_fname)+'.mag'
            new_fname_mag = open(os.path.join(results_direc,new_fname_mag),'w+')

            ap_radii = 37.0
            apertures = pt.CircularAperture(refs, r=ap_radii)
            new_fname_mag.write('aperture_area \t {} \n'.format(apertures.area))
            annulus_apertures = pt.CircularAnnulus(refs, r_in=40.0, r_out=45.0)
            new_fname_mag.write('annulus_area \t {} \n'.format(annulus_apertures.area))
            new_fname_mag.write('# \n')
            appers = [apertures, annulus_apertures]
            phot_table = pt.aperture_photometry(image, appers, method='exact')
            ascii.write(phot_table, new_fname_mag, delimiter='\t')

    bar.next()

bar.finish()
