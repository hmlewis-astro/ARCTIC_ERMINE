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
from astropy import modeling
from astropy.convolution import convolve, Gaussian2DKernel, convolve_fft

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

#bar = ChargingBar('   > ', max=len(files))

def update_coords(img, x_guess, y_guess, mask_max_counts=65000, box_width=70, plot_fit=False, smooth=True, kernel_size=10.):
    
    '''
    img: 2D array. Should be the image you are analyzing
        x_guess: int, 1st guess for the x coordinate. Needs to be closer than box_width
        y_guess: int, 1st guess for the y coordinate. Needs to be closer than box_width
        mask_max_counts: Set all points with counts higher than this number equal to the median
        box_width: int,  The area to consider for the stars coordinates. Needs to be small enough to not include
            extra stars, but big enough not to include errors on your x,y guess
    plot_fit: bool, show a plot to the gauss fit?
        smooth: bool, convolve image with gaussian first? The advantage of this is that it will take out some
            of the errors caused by the image being a donut instead of a gaussian. Especially useful for
            non-uniform PSFs, such as ARCSAT's defocused image. For ARCTIC, this may not be necessary.
            Try it anyway though!
        kernel_size: float, standard deviation of gaussian kernel used to smooth data (pixels). Irrevelvant
            if smooth is set to False
    '''
    box_size = int(box_width/2)
    
    x_guess = int(x_guess)
    y_guess=int(y_guess)
    # cutout the part of the image around the star of interest
    stamp = img[y_guess-box_size:y_guess+box_size,x_guess-box_size:x_guess+box_size ].astype(np.float64)
    cutout = np.copy(stamp)

    # change saturated pixels to 0, so it doesn't throw off fit
    cutout[cutout>mask_max_counts] = 0.

    if smooth:
        # Convolve image with gaussian kernel to limit the noise
        gauss_kernel = Gaussian2DKernel(kernel_size)
        cutout = convolve(cutout, gauss_kernel, boundary='extend')
    else:
        cutout_s = cutout
        # Subtract sky background
    cutout -= np.median(cutout)
    
    # Sum pixels in x,y directions
    x_sum = np.sum(cutout, axis=0)
    y_sum = np.sum(cutout, axis=1)
    
    # Fit a gaussian to the x and y summed columns
    offset = np.arange(box_width)-box_size
    fitter = modeling.fitting.LevMarLSQFitter()
    model = modeling.models.Gaussian1D()   # depending on the data you need to give some initial values
    fitted_x = fitter(model, offset, x_sum)
    fitted_y = fitter(model, offset, y_sum)
    
    # Add the offset from the fitted gaussian to the original guess
    
    x_cen = x_guess + fitted_x.mean
    y_cen = y_guess + fitted_y.mean
    x_diff = x_cen - x_guess
    y_diff = y_cen - y_guess

    print("X Guess : ", x_guess, "; X Corrected To : ", x_cen, "; Difference Of : ", (x_diff))
    print("Y Guess : ", y_guess, "; Y Corrected To: ", y_cen, "; Difference Of : ", y_diff)
    return x_cen, y_cen

    if plot_fit:
    
        f, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,5))
    
        ax1.plot(offset, x_sum, 'o', color='C0', label='x offset')
        ax1.plot(offset, y_sum, 'o', color='C1', label='y offset')
    
        ax1.plot(offset, fitted_x(offset), 'C0')
        ax1.plot(offset, fitted_y(offset), 'C1')
    
        ax1.legend()
    
        m,s = np.median(stamp), np.std(stamp)
        ax2.imshow(stamp, vmin=m-s, vmax=m+s, origin='lower', cmap='Greys_r', interpolation='nearest',
        extent=[-box_size,box_size,-box_size,box_size])
        ax2.plot(fitted_x.mean, fitted_y.mean, 'ro', label='updated')
        ax2.plot(0,0, 'bo', label='guess')
        ax2.legend()
    
        ax3.imshow(img, vmin=m-s, vmax=m+s, origin='lower', cmap='Greys_r', interpolation='nearest',)
        ax3.plot(x_cen, y_cen, 'ro', markersize=1)
        ax3.plot(x_guess, y_guess, 'bo', markersize=1)
    
        plt.tight_layout()
        plt.show()

for ff,fname in enumerate(files):
    hdul = pyfits.open(fname)
    header = hdul[0].header
    wcs = WCS(header)
    filt = hdul[0].header['FILTER']
    image = hdul[0].data
    
    mean, median, std = sigma_clipped_stats(image, sigma=3., iters=10)
    
    sigma = 8.
    #decrease sigma ERROR "xcentroid" (line 119)
    
    daofind = DAOStarFinder(threshold=sigma*std, fwhm=15., exclude_border=True)
    sources = daofind(image - median)
    # sources = sources[sources['xcentroid']<1800 and sources['xcentroid']>500 and sources['ycentroid']<1750 and sources['ycentroid']>1000]
    # print sources
    positions = (sources['xcentroid'], sources['ycentroid'])
    #print positions
    results = xpos, ypos = [], []
    xy = os.path.join(reduced_direc,'xypos.txt')
    with open(xy, 'r') as df:
        for row in df:
            x, y = row.split()
            # print("First : ", x, " ", y)
            x, y = update_coords(image, x, y, box_width = 80)
            #print("Second : ", x, " ", y)
            xpos.append(float(x))
            ypos.append(float(y))
            #print(xpos,ypos)
    
    '''
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
    '''
    
    refs = [(x,y) for x,y in zip(xpos,ypos)]
    plot_apertures = CircularAperture(refs, r=45.)
    #plot_apertures = CircularAperture(refs, r=35.)
    plot_annulus_in = CircularAperture(refs, r=50.)
    plot_annulus_out = CircularAperture(refs, r=55.)
    #plot_annulus_in = CircularAperture(refs, r=40.)
    #plot_annulus_out = CircularAperture(refs, r=45.)

    _, new_fname = os.path.split(fname)
    new_fname = os.path.splitext(new_fname)[0]

    '''
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
    '''

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

    #bar.next()

#bar.finish()
