# ARCTIC ERMINE
> Pipeline for extracting lightcurves from time-series photometry data taken with the ARCTIC instrument on the ARC 3.5-m Telescope at Apache Point Observatory.

The ARCTIC Extracting Relative Magnitudes to INvestigate Exoplanets (ERMINE) pipeline is designed to reduce time-series photometry data from the Astrophysical Research Consortium (ARC) Telescope Imagine Camera (ARCTIC) on the ARC 3.5-m Telescope. Data is assumed to have been taking with (1) the use of a diffuser and (2) in multiple band-passes.

<!-- [](header.png) -->

<!-- ## Installation-->

<!--OS X & Linux: -->

<!--```sh
npm install my-crazy-module --save```-->

<!--Windows: -->

<!--```sh
edit autoexec.bat
``` -->

## Usage example

To run the each component of the pipeline:

```sh
python ARCTIC_file.py path/to/your/data 
```
    
OR place all pieces of the pipeline in the folder with your data and run with no argument:

```sh
python ARCTIC_file.py
```
Pipeline order is:
1. ARCTIC_imagered.py -- Creates /reduced/cals/ and /reduced/data/ directories, and fills those directories with reduced calibration and science images.
2. ARCTIC_phot.py -- Performs aperture photometry on science images in the /reduced/data/ directory for multiple filters.
3. ARCTIC_mag.py -- Converts raw photometry to instrumental magnitudes.
4. ARCTIC_transit.py --Makes various plots of the relative magnitudes, flux versus time, airmass.

<!--## Development setup -->

<!-- Describe how to install all development dependencies and how to run an automated test-suite of some kind. Potentially do this for multiple platforms. -->

<!-- ```sh
make install
npm test
``` -->

## Release History

* 0.0.1
    * Work in progress

## Meta

Hannah Lewis – hlewis@virginia.edu

Distributed under the MIT License. See ``LICENSE`` for more information.

[https://github.com/hmlewis-astro](https://github.com/hmlewis-astro)
