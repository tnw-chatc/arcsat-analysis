#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Filename: darks.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from astropy.io import fits
from astropy.stats import sigma_clip
import numpy as np

import pdb

def create_median_dark(dark_list, bias_filename, median_dark_filename):
    """This function must:

    - Accept a list of dark file paths to combine as dark_list.
    - Accept a median bias frame filename as bias_filename (the one you created using
      create_median_bias).
    - Read all the images in dark_list and create a list of 2D numpy arrays.
    - Read the bias frame.
    - Subtract the bias frame from each dark image.
    - Divide each dark image by its exposure time so that you get the dark current
      per second. The exposure time can be found in the header of the FITS file.
    - Use a sigma clipping algorithm to combine all the bias-corrected dark frames
      using the median and removing outliers outside 3-sigma for each pixel.
    - Save the resulting dark frame to a FITS file with the name median_dark_filename.
    - Return the median dark frame as a 2D numpy array.

    """

    # Initialize a list for later use
    dark_images = []

    # Read the bias image
    bias_data = fits.getdata(bias_filename).astype('f4')

    # Read dark images given from the list and then appends them to the array above
    # Also read the exposure time from their corresponding header.
    for filepath in dark_list:
        # Handle corrupted dark fits
        try:
            dark_data = fits.getdata(filepath).astype('f4')[100:-100, 100:-100]
            dark_exptime = fits.getheader(filepath)["EXPTIME"]

            # Subtract trimmed bias from trimmed dark image
            dark_data -= bias_data

            # Divide the dark image with its exposure time and Appends
            dark_images.append(dark_data / dark_exptime)
        except Exception as e:
            print(f"ERROR Reading dark: {e}. Skipping...")
    
    # Perform sigma-clipping
    dark_images_masked = sigma_clip(dark_images, cenfunc="median", sigma=3.0, axis=0)

    # Now that the outliers are gone, we can combine the masked images
    median_dark = np.ma.mean(dark_images_masked, axis=0).data

    # Create the median dark file
    primary = fits.PrimaryHDU(data=median_dark, header=fits.Header())
    hdul = fits.HDUList([primary])
    hdul.writeto(median_dark_filename, overwrite=True)

    return median_dark
