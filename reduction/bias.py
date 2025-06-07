#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Filename: bias.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from astropy.io import fits
from astropy.stats import sigma_clip
import numpy as np


def create_median_bias(bias_list, median_bias_filename):
    """This function must:

    - Accept a list of bias file paths as bias_list.
    - Read each bias file and create a list of 2D numpy arrays.
    - Use a sigma clipping algorithm to combine all the bias frames using
      the median and removing outliers outside 3-sigma for each pixel.
    - Save the resulting median bias frame to a FITS file with the name
      median_bias_filename.
    - Return the median bias frame as a 2D numpy array.

    """

    # Initializes a list for later use
    bias_images = []

    # Reads bias images given from the list and then appends them to the array above
    for filepath in bias_list:
        bias_data = fits.getdata(filepath)
        bias_images.append(bias_data.astype('f4'))

    # Performs sigma-clipping
    bias_images_masked = sigma_clip(bias_images, cenfunc="median", sigma=3.0, axis=0)

    # Now that the outliers are gone, we can combine the masked images
    median_bias = np.ma.mean(bias_images_masked, axis=0).data

    # Trim bias to get rid of edge pixels for future use
    median_bias = median_bias[100:-100, 100:-100]

    # Create the median bias file
    primary = fits.PrimaryHDU(data=median_bias, header=fits.Header())
    hdul = fits.HDUList([primary])
    hdul.writeto(median_bias_filename, overwrite=True)

    return median_bias
