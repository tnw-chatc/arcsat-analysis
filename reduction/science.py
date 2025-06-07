#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Filename: science.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from astropy.io import fits
from astropy.stats import sigma_clip
from astroscrappy import detect_cosmics

import numpy as np

import pdb

def reduce_science_frame(
    science_filename,
    median_bias_filename,
    median_flat_filename,
    median_dark_filename,
    reduced_science_filename="reduced_science.fits",
):
    """This function must:

    - Accept a science frame filename as science_filename.
    - Accept a median bias frame filename as median_bias_filename (the one you created
      using create_median_bias).
    - Accept a median flat frame filename as median_flat_filename (the one you created
      using create_median_flat).
    - Accept a median dark frame filename as median_dark_filename (the one you created
      using create_median_dark).
    - Read all files.
    - Subtract the bias frame from the science frame.
    - Subtract the dark frame from the science frame. Remember to multiply the
      dark frame by the exposure time of the science frame. The exposure time can
      be found in the header of the FITS file.
    - Correct the science frame using the flat frame.
    - Optionally, remove cosmic rays.
    - Save the resulting reduced science frame to a FITS file with the filename
      reduced_science_filename.
    - Return the reduced science frame as a 2D numpy array.

    """
    # Read all the required images
    science = fits.getdata(science_filename).astype("f4")
    bias = fits.getdata(median_bias_filename).astype("f4")
    flat = fits.getdata(median_flat_filename).astype("f4")
    dark = fits.getdata(median_dark_filename).astype("f4")

    # Trim science to match the other three images
    reduced_science = science[100:-100, 100:-100]

    # Gets science exposure time
    science_exptime = fits.getheader(science_filename)["EXPTIME"]

    # Subtracts science with bias
    reduced_science -= bias

    # Subtracts science with dark
    reduced_science -= dark * science_exptime

    # Corrects science using flat
    reduced_science /= flat

    # Clean cosmic ray using Astroscrappy
    mask, reduced_science = detect_cosmics(reduced_science)

    # Create the reduced science file
    primary = fits.PrimaryHDU(data=reduced_science, header=fits.Header())
    hdul = fits.HDUList([primary])
    hdul.writeto(reduced_science_filename, overwrite=True)

    return reduced_science
