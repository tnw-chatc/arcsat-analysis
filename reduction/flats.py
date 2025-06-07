#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Filename: flats.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from astropy.io import fits
from astropy.stats import sigma_clip
from scipy.stats import mode
from astropy.visualization import ImageNormalize, LinearStretch, ZScaleInterval

import numpy as np
import matplotlib.pyplot as plt

import pdb


def create_median_flat(
    flat_list,
    bias_filename,
    median_flat_filename,
    dark_filename=None,
):
    """This function must:

    - Accept a list of flat file paths to combine as flat_list. Make sure all
      the flats are for the same filter.
    - Accept a median bias frame filename as bias_filename (the one you created using
      create_median_bias).
    - Read all the images in flat_list and create a list of 2D numpy arrays.
    - Read the bias frame.
    - Subtract the bias frame from each flat image.
    - Optionally you can pass a dark frame filename as dark_filename and subtract
      the dark frame from each flat image (remember to scale the dark frame by the
      exposure time of the flat frame).
    - Use a sigma clipping algorithm to combine all the bias-corrected flat frames
      using the median and removing outliers outside 3-sigma for each pixel.
    - Create a normalised flat divided by the median flat value.
    - Save the resulting median flat frame to a FITS file with the name
      median_flat_filename.
    - Return the normalised median flat frame as a 2D numpy array.

    """

    # Initializes the array for later use
    flat_r_images = []

    # Reads the bias filename
    median_bias = fits.getdata(bias_filename)

    # Reads data and header from the flat list
    for filepath in flat_list:
        flat_data = fits.getdata(filepath)
        flat_filter = fits.getheader(filepath)["FILTER"]

        # Since the given flat images are in r filter, we will use it here
        if flat_filter == "g":
            # Subtract trimmed flat with median trimmed flat image 
            # Trim to get rid of edge pixels
            subtracted_flat = flat_data[100:-100, 100:-100].astype("f4") - median_bias

            # Subtracts with median dark image if parsed
            if dark_filename is not None:
                median_dark = fits.getdata(dark_filename).astype("f4")
                subtracted_flat -= median_dark

            flat_r_images.append(subtracted_flat)
      
    # Performs sigma-clipping
    flat_r_masked = sigma_clip(flat_r_images, cenfunc="median", sigma=3, axis=0)

    # Now that the outliers are gone, we can combine the masked images
    median_flat_g_unnorm = np.ma.mean(flat_r_masked, axis=0).data

    # Normalization
    flat_mode, _ = mode(median_flat_g_unnorm.flatten(), nan_policy="omit")
    median_flat = median_flat_g_unnorm / flat_mode # This one is normalized

    # Replace the zeros to prevents error when we try to use it later
    median_flat[median_flat == 0] = 1

    # Create the median flat file
    primary = fits.PrimaryHDU(data=median_flat, header=fits.Header())
    hdul = fits.HDUList([primary])
    hdul.writeto(median_flat_filename, overwrite=True)

    return median_flat


def plot_flat(
    median_flat_filename,
    ouput_filename="median_flat.png",
    profile_ouput_filename="median_flat_profile.png",
):
    """This function must:

    - Accept a normalised flat file path as median_flat_filename.
    - Read the flat file.
    - Plot the flat frame using matplotlib.imshow with reasonable vmin and vmax
      limits. Save the plot to the file specified by output_filename.
    - Take the median of the flat frame along the y-axis. You'll end up with a
      1D array.
    - Plot the 1D array using matplotlib.
    - Save the plot to the file specified by profile_output_filename.

    """
    # Loads the median flat image
    median_flat = fits.getdata(median_flat_filename).astype("f4")

    # Normalizes the image for visualization and plot
    norm = ImageNormalize(median_flat, interval=ZScaleInterval(), stretch=LinearStretch())
    im = plt.imshow(median_flat, origin="lower", norm=norm, cmap="gray")

    # Saves the median flat image
    plt.savefig(ouput_filename) # A typo?? I don't think I can change the argument name or the autograder will break

    # Calculates the median along y-axis (axis=0)
    y_median = np.median(median_flat, axis=0)

    # Plots the 1D array and saves the plot
    plt.clf() # Clear the figure
    _ = plt.plot(y_median)
    plt.savefig(profile_ouput_filename)

    return
