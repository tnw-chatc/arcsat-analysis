#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Filename: reduction.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from .bias import create_median_bias
from .darks import create_median_dark
from .flats import create_median_flat, plot_flat
from .science import reduce_science_frame
from .ptc import calculate_gain, calculate_readout_noise
from .photometry import do_aperture_photometry, plot_radial_profile

import glob
import os
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from photutils.centroids import centroid_sources, centroid_2dg, centroid_quadratic, centroid_1dg
from astropy.stats import sigma_clipped_stats

import pdb

def run_reduction(data_dir):
    """This function must run the entire CCD reduction process. You can implement it
    in any way that you want but it must perform a valid reduction for the two
    science frames in the dataset using the functions that you have implemented in
    this module. Then perform aperture photometry on at least one of the science
    frames, using apertures and sky annuli that make sense for the data. The function
    must accept the data_dir as an argument, which is the path to the directory with
    the raw data.

    No specific output is required but make sure the function prints/saves all the
    relevant information to the screen or to a file, and that any plots are saved to
    PNG or PDF files.

    """
    # Read all necessary files from data_dir
    bias_filepath = sorted(glob.glob(data_dir + "/Bias*"))
    dark_filepath = sorted(glob.glob(data_dir + "/Dark*"))
    flat_filepath = sorted(glob.glob(data_dir + "/domeflat*"))
    science_filepaths = sorted(glob.glob(data_dir + "/LPSEB35*"))

    # Filenames of bias, dark, and flat
    median_bias_filepath = data_dir + "/median_bias.fits"
    median_dark_filepath = data_dir + "/median_dark.fits"
    median_flat_filepath = data_dir + "/median_flat.fits"

    # Create bias/dark/flat
    median_bias = create_median_bias(bias_filepath, median_bias_filepath)
    median_dark = create_median_dark(dark_filepath, median_bias_filepath, median_dark_filepath)
    median_flat = create_median_flat(flat_filepath, median_bias_filepath, median_flat_filepath, median_dark_filepath)

    # Calculate gain and readout noise
    gain = calculate_gain(flat_filepath)
    readout_noise = calculate_readout_noise(bias_filepath, gain)

    print(f"[info] Gain: {gain} e-/ADU")
    print(f"[info] Readout Noise: {readout_noise} e-")

    # Iterate over all science
    reduced_sciences = []
    reduced_science_filepath = []
    times = []
    for i, _ in enumerate(science_filepaths):
        reduced_science_filepath.append(data_dir + f"/reduced_science_{str(i).zfill(3)}.fits")
        times.append(fits.getheader(science_filepaths[i])['JD-OBS']) # TODO: check if really works
        # Reduce and save reduced science file to disk
        # Skip if already exists to save time
        if os.path.exists(reduced_science_filepath[i]):
            print(f"Image {reduced_science_filepath[i]} exists. Skipping...")
        else:
            reduced_sciences.append(reduce_science_frame(science_filepaths[i], median_bias_filepath, median_flat_filepath, 
                                                median_dark_filepath, reduced_science_filepath[i]))
            print(f"[info] Image {reduced_science_filepath[i]} saved!")

    # I will perform aperture photometry on this object on the reduced science images
    # The science images are split into two parts: center (indices 0-120) and off-center (indices 121-142)
    POS_1 = np.array([[409, 412], [385, 526], [574, 110]])
    POS_2 = np.array([[482, 441], [460, 559], [641, 144]])
    
    # Perform the aperture photometry
    target_fluxes = []
    comp_fluxes = []
    fluxes = []
    for i, _ in enumerate(reduced_science_filepath):
        temp_img = fits.getdata(reduced_science_filepath[i]).astype("f4") # type: ignore

        mean, median, std = sigma_clipped_stats(temp_img, sigma=2.5)

        if i < 121:
            positions = np.asarray(centroid_sources(temp_img - median, xpos=POS_1[:,0], ypos=POS_1[:,1], box_size=35, centroid_func=centroid_2dg)).T
        else:
            positions = np.asarray(centroid_sources(temp_img - median, xpos=POS_2[:,0], ypos=POS_2[:,1], box_size=35, centroid_func=centroid_2dg)).T

        fluxes_table = do_aperture_photometry(reduced_science_filepath[i], positions, radii=[10], sky_radius_in=15, sky_annulus_width=5)
        target_fluxes.append(fluxes_table["aperture_sum_0"][0])
        comp_fluxes.append(np.mean([fluxes_table["aperture_sum_0"][0], fluxes_table["aperture_sum_0"][1]]))
        fluxes.append(target_fluxes[i] / comp_fluxes[i])
        print(i, science_filepaths[i], fluxes[i])

    np.save("times.npy", times)
    np.save("fluxes.npy", fluxes)
    
    plt.scatter(np.asarray(times) - np.min(times), fluxes)
    plt.show()

    return
