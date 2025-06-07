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
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

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

    # Plot flats
    plot_flat(median_flat_filepath)

    # Calculate gain and readout noise
    gain = calculate_gain(flat_filepath)
    readout_noise = calculate_readout_noise(bias_filepath, gain)

    print(f"[info] Gain: {gain} e-/ADU")
    print(f"[info] Readout Noise: {readout_noise} e-")

    # Iterate over all science
    reduced_sciences = []
    reduced_science_filepath = []
    for i, _ in enumerate(science_filepaths):
        reduced_science_filepath.append(data_dir + f"/reduced_science_{str(i).zfill(3)}.fits")
        # Reduce and save reduced science file to disk
        reduced_sciences.append(reduce_science_frame(science_filepaths[i], median_bias_filepath, median_flat_filepath, 
                                               median_dark_filepath, reduced_science_filepath[i]))

    # I will perform aperture photometry on this object on the first reduced science image
    # positions = [(1402.17, 1617.67)]
    positions = [(407, 404)]
    
    # Perform the aperture photometry
    # fluxes_table = do_aperture_photometry(reduced_science_filepath[0], positions, radii=np.linspace(1, 30, 30), sky_radius_in=40, sky_annulus_width=5)
    for i, _ in enumerate(reduced_science_filepath):
        fluxes_table = do_aperture_photometry(reduced_science_filepath[i], positions, radii=[15], sky_radius_in=40, sky_annulus_width=5)
        print(i, science_filepaths[i], fluxes_table["aperture_sum_0"][0])
    
    # Clear matplotlib before plotting a new one
    plt.clf()

    # Plot radial profile based on our new fluxes table
    plot_radial_profile(fluxes_table)

    return
