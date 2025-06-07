#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Filename: ptc.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from astropy.io import fits
import numpy as np


def calculate_gain(files):
    """This function must:

    - Accept a list of files that you need to calculate the gain
      (two files should be enough, but what kind?).
    - Read the files and calculate the gain in e-/ADU.
    - Return the gain in e-/ADU.

    """

    # Initializes a list for later use
    flat_images = []

    # Reads flat images given from the list and then appends them to the array above
    for filepath in files:
        flat_data = fits.getdata(filepath)
        flat_images.append(flat_data.astype('f4'))

    # Uses only first two flats to calculate gain
    flat_images = flat_images[:2]

    # Calculates difference and the variance of difference
    flat_diff = flat_images[0] - flat_images[1]
    flat_diff_var = np.var(flat_diff)

    # Calculates the mean signal
    mean_signal = 0.5 * np.mean(flat_images[0] + flat_images[1])

    # This is a placeholder for the actual implementation.
    gain = (2 * mean_signal / flat_diff_var).astype(np.float64)

    return gain


def calculate_readout_noise(files, gain):
    """This function must:

    - Accept a list of files that you need to calculate the readout noise
      (two files should be enough, but what kind?).
    - Accept the gain in e-/ADU as gain. This should be the one you calculated
      in calculate_gain.
    - Read the files and calculate the readout noise in e-.
    - Return the readout noise in e-.

    """
    # Initializes a list for later use
    flat_images = []

    # Reads flat images given from the list and then appends them to the array above
    for filepath in files:
        flat_data = fits.getdata(filepath)
        flat_images.append(flat_data.astype('f4'))

    # Uses only first two flats to calculate gain
    flat_images = flat_images[:2]

    # Calculates difference and the variance of difference
    flat_diff = flat_images[0] - flat_images[1]
    flat_diff_var = np.var(flat_diff)

    # Calculates the readout noise
    readout_noise = np.sqrt((flat_diff_var) * (gain ** 2) / 2).astype(np.float64)

    return readout_noise
