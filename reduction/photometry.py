#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Filename: photometry.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from astropy.io import fits
from astropy.table import Table, vstack
from astropy.stats import sigma_clipped_stats
from photutils.aperture import CircularAnnulus, CircularAperture, ApertureStats, aperture_photometry

import numpy as np
import matplotlib.pyplot as plt

import pdb


def do_aperture_photometry(
    image,
    positions,
    radii,
    sky_radius_in,
    sky_annulus_width,
):
    """This function must:

    - Accept a fully reduced science image as a file and read it.
    - Accept a list of positions on the image as a list of tuples (x, y).
    - Accept a list of aperture radii as a list of floats.
    - Accept a the radius at which to measure the sky background as sky_radius_in.
    - Accept a the width of the annulus as sky_annulus_width.
    - For each position and radius, calculate the flux in the aperture, subtracting
      the sky background. You can do this any way that you like but probably you'll
      want to use SkyCircularAnnulus from photutils.
    - The function should return the results from the aperture photometry. Usually
      this will be an astropy table from calling photutils aperture_photometry, but
      it can be something different if you use a different library.

    Note that the automated tests just check that you are returning from this
    function, but they do not check the contents of the returned data.

    """

    # Load the image
    image_data = fits.getdata(image).astype('f4')

    # Initialize fluxes no bg
    fluxes_no_bg = []

    # Iterate through each position, and radius will be iteration within the if-block
    for x, y in positions:

        # Create CircularAnnulus for subtracting background flux
        annulus = CircularAnnulus((x, y), sky_radius_in, sky_radius_in + sky_annulus_width)

        # Estimate background noise
        annulus_stats = ApertureStats(image_data, annulus)
        back = annulus_stats.median

        # Create a list of CircularAperture for each radius for given position
        apertures = [CircularAperture((x, y), r) for r in radii]
        aperture_areas = [ap.area_overlap(image_data) for ap in apertures]

        # Create a photometry table from image_data and apertures
        phot_table = aperture_photometry(image_data, apertures)

        # Subtract local background from total aperture signal for each radius
        # Create a new dictionary to update the table
        temp_table = Table({f"aperture_sum_{i}": [phot_table[f"aperture_sum_{i}"][0] - back * aperture_areas[i]] for i, r in enumerate(radii)})
        phot_table |= temp_table

        # Append the tables
        fluxes_no_bg.append(phot_table)

    # Join all sub-tables into a single table
    return vstack(fluxes_no_bg)

def plot_radial_profile(aperture_photometry_data, output_filename="radial_profile.png"):
    """This function must:

    - Accept a table of aperture photometry data as aperture_photometry_data. This
      is probably a photutils table, the result of do_aperture_photometry, but you
      can pass anything you like. The table/input data can contain a single target
      or multiple targets, but it must include multiple apertures.
    - Plot the radial profile of a target in the image using matplotlib. If you
      have multiple targets, label them correctly.
    - Plot a vertical line at the radius of the sky aperture used for the photometry.
    - Save the plot to the file specified in output_filename.

    """
    # Get the number of apertures used in the object
    # I.e., get the number of columns, excluding id, x, and y
    num_apertures = len(aperture_photometry_data.columns) - 3

    # Define aperture radius
    ap_radius = 15
    
    # Iterate over each object
    for j in range(len(aperture_photometry_data)):
        # Read radius from the object
        radii = [aperture_photometry_data.meta[f"aperture{i}_r"] for i in range(num_apertures)]
        ap_fluxes = [aperture_photometry_data[f"aperture_sum_{i}"][j] for i in range(num_apertures)]

        # Plot line plots for each radii
        plt.plot(radii, ap_fluxes, marker="o", label=f"Object {j}")

        # Draw the vertical line
        plt.axvline(ap_radius, label="Aperture radius", c='orange', linestyle='--')

        # Cosmetics
        plt.xlim(0, radii[-1])
        plt.xlabel("Radius (pixels)")
        plt.ylabel("Flux (ADU)")

    # Cosmetics
    plt.title("Radial Profile")
    plt.grid(linestyle=':', alpha=0.5)
    plt.legend()
    plt.savefig(output_filename)

    pass
