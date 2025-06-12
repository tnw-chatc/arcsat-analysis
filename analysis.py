import numpy as np
import polars
import seaborn
import glob
import gc
import matplotlib.pyplot as plt

from astropy.timeseries import LombScargle
import astropy.units as u

import pdb


def plot_light_curve(times, fluxes):
    """Plot the light curve obtained from the reduction process"""

    # Convert JD to hours from the first observation
    # Offset the time such that the first data point is zero
    tt = (times - np.min(times))

    # Normalize flux to one
    ff = fluxes / np.max(fluxes)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(tt, ff)
    ax.set_xlabel("Time Since First Observation (hours)", fontsize=20)
    ax.set_ylabel("Relative Flux", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(linestyle=":", alpha=0.5)
    
    fig.savefig("figures/light_curve.pdf")

    # Clear
    plt.clf()


def determine_lc_period(times, fluxes, plot=False):

    # Construct the periodogram using LombScargle
    frequency, power = LombScargle(times, fluxes).autopower()

    # Pick the strongest frequency
    best_freq = frequency[np.argmax(power)]

    # The actual period is twice the period of the periodiogram
    period = 1/best_freq
    lc_period = period * 2 # type: ignore

    # Plot periodogram
    if plot:
        fig, ax = plt.subplots(figsize=(8,6))

        ax.plot(frequency, power)
        ax.axvline(x=best_freq.value, label="Best frequency", c='orange', linestyle="--")
        ax.set_xlabel("Frequency ($h^{-1}$)", fontsize=20)
        ax.set_ylabel("Power", fontsize=20)
        ax.set_title(f"System period: {lc_period:.2f}", fontsize=24)
        ax.set_xlim(0, 6)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.grid(linestyle=":", alpha=0.5)
        ax.legend(fontsize=20)
        
        fig.savefig("figures/periodogram.pdf")

    return lc_period


if __name__ == "__main__":

    # Load the datas
    times = (np.load("times.npy") * u.day).to(u.h) # type: ignore
    fluxes = np.load("fluxes.npy")    

    plot_light_curve(times, fluxes)

    lc_period = determine_lc_period(times, fluxes, plot=True)
    print(lc_period)