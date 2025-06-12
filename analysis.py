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
    tt = (times - np.min(times)).to(u.h)

    # Normalize flux to one
    ff = fluxes / np.max(fluxes)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(tt, ff)
    ax.set_xlabel("Time Since First Observation (hours)")
    ax.set_ylabel("Relative Flux")
    ax.grid(linestyle=":", alpha=0.5)
    
    fig.savefig("figures/light_curve.pdf")

    # Clear
    plt.clf()


def determine_lc_period(times, fluxes, plot=False):

    frequency, power = LombScargle(times, fluxes).autopower()

    best_freq = frequency[np.argmax(power)]

    period = 1/best_freq

    # The actual period is twice the period of the periodiogram
    lc_period = period.to(u.h) * 2 # type: ignore

    if plot:
        fig, ax = plt.subplots(figsize=(8,6))

        ax.plot(frequency, power)
        ax.axvline(x=best_freq.value, label=period, c='orange')
        ax.set_xlabel("Freq (1/day)")
        ax.set_title(f"System period: {lc_period}")
        ax.set_xlim(0, 30)
        ax.legend()
        
        fig.savefig("figures/periodogram.pdf")

    return lc_period


if __name__ == "__main__":

    # Load the datas
    times = np.load("times.npy") * u.day # type: ignore
    fluxes = np.load("fluxes.npy")    

    plot_light_curve(times, fluxes)

    lc_period = determine_lc_period(times, fluxes, plot=True)
    print(lc_period)