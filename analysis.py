import numpy as np
import polars
import seaborn
import glob
import gc
import matplotlib.pyplot as plt

from astropy.timeseries import LombScargle, TimeSeries
from astropy.time import Time
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
    """Calculate the period of the light curve using Lomb-Scargle Periodogram
    
    Optionally, produce a plot of the periodogram"""

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


def plot_phase_folded(times, fluxes, period):
    """Plot a phase-folded light curve"""

    # Normalize flux to one
    fluxes = fluxes / np.max(fluxes)

    # Create a TimeSeries object for phase folding
    tt = TimeSeries(data=fluxes.reshape(-1, 1), time=Time(times.to(u.day), format='jd'))

    # Phase fold using the primary eclipse as reference point
    PRIMARY_ECLIPSE_TIME = Time((np.min(times) + 1.25 * u.h).to(u.day), format="jd")
    folded_tt = tt.fold(period.to(u.day), epoch_time = PRIMARY_ECLIPSE_TIME)

    plot_time = np.array([t.value for t in folded_tt["time"]])

    # Plot
    fig, ax = plt.subplots(figsize=(8,6))

    ax.scatter(plot_time, folded_tt["col0"])
    ax.grid(linestyle=":", alpha=0.5)
    ax.set_xlabel("Phase", fontsize=20)
    ax.set_xlim(-1, 1)
    ax.set_ylabel("Relative Flux", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    fig.savefig("figures/phase_plot.pdf")


if __name__ == "__main__":

    # Load the datas
    times = (np.load("times.npy") * u.day).to(u.h) # type: ignore
    fluxes = np.load("fluxes.npy")    

    plot_light_curve(times, fluxes)

    lc_period = determine_lc_period(times, fluxes, plot=True)
    print(lc_period)

    plot_phase_folded(times, fluxes, lc_period)