import pandas as pd
import numpy as np
import warnings
from scipy import signal
from importlib import resources
import sklearn.metrics as skm
import scipy.optimize as opt
import csaps as cs

def _merge_overlapping_ranges(ranges):
    """
    Merge

    Parameters
    ----------
    ranges : List[List]
        list of lists containing start and end for per range

    Returns
    -------
    merged ranges : List[List]
        [xmin, xmax] of merged ranges
    """

    merged = [False] * len(ranges)

    merged_ranges = []

    for k, (area1, area2) in enumerate(zip(ranges[:-1], ranges[1:])):

        if merged[k]:
            if (k + 2) == len(ranges):
                merged_ranges.append(area2)
            continue

        if np.max(area1) > np.min(area2):
            merged_ranges.append([np.min(area1 + area2), np.max(area1 + area2)])
            merged[k + 1] = True
        else:
            merged_ranges.append(sorted(area1))
            if (k + 2) == len(ranges):
                merged_ranges.append(area2)

    return merged_ranges


def _get_peakFit_ranges(centers, half_widths, fit_window=4, merge_overlap=True):
    """
    Parameters
    ----------
    centers : array-like
        centers of peaks
    half_widths : array-like
        Full width at half maximum of peaks
    fit_window : int, float
        ranges are calculated as centers +- (FWHM * fit_window)
    merge_overlap : bool
        merge ranges that overlap

    Returns
    -------
    ranges : list[list]
        [xmin, xmax] of ranges around peak centers
    """

    if isinstance(centers, (int, float)):
        minimum = centers - fit_window * half_widths
        maximum = centers + fit_window * half_widths

        return [minimum, maximum]

    sort = np.argsort(centers)
    centers = centers[sort]
    half_widths = half_widths[sort]

    ranges = []

    for center, width in zip(centers, half_widths):
        minimum = center - fit_window * width
        maximum = center + fit_window * width
        ranges.append([minimum, maximum])

    if merge_overlap:
        merged_ranges = _merge_overlapping_ranges(ranges)

        merged = len(centers) - len(merged_ranges)

        while merged > 0:

            old = len(merged_ranges)
            merged_ranges = _merge_overlapping_ranges(merged_ranges)
            merged = old - len(merged_ranges)
            if len(merged_ranges) == 1:
                break

        ranges = merged_ranges

    return ranges


def _trimxy_ranges(x, y, ranges):
    """ "
    Parameters
    ----------
    x : array-like
        x
    y : array-like
        y
    ranges : list, list[list]
        ranges in x as [xmin, xmax] to which y and y will be trimmed

    Returns
    -------
    trimmed_xy : list, list[list]
        x and y trimmed to ranges
    """

    if not isinstance(ranges[0], list):
        trim = (x > ranges[0]) & (x < ranges[1])
        return [x[trim], y[trim]]

    trimmed_xy = []

    for range in ranges:
        trim = (x > range[0]) & (x < range[1])
        trimmed_xy.append([x[trim], y[trim]])

    return trimmed_xy


def _trim_peakFit_ranges(x, y, centers, half_widths, fit_window=4, merge_overlap=True):
    """
    Paramters
    ---------
    x : array-like
        x
    y : array-like
        y
    centers : array-like
        centers of peaks
    half-widths : array-like
        Full width at half maximum (FWHM) for peaks
    fit_window : int, float
        ranges are calculated as centers +- FWHM * fit_window
    merge_overlap : bool
        merge ranges that overlap

    Returns
    -------
    trimmed ranges : list, list[list]
        x and y trimmed around every peak center
    """
    ranges = _get_peakFit_ranges(
        centers, half_widths, fit_window=fit_window, merge_overlap=merge_overlap
    )

    return _trimxy_ranges(x, y, ranges)
    

def _find_peak_parameters(x, y, prominence, **kwargs):
    """
    Paramters
    ---------
    x : array-like
        x
    y : array-like
        y
    prominence : int, float
        prominence of peaks, passed to scipy.signal.find_peaks
    **kwargs
        passed to scipy.signal.find_peaks

    Returns
    -------
    amplitudes : array
        amplitudes of peaks
    centers : array
        centers of peaks
    widths : array
        widths of peaks
    """

    prominence_absolute = (prominence / 100) * np.max(y)

    peaks = signal.find_peaks(y, prominence=prominence_absolute, **kwargs)

    amplitudes, centers = y[peaks[0]], x[peaks[0]]
    # full width half maximum in x
    widths = signal.peak_widths(y, peaks[0])[0] * abs(np.diff(x).mean())

    sort = np.argsort(centers)

    amplitudes = amplitudes[sort]
    centers = centers[sort]
    widths = widths[sort]

    return amplitudes, centers, widths

def diad(x, intensities, peak_prominence=40, fit_window=8, curve="GL"):
    """
    Paramters
    ---------

    Returns
    -------
    """
    # Fit curves to the two highest peaks in the 1250 - 1450cm-1 window

    # set up the cost function
    curveDict = {"GL": GaussLorentz, "G": Gaussian, "L": Lorentzian}
    residuals = lambda params, x, spectrum: curveDict[curve](x, *params) - spectrum

    # check if the diad is within range of the spectrum
    if (x.min() > 1250) | (x.max() < 1450):
        raise RuntimeError("spectrum not within 1250 - 1450cm-1")

    intensities = intensities[(x > 1250) & (x < 1450)]
    x = x[(x > 1250) & (x < 1450)]

    # find initial guesses for peak fitting
    amplitudes, centers, widths = _find_peak_parameters(
        x=x, y=intensities, prominence=peak_prominence
    )

    if amplitudes.shape[0] < 2:
        raise RuntimeError("less than two peaks found")
    if amplitudes.shape[0] > 2:
        warnings.warn("more than two peaks found")

    # Sort peaks low to high amplitude and select the two highest
    sort_index = np.argsort(amplitudes)
    amplitudes = amplitudes[sort_index][-2:]
    centers = centers[sort_index][-2:]
    widths = widths[sort_index][-2:]
    # Gaussian - Lorentzian mixing paramter
    shape = 0.5
    # baselevel, should be 0 for baseline corrected spectra
    baselevel = 0

    # Set bounds for fitting algorithm
    if curve == "GL":
        bounds = (
            [-np.inf, -np.inf, -np.inf, -np.inf, 0],
            [np.inf, np.inf, np.inf, np.inf, 1],
        )
    else:
        bounds = (-np.inf, np.inf)

    # Initialise output variables
    fit_params = []
    fit_x = []
    # Fit curves to the two peaks
    for amplitude, center, width in zip(amplitudes, centers, widths):
        # Set initial guesses
        init_values = np.array(amplitude, center, width, baselevel)
        if curve == "GL":
            init_values = np.append(init_values, shape)

        x_fit, y_fit = _trim_peakFit_ranges(
            x, intensities, center, width, fit_window=fit_window
        )

        params = opt.least_squares(
            fun=residuals, x0=init_values, bounds=bounds, args=(x_fit, y_fit)
        ).x
        fit_params.append(params)
        fit_x.append(x_fit)

    # Unpack output data
    fit_params1, fit_params2 = fit_params
    x1, x2 = fit_x

    # Tidy data
    labels = ["amplitude", "center", "width", "baselevel"]
    if curve == "GL":
        labels.append("shape")

    fit_params1 = {labels[i]: j for i, j in enumerate(fit_params1)}
    fit_params1["x"] = x1

    fit_params2 = {labels[i]: j for i, j in enumerate(fit_params2)}
    fit_params2["x"] = x2

    return fit_params1, fit_params2