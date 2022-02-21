# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 09:53:00 2021

@author: u0123694
"""
import pandas as pd
import numpy as np
import warnings
from scipy import signal
from scipy.optimize import least_squares
from importlib import resources


def gaussian(x, amplitude, center, width, baselevel=0):

    return amplitude * np.exp(-((x - center) ** 2) / (2 * width ** 2)) + baselevel


def lorentzian(x, amplitude, center, width, baselevel=0):

    return amplitude * width ** 2 / ((x - center) ** 2 + width ** 2) + baselevel


def GaussLorentz(x, amplitude, center, width, baselevel, shape):

    return gaussian(x, amplitude * (1 - shape), center, width, baselevel) + lorentzian(
        x, amplitude * shape, center, width, baselevel
    )


def composeCurves(x, centers, amplitudes, widths, shapes, baselevels):
    """add mixed Gauss-Lorent curves together

    Parameters
    ----------
    x : list-like
    centers, amplitudes, widths, shapes : list-like, 1-dimensional
        parameters for the individual curves. Lists have lenght n for n curves.
    """

    peakAmount = len(centers)

    if isinstance(baselevels, (int, float)):
        baselevels = [baselevels] * peakAmount
        
    params = [
        {"center": i, "amplitude": j, "width": k, "baselevel": l, "shape": m}
        for i, j, k, l, m in 
            zip(centers, amplitudes, widths, baselevels, shapes)
    ]

    curves = GaussLorentz(x, **params[0])

    for peak in params[1:]:
        curves = curves + GaussLorentz(x, **peak)

    return curves


##DATA PROCESSING


def wavelengthToShift(wavelength, laser=532.18):

    return 1e7 / laser - 1e7 / wavelength


def ShiftToWavelength(shift, laser=532.18):

    return 1 / (1 / laser - shift / 1e7)


def neonEmission(laser=532.18):
    "from https://physics.nist.gov/PhysRefData/Handbook/Tables/neontable2.htm"

    with resources.open_text("static", "neon_emissionLines.csv") as df:
        neon = pd.read_csv(df)

    # neon= pd.read_csv('D:/Dropbox/python/packages/petroPy/neon_emissionLines.csv')

    ramanShift = "RamanShift" + str(int(laser))
    neon[ramanShift] = wavelengthToShift(neon["wavelength_nm"], laser=laser)

    return neon


def smooth(y, smoothType="gaussian", kernelWidth=9):
    """
    Parameters
    ----------
    smoothtype  str 
        'movingAverage' or 'gaussian'
    kernelWidth (int)
        width of smoothing kernel in elements of y
    """
    kernelWidth = int(kernelWidth)

    if smoothType == "movingAverage":
        kernel = np.ones((kernelWidth,)) / kernelWidth
    elif smoothType == "gaussian":
        kernel = np.fromiter(
            (
                gaussian(x, 1, 0, kernelWidth / 3, 0)
                for x in range(-(kernelWidth - 1) // 2, (kernelWidth + 1) // 2, 1)
            ),
            np.float,
        )
        kernel = kernel / sum(kernel)
    else:
        ValueError("select smoothtype 'movingAverage' or 'gaussian'")

    return np.convolve(y, kernel, mode="valid")


def long_correction(x, intensities, T_C=25.0, laser=532.18, normalisation="area"):

    """
    Long correction of Raman spectra
    From Long (1977) and Behrens (2006)

    Parameters
    ----------
    spectrum
        dataframe with wavelengths in column 0 and intensities in column 1
    T_C
        temperature of aquisition in degrees celsius
    wavelength
        laser wavelength in nanometers
    normalisation
        'area' for normalisation over the total area underneath the spectrum, or False for no normalisation
    """
    from scipy.constants import c, h, k

    intensities = np.array(intensities)[np.argsort(x)]
    x = np.array(x)[np.argsort(x)]

    # nu0 laser is in M-1 (wave is in nm)
    nu0 = 1.0 / laser * 1e9
    # K temperature
    T = T_C + 273.15  

    # Raman shift from cm-1 to m-1
    nu = 100.0 * x 

    # frequency correction; dimensionless
    frequency = nu0 ** 3 * nu / ((nu0 - nu) ** 4)  
    # temperature correction with Boltzman distribution; dimensionless
    boltzman = 1.0 - np.exp(
        -h * c * nu / (k * T)
    )  
    intensityLong = intensities * frequency * boltzman  # correction

    if normalisation == "area":
        # normalisation over total area
        intensityLong = intensityLong / np.trapz(intensityLong, x)
    elif not normalisation:
        pass
    else:
        raise KeyError("Set normalisation to 'area' or 'False'.")

    return intensityLong



def H2Oraman(rWS, slope):
    """Calculate water contents using the equation (3) from Le Losq et al. (2012)

    equation:
    H2O/(100-H2O)= intercept + slope * rWS

    rWS= (Area water peaks / Area silica peaks) of sample raman spectra

    intercept & slope are determined empirically through calibration with standards
    """
    
    return (100 * slope * rWS) / (1 + slope * rWS)


def diads(x, intensities, peak_prominence=40, fit_window=8, curve="GL"):
    # Fit curves to the two highest peaks in the 1250 - 1450cm-1 window

    # set up the cost function
    curveDict = {"GL": GaussLorentz, "G": gaussian, "L": lorentzian}
    residuals = lambda params, x, spectrum: curveDict[curve](x, *params) - spectrum

    # check if the diads are within range of the spectrum
    if (x.min() > 1250) | (x.max() < 1450):
        raise RuntimeError("spectrum not within 1250 - 1450cm-1")

    intensities = intensities[(x > 1250) & (x < 1450)]
    x = x[(x > 1250) & (x < 1450)]

    # find initial guesses for fitting 2 peaks
    amplitudes = intensities[
        signal.find_peaks(intensities, prominence=peak_prominence)[0]
    ]
    if amplitudes.shape[0] < 2:
        raise RuntimeError("less than two peaks found")
    if amplitudes.shape[0] > 2:
        warnings.warn("more than two peaks found")
    
    sort_index = np.argsort(amplitudes)
    amplitudes = amplitudes[sort_index]

    centers = x[signal.find_peaks(intensities, prominence=peak_prominence)[0]][sort_index]

    # full width half maximum in wavenumbers
    widths = (
        signal.peak_widths(
            intensities, signal.find_peaks(intensities, prominence=peak_prominence)[0]
        )[0]
        * abs(np.diff(x).mean())
    )[sort_index]  

    # Gaussian - Lorentzian mixing paramter
    shape = 0.5  
    # baselevel, should be 0 for baseline corrected spectra
    baselevel = 0  

    init_values1 = np.array([amplitudes[-2], centers[-2], widths[-2], baselevel])
    init_values2 = np.array([amplitudes[-1], centers[-1], widths[-1], baselevel])

    if curve == "GL":
        init_values1 = np.append(init_values1, shape)
        init_values2 = np.append(init_values2, shape)

    # trim fit areas
    trim1 = (x > (init_values1[1] - init_values1[2] * fit_window)) & (
        x < (init_values1[1] + init_values1[2] * fit_window)
    )
    trim2 = (x > (init_values2[1] - init_values2[2] * fit_window)) & (
        x < (init_values2[1] + init_values2[2] * fit_window)
    )
    x1 = x[trim1]
    x2 = x[trim2]
    intensity1 = intensities[trim1]
    intensity2 = intensities[trim2]

    # upper and lower bounds for fit parameters
    if curve == "GL":
        bounds = (
            [-np.inf, -np.inf, -np.inf, -np.inf, 0],
            [np.inf, np.inf, np.inf, np.inf, 1],
        )
    else:
        bounds = (-np.inf, np.inf)

    # least sqaure regressions of the residuals
    fit_params1 = least_squares(
        fun=residuals, x0=init_values1, bounds=bounds, args=(x1, intensity1)
    ).x
    fit_params2 = least_squares(
        fun=residuals, x0=init_values2, bounds=bounds, args=(x2, intensity2)
    ).x

    # tidy data
    labels = ["amplitude", "center", "width", "baselevel"]
    if curve == "GL":
        labels.append("shape")

    fit_params1 = {labels[i]: j for i, j in enumerate(fit_params1)}
    fit_params1["x"] = x1

    fit_params2 = {labels[i]: j for i, j in enumerate(fit_params2)}
    fit_params2["x"] = x2

    return fit_params1, fit_params2


def _extractBIR(x, y, birs):
    """Extract baseline interpolation regions (birs) from a spectrum

    Parameters
    ----------
    x, y : numpy.array
        1-dimensional array with Raman shift (x) and intensity (y)
    birs : numpy.array
        (n,2) shaped array for n baseline interpolation regions (birs). Each row is [lower limit, upper limmit]

    Returns
    -------
    numpy.array, numpy.array
        arrays with values for x and y within the baseline interpolation regions
    """

    spectrum = np.column_stack((x, y))
    for i, j in enumerate(birs):
        if i == 0:
            spectrumBir = spectrum[(spectrum[:, 0] > j[0]) & (spectrum[:, 0] < j[1]), :]
        else:
            birRegion = spectrum[(spectrum[:, 0] > j[0]) & (spectrum[:, 0] < j[1]), :]
            # xfit= np.vstack((xfit,xtemp))
            spectrumBir = np.row_stack((spectrumBir, birRegion))

    return spectrumBir[:, 0], spectrumBir[:, 1]
