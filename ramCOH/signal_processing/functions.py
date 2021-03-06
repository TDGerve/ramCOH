import pandas as pd
import numpy as np
from importlib import resources
import csaps as cs
from . import curves as c


def wavelengthToShift(wavelength, laser=532.18):

    return 1e7 / laser - 1e7 / wavelength


def ShiftToWavelength(shift, laser=532.18):

    return 1 / (1 / laser - shift / 1e7)


def neonEmission(laser=532.18):
    "from https://physics.nist.gov/PhysRefData/Handbook/Tables/neontable2.htm"

 

    with resources.open_text("ramCOH.static", "neon_emissionLines.csv") as df:
        neon = pd.read_csv(df)

    # neon= pd.read_csv('D:/Dropbox/python/packages/petroPy/neon_emissionLines.csv')

    ramanShift = "RamanShift" + str(int(laser))
    neon[ramanShift] = wavelengthToShift(neon["wavelength_nm"], laser=laser)

    return neon


def smooth(y, smoothType="Gaussian", kernelWidth=9):
    """
    Parameters
    ----------
    y : array-like
        y
    smoothtype : str
        'movingAverage' or 'Gaussian'
    kernelWidth : int
        width of smoothing kernel in elements of y

    Returns
    -------
    smoothed : array
        y smoothed by a kernel
    """
    kernelWidth = int(kernelWidth)

    if smoothType == "movingAverage":
        kernel = np.ones((kernelWidth,)) / kernelWidth
    elif smoothType == "Gaussian":
        kernel = np.fromiter(
            (
                c.Gaussian(x, 1, 0, kernelWidth / 3, 0)
                for x in range(-(kernelWidth - 1) // 2, (kernelWidth + 1) // 2, 1)
            ),
            np.float,
        )
        kernel = kernel / sum(kernel)
    else:
        ValueError("select smoothtype 'movingAverage' or 'Gaussian'")

    return np.convolve(y, kernel, mode="valid")


def long_correction(x, intensities, T_C=25.0, laser=532.18, normalisation=True):

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
    boltzman = 1.0 - np.exp(-h * c * nu / (k * T))
    intensityLong = intensities * frequency * boltzman  # correction

    if normalisation:
        # normalisation over total area
        intensityLong = intensityLong / np.trapz(intensityLong, x)

    return intensityLong


def H2Oraman(rWS, slope):
    """Calculate water contents using the equation (3) from Le Losq et al. (2012)

    equation:
    H2O/(100-H2O)= intercept + slope * rWS

    rWS= (Area water peaks / Area silica peaks) of sample raman spectra

    intercept & slope are determined empirically through calibration with standards
    """

    return (100 * slope * rWS) / (1 + slope * rWS)


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
    birs_x : array
        values for x within baseline interpolation regions
    birs_y : array
        alues for y within baseline interpolation regions
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


def _calculate_noise(x, y, smooth_factor=1):
    """
    Parameters
    ----------
    x : array-like
        x
    y : array-like
        y
    smooth_factor : int, float
        scaling factor applied to 'smooth' parameter of csaps

    Returns
    -------
    noise : float
        Noise on y calculated as the standard deviation on the residuals of y and a fitted smoothed spline
    spline :
        smoothed spline fitted to y
    """
    # Max range in y
    max_difference = y.max() - y.min()
    # Emperically found this is gives ok smoothing factors for most spectra
    smooth = 2e-6 * max_difference * smooth_factor
    # Fit spline
    spline = cs.csaps(x, y, smooth=smooth)
    # Standard deviation on the residuals of y and spline
    noise_data = y - spline(x)
    noise = noise_data.std(axis=None)

    return noise, spline
