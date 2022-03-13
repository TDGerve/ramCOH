import pandas as pd
import numpy as np
import warnings
from scipy import signal
from importlib import resources
import sklearn.metrics as skm
import scipy.optimize as opt
import csaps as cs


def gaussian(x, amplitude, center, width, baselevel=0):

    return amplitude * np.exp(-((x - center) ** 2) / (2 * width ** 2)) + baselevel


def lorentzian(x, amplitude, center, width, baselevel=0):

    return amplitude * width ** 2 / ((x - center) ** 2 + width ** 2) + baselevel


def GaussLorentz(x, amplitude, center, width, baselevel, shape):

    return gaussian(x, amplitude * (1 - shape), center, width, baselevel) + lorentzian(
        x, amplitude * shape, center, width, baselevel
    )


def sum_GaussLorenz(x, centers, amplitudes, widths, shapes, baselevels):
    """add mixed Gauss-Lorentzian curves together

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
        for i, j, k, l, m in zip(centers, amplitudes, widths, baselevels, shapes)
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
    boltzman = 1.0 - np.exp(-h * c * nu / (k * T))
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


def _find_peak_parameters(x, y, prominence, **kwargs):

    prominence_absolute = (prominence / 100) * np.max(y)

    peaks = signal.find_peaks(y, prominence=prominence_absolute, **kwargs)

    amplitudes, centers = y[peaks[0]], x[peaks[0]]
    # full width half maximum in x
    widths = signal.peak_widths(y, peaks[0])[0] * abs(np.diff(x).mean())

    return amplitudes, centers, widths


def _trim_peakFit_areas_OLD(x, y, centers, half_widths, fit_window=4):

    trimmed_areas = []

    for center, width in zip(centers, half_widths):
        trim = (x > (center - width * fit_window)) & (x < (center + width * fit_window))
        trimmed_areas.append([x[trim], y[trim]])

    if len(trimmed_areas) == 1:
        trimmed_areas = trimmed_areas[0]

    return trimmed_areas


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
        List of lists containing start and end for merged ranges
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

    if isinstance(centers, (int, float)):
        minimum = centers - fit_window * half_widths
        maximum = centers + fit_window * half_widths

        return [minimum, maximum]

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

    if not isinstance(ranges[0], list):
        trim = (x > ranges[0]) & (x < ranges[1])
        return [x[trim], y[trim]]

    trimmed_xy = []

    for range in ranges:
        trim = (x > range[0]) & (x < range[1])
        trimmed_xy.append([x[trim], y[trim]])

    return trimmed_xy


def _trim_peakFit_ranges(x, y, centers, half_widths, fit_window=4, merge_overlap=True):

    ranges = _get_peakFit_ranges(
        centers, half_widths, fit_window=fit_window, merge_overlap=merge_overlap
    )

    return _trimxy_ranges(x, y, ranges)


def deconvolve_curve(x, y, prominence=2., noise_threshold=1.5, baseline0=False, max_iterations=15, extra_loops : int=0):
    """
    Docstrings
    """
    # Calculate absolute noise on the signal
    noise, spline = _calculate_noise(x, y)

    # Boundary conditions
    resolution = np.diff(x).mean()
    min_width = 6 * resolution
    xlength = x.max() - x.min()
    # Left and right limits for: center, amplitude, width, shape and baselevel
    leftBoundSimple = [x.min(), noise, min_width, 0., - 5]
    rightBoundSimple = [x.max(), y.max() * 1.5, xlength, 1., y.max()]

    # Initial guesses for peak parameters
    amplitudes, centers, widths = _find_peak_parameters(x, spline, prominence)
    # Remove initial guesses that are too narrow or too low amplitude
    keep = np.where((widths > min_width) & (amplitudes > noise))
    amplitudes = amplitudes[keep]
    centers = centers[keep]
    widths = widths[keep]

    peakAmount = len(amplitudes)
    shapes = np.array([1.0] * peakAmount)
    baselevels = np.array([0.0] * peakAmount)

    initvalues = np.concatenate((centers, amplitudes, widths, shapes, baselevels))
    # Number of parameters, 5 for fitted baselevel, 4 for 0 baselevel
    parameters = 5

    if baseline0:
        leftBoundSimple = leftBoundSimple[:-1]
        rightBoundSimple = rightBoundSimple[:-1]
        initvalues = initvalues[:-peakAmount]
        parameters = 4    
        
    def sumGaussians_reshaped(x, params, peakAmount, baseline_fixed=baseline0, baseline=0.):
        "Reshape parameters to use sum_GaussLorenz in least-squares regression"
        
        if baseline_fixed:
            baselevels = np.array([baseline] * peakAmount)
            params = np.concatenate((params, baselevels))

        values = params.reshape((5, peakAmount))

        return sum_GaussLorenz(x, *values)

    # Noise on ititial fit, is used in the main loop to check if the fit has improved each iteration.
    fit_noise_old = (y - sumGaussians_reshaped(x, initvalues, peakAmount)).std()

    # Cost function to minimise
    residuals = lambda params, x, y, peakAmount: sumGaussians_reshaped(x, params, peakAmount) - y
    
    # Flag for stopping the while loop
    stop = 0
    iterations = 0
    while True:

        # Set up bounds
        leftBound = np.repeat(leftBoundSimple, peakAmount)
        rightBound = np.repeat(rightBoundSimple, peakAmount)
        bounds = (leftBound, rightBound)
        # Optimise fit parameters
        LSfit = opt.least_squares(
            residuals,
            x0=initvalues,
            bounds=bounds,
            args=(x, y, peakAmount),
            loss="soft_l1"
        )
        # Parameters for sum_GaussLorentz
        fitParams = LSfit.x.reshape((parameters, peakAmount))
        if baseline0:
            fitParams = np.vstack((fitParams, np.array([0.] * peakAmount)))

        # Parameters for individual peaks
        paramDict = [
            {"center": i, "amplitude": j, "width": k, "shape": l, "baselevel": m}
            for _, (i, j, k, l, m) in enumerate(zip(*fitParams))
        ]

        # R squared adjusted for noise
        data_mean = y.mean()
        residue =  y - sum_GaussLorenz(x, *fitParams)
        residual_sum = sum((residue / noise)**2)
        sum_squares = sum((y - data_mean)**2)
        R2_noise = 1 - (residual_sum/sum_squares)  

        # Residual noise on the fit, as standard deviation on the residuals
        fit_noise = (y - sum_GaussLorenz(x, *fitParams)).std()    

        iterations += 1
        if iterations >= max_iterations:
            warnings.warn(f"max iterations reached: {max_iterations}")
            break
        # Stop if noise has increased from previous iteration
        if fit_noise_old < fit_noise:
            RuntimeError("Noise increased from last iteration, calculation stopped")

        if fit_noise < (noise * noise_threshold):
            # Stop after some extra loops
            if stop == extra_loops:
                break
            stop += 1

        fit_noise_old = fit_noise.copy()   

        # residue_abs = abs(residue)
        # Add new peak where residuals are higest
        peakAmount += 1
        amplitudes = np.append(amplitudes, y[np.where(residue == residue.max())])
        centers = np.append(centers, x[np.where(residue == residue.max())])
        widths = np.append(widths, widths.mean())
        shapes = np.append(shapes, 1)
        baselevels = np.append(baselevels, 0)

        initvalues = np.concatenate((centers, amplitudes, widths, shapes, baselevels))

        if baseline0:
            initvalues = initvalues[:-peakAmount]

    return fitParams, R2_noise, fit_noise


def _calculate_noise(x, y, smooth_factor=1):

    max_difference = y.max() - y.min()
    # Emperically found this is gives ok smoothing factors
    smooth = 2e-4 * max_difference * smooth_factor

    spline = cs.csaps(x, y, x, smooth=smooth)
    noise_data = y - spline
    noise = noise_data.std(axis=None)

    return noise, spline
