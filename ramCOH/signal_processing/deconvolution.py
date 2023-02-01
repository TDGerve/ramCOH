import warnings

import numpy as np
import scipy.optimize as opt
from sklearn.metrics import mean_squared_error

from . import curve_fitting as cf
from . import curves as c
from . import functions as f


def deconvolve_signal(
    x,
    y,    
    min_peak_width=4,
    min_amplitude=2,
    residuals_threshold=0.9,
    max_iterations=5,
    baseline0=True,
    noise=None,
    
):
    """
    Parameters
    ----------
    x : array-like
        x
    y : array-like
        y
    noise_threshold : float
        Fit is accepted when the standard deviation of the fit residuals fall below (noise on y) * noise_threshold
    min_peak_width : int, float
        minimum width of fitted peaks (full width at half maximum) in x stepsize.
    min_amplitude : int, float
        minium amplitude of fitted peaks as a factor of noise on y.
    baseline0 : bool
        fix baselevel of fitted curves to 0
    noise : float, int (optional)
        Absolute noise on y
    max_iterations : int
        maximum loop iterations. One new curve is added each loop

    Returns
    -------
    fitParams : list[list]
        fitted parameters for sum_GaussLorentz
    R2_noise : float
        R squared of fit result to y, adjusted for noise (on y)
    fit_noise : float
        standard deviation on the residuals of y and the fit result
    """

    # Calculate noise on the total signal
    if noise is None:
        noise, _ = f._calculate_noise(x=x, y=y)

    # Set peak prominence to x times noise levels
    # prominence = ((noise * (min_amplitude-1)) / y.max()) * 100

    # Boundary conditions
    resolution = abs(np.diff(x).mean())
    min_width = resolution * min_peak_width#min_peak_width at FWHM
    min_amplitude = noise * min_amplitude
    prominence = min_amplitude + (noise / 2)
    xlength = x.max() - x.min()
    # Left and right limits for: center, amplitude, width, shape and baselevel
    leftBoundSimple = [x.min(), min_amplitude, min_width, 0.0, -5]
    rightBoundSimple = [x.max(), y.max() * 1.5, xlength, 1.0, y.max()]

    # Initial guesses for peak parameters
    amplitudes, centers, widths = cf._find_peak_parameters(x, y, prominence, height=noise, width=3/resolution)
    # Remove initial guesses that are too narrow or too low amplitude
    keep = np.where((widths > min_width) & (amplitudes > min_amplitude))
    amplitudes = amplitudes[keep]
    centers = centers[keep]
    widths = widths[keep]

    peakAmount = len(amplitudes)
    shapes = np.array([1.0] * peakAmount)
    baselevels = np.array([0.0] * peakAmount)

    initvalues = np.concatenate((centers, amplitudes, widths, shapes, baselevels))

    # Number of fit parameters per curve: 5 for fitted baselevel, 4 for fixed baselevel
    parameters = 5
    # Remove bounds if baseline is fixed at 0
    if baseline0:
        leftBoundSimple = leftBoundSimple[:-1]
        rightBoundSimple = rightBoundSimple[:-1]
        initvalues = initvalues[:-peakAmount]
        parameters = 4

    def sumGaussLorentz_reshaped(
        x, params, peakAmount, baseline_fixed=baseline0, baseline=0.0
    ):
        "Reshape parameters to use sum_GaussLorentz in least-squares regression"

        if baseline_fixed:
            baselevels = np.array([baseline] * peakAmount)
            params = np.concatenate((params, baselevels))

        values = params.reshape((5, peakAmount))

        return c.sum_GaussLorentz(x, *values)

    # Residual on ititial fit
    residual_old = mean_squared_error(y, sumGaussLorentz_reshaped(x, initvalues, peakAmount), squared=False)#(y - sumGaussLorentz_reshaped(x, initvalues, peakAmount)).std()
    # Save the initial values in case the first iteration doesn't give an imporovement
    fitParams_old = initvalues.reshape((parameters, peakAmount))
    if baseline0:
        fitParams_old = np.vstack((fitParams_old, np.array([0.0] * peakAmount)))

    # Cost function to minimise
    calculate_residuals = (
        lambda params, x, y, peakAmount: sumGaussLorentz_reshaped(x, params, peakAmount)
        - y
    )

    # Flags for stopping the while loop
    for _ in range(max_iterations):

        # Set up bounds
        leftBound = np.repeat(leftBoundSimple, peakAmount)
        rightBound = np.repeat(rightBoundSimple, peakAmount)
        bounds = (leftBound, rightBound)
        # Optimise fit parameters
        LSfit = opt.least_squares(
            calculate_residuals,
            x0=initvalues,
            bounds=bounds,
            args=(x, y, peakAmount),
            loss="linear",
        )
        # Fitted parameters for sum_GaussLorentz
        fitParams = LSfit.x.reshape((parameters, peakAmount))
        if baseline0:
            fitParams = np.vstack((fitParams, np.array([0.0] * peakAmount)))

        # R squared adjusted for noise
        R2_noise = (y, c.sum_GaussLorentz(x, *fitParams), noise)
        # data_mean = y.mean()
        # residual_vector = y - c.sum_GaussLorentz(x, *fitParams)
        # residual_sum = sum((residual_vector / noise) ** 2)
        # sum_squares = sum((y - data_mean) ** 2)
        # R2_noise = 1 - (residual_sum / sum_squares)

        # RSME on the fit
        residual = mean_squared_error(y, c.sum_GaussLorentz(x,*fitParams), squared=False)#(y - c.sum_GaussLorentz(x, *fitParams)).std()
        residual_vector = abs(y - c.sum_GaussLorentz(x, *fitParams))
        # Stop if noise has reduced less than 10%
        if (residual_old * residuals_threshold) < residual:
            # warnings.warn(f"Noise improved by <{(1 - residuals_threshold):.0%}, using previous result")
            # Revert back to previous fitted values
            fitParams = fitParams_old.copy()
            residual = residual_old
            break
        # Stop if noise on the fit is below the set threshold
        # if residual < (noise * noise_threshold):
        #     print("noise threshold reached!")
        #     break

        # Add new peak
        peakAmount += 1
        # Get initial guess for new peak
        # Y at the highest residual, or the set mimumum ampltude, whichever one is higher
        amplitude = np.max((y[residual_vector == residual_vector.max()][0], min_amplitude))
        center = x[residual_vector == residual_vector.max()][0]
        width = np.max((widths.mean(), min_width))

        amplitudes = np.append(amplitudes, amplitude)
        centers = np.append(centers, center)
        widths = np.append(widths, width)
        shapes = np.append(shapes, 1)
        baselevels = np.append(baselevels, 0)

        initvalues = np.concatenate((centers, amplitudes, widths, shapes, baselevels))

        if baseline0:
            initvalues = initvalues[:-peakAmount]
        # Save old noise and fitted parameters for comparison in next iteration.
        fitParams_old = fitParams.copy()
        residual_old = residual.copy()

    return fitParams, residual

def _R2_noise(y, y_predict, noise):
    data_mean = y.mean()
    residual_vector = y - y_predict
    residual_sum = sum((residual_vector / noise) ** 2)
    sum_squares = sum((y - data_mean) ** 2)
    R2_noise = 1 - (residual_sum / sum_squares)
    return R2_noise
