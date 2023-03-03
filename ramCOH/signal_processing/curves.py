"""
======
Curves
======
Curve shape functions

"""

import numpy as np


def Gaussian(x, amplitude, center, width, baselevel=0):
    """
    Gaussian curve

    .. math::
            f(x) = A * exp \left( - \\frac{(x - b)^{2}}{2c^2} \\right) + d


    where:
        * :math:`A` = amplitude
        * :math:`b` = center
        * :math:`c` = width (standard deviation)
        * :math:`d` = baselevel


    Parameters
    ----------
    x   :   float or array-like
        x-axis
    amplitude   :   float
        maximum peak height
    center  :   float
        peak center
    width   :   float
        peak width as standard deviation
    baselevel : float, default 0
        peak baselevel


    Returns
    -------
    float, array-like
        Gaussian curve evaluated at x
    """

    return amplitude * np.exp(-((x - center) ** 2) / (2 * width**2)) + baselevel


def Lorentzian(x, amplitude, center, width, baselevel=0):
    """
    Three-parameter Lorentzian curve

    .. math::
            f(x) = A\left(\\frac{c^2}{(x-b)^2 + c^2}\\right) + d


    where:
        * :math:`A` = amplitude
        * :math:`b` = center
        * :math:`c` = width (standard deviation)
        * :math:`d` = baselevel

    Parameters
    ----------
    x   :   float or array-like
        x-axis
    amplitude   :   float
        maximum peak height
    center  :   float
        peak center
    width   :   float
        peak width as half-width at full maximum
    baselevel : float, default 0
        peak baselevel

    Returns
    -------
    float, array-like
        Lorentzian curve evaluated at x
    """

    return amplitude * width**2 / ((x - center) ** 2 + width**2) + baselevel


def GaussLorentz(x, amplitude, center, width, baselevel, shape):
    """
    Mixed Gaussian-Lorentzian (Pseudo-Voigt) curve

    .. math::
            f(x) = Gaussian(x, A * \\alpha, b, c, d) + Lorentzian(x, A * (1 - \\alpha), b, c, d)

    where:
        * :math:`A` = amplitude
        * :math:`b` = center
        * :math:`c` = width (standard deviation)
        * :math:`d` = baselevel
        * :math:`\\alpha` = shape

    with parameterisations from :py:func:`~ramCOH.signal_processing.curves.Gaussian` and :py:func:`~ramCOH.signal_processing.curves.Lorentzian`

    Parameters
    ----------
    x   :   float or array-like
        x-axis
    amplitude   :   float
        maximum peak height
    center  :   float
        peak center
    width   :   float
        peak width as half-width at full maximum
    baselevel : float, default 0
        peak baselevel
    shape   :   float
        mixing parameter between 0 and 1

    Returns
    -------
    float, array-like
        Pseudo-Voigt curve evaluated at x

    """

    return (
        Gaussian(x, amplitude * (1 - shape), center, width, 0)
        + Lorentzian(x, amplitude * shape, center, width, 0)
        + baselevel
    )


def sum_GaussLorentz(x, centers, amplitudes, widths, shapes, baselevels):
    """
    add mixed Gauss-Lorentzian curves together


    :meta private:
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
