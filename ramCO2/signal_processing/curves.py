import numpy as np

def Gaussian(x, amplitude, center, width, baselevel=0):

    return amplitude * np.exp(-((x - center) ** 2) / (2 * width ** 2)) + baselevel


def Lorentzian(x, amplitude, center, width, baselevel=0):

    return amplitude * width ** 2 / ((x - center) ** 2 + width ** 2) + baselevel


def GaussLorentz(x, amplitude, center, width, baselevel, shape):

    return (
        Gaussian(x, amplitude * (1 - shape), center, width, 0)
        + Lorentzian(x, amplitude * shape, center, width, 0)
        + baselevel
    )


def sum_GaussLorentz(x, centers, amplitudes, widths, shapes, baselevels):
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