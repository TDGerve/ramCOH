from . import general as ram
import numpy as np
from warnings import warn
from ..signal_processing import functions as f
from ..signal_processing import curve_fitting as cf
from ..signal_processing import curves as c
import csaps as cs
import scipy.optimize as opt


class H2O(ram.RamanProcessing):
    # Baseline regions
    birs = np.array([[20, 150], [640, 655], [800, 810], [1220, 2800], [3850, 4000]])

    def __init__(self, x, intensity):

        super().__init__(x, intensity)
        self.LC = False
        self.interpolated = False

    def longCorrect(self, T_C=23.0, laser=532.18, normalisation="area", **kwargs):

        y = kwargs.get("y", self.spectrumSelect)
        spectrum = self.intensities[y]

        if self.BC:
            warn(
                "Run baseline correction again to to subtract baseline from Long corrected spectrum"
            )

        self.intensities["long"] = f.long_correction(
            self.x, spectrum, T_C, laser, normalisation
        )
        # self.LC = 1
        self.spectrumSelect = "long"
        self.LC = True

    def interpolate(
        self, interpolate, smooth=1e-6, **kwargs
    ):
        birs = kwargs(interpolate, ram.olivine.birs)
        y = kwargs.get("y", self.spectrumSelect)
        spectrum = self.intensities[y]

        xbir, ybir = f._extractBIR(self.x, spectrum, birs)

        # Boolean array for glass only regions; no olivine peaks
        for i, region in enumerate(birs):
            if i == 0:
                glassIndex = (self.x > region[0]) & (self.x < region[1])
            else:
                glassIndex = glassIndex | ((self.x > region[0]) & (self.x < region[1]))
        # regions with olivine peaks
        interpolate_index = ~glassIndex

        # Fit spline to olivine free regions of the spectrum
        spline = cs.csaps(xbir, ybir, smooth=smooth)
        self.spectrumSpline = spline(self.x)
        # Interpolated residual
        self.interpolated = spectrum - self.spectrumSpline

        # only replace interpolated parts of the spectrum
        self.intensities["interpolated"] = spectrum.copy()
        self.intensities["interpolated"][interpolate_index] = self.spectrumSpline[interpolate_index]

        # Area of olivine spectrum
        self.olivineArea = np.trapz(self.olivine[interpolate_index], self.x[interpolate_index])

        self.spectrumSelect = "interpolated"
        self.interplated = True

    def olivineExtract(self, cutoff=1400, peak_prominence=50, smooth=1e-6, **kwargs):

        birs = kwargs.setdefault("birs", ram.olivine.birs)
        y = kwargs.get("y", self.spectrumSelect)
        spectrum = self.intensities[y]

        xbir, ybir = f._extractBIR(self.x, spectrum, birs)

        # fit spline to olivine free regions of the spectrum
        spline = cs.csaps(xbir, ybir, smooth=smooth)
        spectrumSpline = spline(self.x)
        self.olivine = spectrum - spectrumSpline

        # Remove part of the spectrum with no olivine peaks
        olivine = self.olivine[self.x < cutoff]
        x = self.x[self.x < cutoff]

        # Get initial guesses for olivine peaks
        amplitudes, centers, widths = cf._find_peak_parameters(
            x, olivine, prominence=peak_prominence / 100 * olivine.max()
        )

        peakAmount = len(centers)

        # baselevels = [0] * peakAmount
        shapes = [0.5] * peakAmount

        init_values = np.concatenate([centers, amplitudes, widths, shapes])

        # Set boundary conditions: center, amplitude, width, shape
        leftBoundSimple = [x.min(), 0, 0, 0]
        rightBoundSimple = [x.max(), olivine.max() * 2, (x.max() - x.min()), 1]

        leftBound = np.repeat(leftBoundSimple, peakAmount)        
        rightBound = np.repeat(rightBoundSimple, peakAmount)

        bounds = (leftBound, rightBound)

        def sumGaussians_reshaped(x, params, peakAmount, baselevel=0):
            "Reshape parameters to use sum_GaussLorentz in least-squares regression"

            baselevels = np.array([baselevel] * peakAmount)
            params = np.concatenate((params, baselevels))

            values = params.reshape((5, peakAmount))

            return c.sum_GaussLorentz(x, *values)

        # Fit peaks
        residuals = (
            lambda params, x, peakAmount, spectrum: sumGaussians_reshaped(
                x, params, peakAmount, baselevel=0
            )
            - spectrum
        )

        LSfit = opt.least_squares(
            fun=residuals, x0=init_values, bounds=bounds, args=(x, peakAmount, olivine)
        )

        fitParams = LSfit.x.reshape((4, peakAmount))

        self.olivinePeaks = [
            {"center": i, "amplitude": j, "width": k, "shape": l}
            for _, (i, j, k, l) in enumerate(zip(*fitParams))
        ]

    def SiH2Oareas(self, **kwargs):

        y = kwargs.get("y", self.spectrumSelect)
        spectrum = self.intensities[y]
        self.SiArea = np.trapz(
            spectrum[(self.x > 150) & (self.x < 1400)],
            self.x[(self.x > 150) & (self.x < 1400)],
        )
        self.H2Oarea = np.trapz(
            spectrum[(self.x > 2800) & (self.x < 3900)],
            self.x[(self.x > 2800) & (self.x < 3900)],
        )
