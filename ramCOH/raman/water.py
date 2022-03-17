from . import general as ram
import numpy as np
from warnings import warn
from ..signal_processing import functions as f
from ..signal_processing import curve_fitting as cf
from ..signal_processing import curves as c
import csaps as cs
import scipy.optimize as opt

class H2O(ram.RamanProcessing):

    birs = np.array([[20, 150], [640, 655], [800, 810], [1220, 2800], [3850, 4000]])

    def __init__(self, x, intensity):

        super().__init__(x, intensity)
        self.LC = False
        self.OlC = False

    def longCorrect(self, T_C=25.0, laser=532.18, normalisation="area", **kwargs):

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

    def olivineInterpolate(
        self, ol=[0, 780, 902, 905, 932, 938, 980, 4005], smooth=1e-6, **kwargs
    ):

        y = kwargs.get("y", self.spectrumSelect)
        spectrum = self.intensities[y]

        # olivine baseline interpolatin regions
        if isinstance(ol, list):
            olBirs = np.array(ol).reshape((len(ol) // 2, 2))
        xbir, ybir = f._extractBIR(self.x, spectrum, olBirs)

        # Boolean array for glass only regions; no olivine peaks
        for i, region in enumerate(olBirs):
            if i == 0:
                glassIndex = (self.x > region[0]) & (self.x < region[1])
            else:
                glassIndex = glassIndex | ((self.x > region[0]) & (self.x < region[1]))
        # regions with olivine peaks
        olIndex = ~glassIndex

        # Fit spline to olivine free regions of the spectrum
        spline = cs.csaps(xbir, ybir, smooth=smooth)
        self.spectrumSpline = spline(self.x)
        # Olivine residual
        self.olivine = spectrum - self.spectrumSpline

        # only replace interpolated parts of the spectrum
        self.intensities["olC"] = spectrum.copy()
        self.intensities["olC"][olIndex] = self.spectrumSpline[olIndex]

        # Area of olivine spectrum
        self.olivineArea = np.trapz(self.olivine[olIndex], self.x[olIndex])

        self.spectrumSelect = "olC"
        self.olC = True

    def olivineExtract(self, cutoff=1400, peak_prominence=20, smooth=1e-6, **kwargs):


        birs = kwargs.setdefault("birs", olivine.birs)
        y = kwargs.get("y", self.spectrumSelect)
        spectrum = self.intensities[y]

        # regions without olivine peaks
        if isinstance(birs, list):
            birs = np.array(birs).reshape((len(birs) // 2, 2))
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
            self.x, olivine, prominence=peak_prominence / 100 * olivine.max()
        )

        peakAmount = len(centers)

        # baselevels = [0] * peakAmount
        shapes = [0.5] * peakAmount

        init_values = np.concatenate([centers, amplitudes, widths, shapes])

        # Set boundary conditions
        leftBoundSimple = [-np.inf, 0, 0, 0]
        leftBound = np.repeat(leftBoundSimple, peakAmount)

        rightBoundSimple = [np.inf, np.inf, np.inf, 1]
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