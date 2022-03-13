from . import functions as f
import numpy as np
from warnings import warn
from scipy import signal
from scipy.optimize import least_squares
from itertools import compress
from csaps import csaps


class RamanProcessing:
    def __init__(self, x, intensity):
        self.intensities = {"raw": np.array(intensity)[np.argsort(x)]}
        self.x = np.array(x)[np.argsort(x)]
        # flag to check if baseline correction has been used
        self.BC = False
        # flag to check if normalisation has been used
        self.norm = False
        # flag to check if smoothing has been used
        self.smoothing = False
        self.spectrumSelect = "raw"

    def smooth(self, smoothType="gaussian", kernelWidth=9, **kwargs):
        """
        Smoothing by either a moving average or with a gaussian kernel.
        Be careful, each application shortens the spectrum by one kernel width
        """

        y = kwargs.get("y", self.spectrumSelect)
        spectrum = self.intensities[y]

        self.intensities["smooth"] = f.smooth(spectrum, smoothType, kernelWidth)
        # match length of x with length of smoothed intensities
        self.x = self.x[(kernelWidth - 1) // 2 : -(kernelWidth - 1) // 2]
        # do the same for any other pre-existing spectra
        for i, j in self.intensities.items():
            if not len(j) == len(self.x):
                self.intensities[i] = self.intensities[i][
                    (kernelWidth - 1) // 2 : -(kernelWidth - 1) // 2
                ]

        self.smoothing = True
        self.spectrumSelect = "smooth"

    def baselineCorrect(self, birs, smooth=1e-6, **kwargs):
        """
        Baseline correction with fitted natural smoothing splines from csaps

        birs: n x 2 array for n interpolation regions, where each row is [lower_limit, upper_limit]
        smooth: smoothing factor in range [0,1]
        """
        y = kwargs.get("y", self.spectrumSelect)
        spectrum = self.intensities[y]

        if self.norm:
            warn("run normalisation again to normalise baseline corrected spectrum")

        xbir, ybir = f._extractBIR(self.x, spectrum, birs)

        spline = csaps(xbir, ybir, smooth=smooth)
        self.baseline = spline(self.x)
        self.intensities["BC"] = spectrum - self.baseline

        self.BC = True
        # self.spectrumSelect = intensityDict[self.BC + self.norm]
        self.spectrumSelect = "BC"

    def normalise(self, **kwargs):

        y = kwargs.get("y", self.spectrumSelect)
        spectrum = self.intensities[y]

        # normalisation to maximum intensity
        self.intensities["norm"] = spectrum * 100 / spectrum.max()
        self.norm = True
        # self.spectrumSelect = intensityDict[self.BC + self.norm]
        self.spectrumSelect = "norm"

    def fitPeaks(self, peak_prominence=3, fit_window=12, curve="GL", **kwargs):

        y = kwargs.get("y", self.spectrumSelect)
        spectrum = self.intensities[y]
        self.peaks = {}
        self.curve = curve
        curveDict = {"GL": f.GaussLorentz, "G": f.gaussian, "L": f.lorentzian}

        residuals = lambda params, x, spectrum: curveDict[curve](x, *params) - spectrum

        amplitudes, centers, widths = f._find_peak_parameters(
            x=self.x, y=spectrum, prominence=peak_prominence
        )

        # Gaussian - Lorentian mixing paramter
        shape = 0.5
        # baselevel should be 0 for baseline corrected spectra
        baselevel = 0

        for i, _ in enumerate(amplitudes):
            trimBool = (self.x > (centers[i] - widths[i] * fit_window)) & (
                self.x < (centers[i] + widths[i] * fit_window)
            )
            xTrim = self.x[trimBool]
            intensityTrim = spectrum[trimBool]

            init_values = [amplitudes[i], centers[i], widths[i], baselevel]
            bounds = (-np.inf, np.inf)
            if curve == "GL":
                init_values.append(shape)
                bounds = (
                    [-np.inf, -np.inf, -np.inf, -np.inf, 0],
                    [np.inf, np.inf, np.inf, np.inf, 1],
                )

            fitParams = least_squares(
                fun=residuals,
                x0=init_values,
                bounds=bounds,
                args=(xTrim, intensityTrim),
            ).x

            params = ["amplitude", "center", "width", "baselevel"]
            if curve == "GL":
                params.append("shape")

            self.peaks[i] = {k: fitParams[j] for j, k in enumerate(params)}


class neon(RamanProcessing):
    def neonCorrection(
        self, left_nm=565.666, right_nm=574.83, laser=532.18, search_window=6
    ):

        if not hasattr(self, "peaks"):
            raise NameError("peaks not found, run fitPeaks first")

        neonEmission = f.neonEmission(laser=laser)
        left = np.round(
            np.float(
                neonEmission.iloc[:, 4][
                    np.isclose(left_nm, neonEmission.iloc[:, 1], atol=0.001)
                ]
            ),
            2,
        )
        right = np.round(
            np.float(
                neonEmission.iloc[:, 4][
                    np.isclose(right_nm, neonEmission.iloc[:, 1], atol=0.001)
                ]
            ),
            2,
        )

        # All emission lines within spectrum limits
        neonEmissionTrim = np.array(
            neonEmission.iloc[:, 4][
                (neonEmission.iloc[:, 4] > self.x.min())
                & (neonEmission.iloc[:, 4] < self.x.max())
            ]
        )

        # theoretical emission line positions with a match found in spectrum
        self.peakEmission = np.array([])
        # measured emission line positions
        self.peakMeasured = np.array([])

        for i, j in self.peaks.items():
            peak = j["center"]
            emissionCheck = np.isclose(peak, neonEmissionTrim, atol=search_window)

            if emissionCheck.sum() == 1:
                self.peakEmission = np.append(
                    self.peakEmission, neonEmissionTrim[emissionCheck].tolist()
                )
                self.peakMeasured = np.append(self.peakMeasured, peak)
            elif emissionCheck.sum() > 1:
                print(
                    "multiple emission line fits foundfor peak: "
                    + str(round(peak, 2))
                    + " cm-1"
                )

        # find correction factor for the calibration lines
        if np.isin([left, right], np.round(self.peakEmission, 2)).sum() == 2:
            # boolean array for the location of left and right calibration lines
            calibration_lines = np.array(
                np.isclose(left, self.peakEmission, atol=0.001)
                + np.isclose(right, self.peakEmission, atol=0.001)
            )
            # boolean to index
            calibration_lines = list(
                compress(range(len(calibration_lines)), calibration_lines)
            )
            # indices for differenced array
            calibration_lines = np.unique(calibration_lines - np.array([0, 1]))

            self.correctionFactor = np.float(
                np.sum(np.diff(self.peakEmission)[calibration_lines])
                / np.sum(np.diff(self.peakMeasured)[calibration_lines])
            )
            self.offset = np.float(
                self.peakMeasured[np.isclose(left, self.peakMeasured, atol=10)]
                - self.peakEmission[np.isclose(left, self.peakEmission, atol=0.1)]
            )

        else:
            print("calibration lines not found in spectrum")


class CO2(RamanProcessing):
    def FermiDiad(self, peak_prominence=40, fit_window=8, **kwargs):

        y = kwargs.get("y", self.spectrumSelect)
        spectrum = self.intensities[y]
        self.diad = {}

        # fit parameters for diad
        self.diad["fit_params1"], self.diad["fit_params2"] = f.diad(
            x=self.x,
            intensities=spectrum,
            peak_prominence=peak_prominence,
            fit_window=fit_window,
        )

        # diad curves
        self.diad["peak1"] = {
            "x": self.diad["fit_params1"]["x"],
            "y": f.GaussLorentz(**self.diad["fit_params1"]),
        }
        self.diad["peak2"] = {
            "x": self.diad["fit_params2"]["x"],
            "y": f.GaussLorentz(**self.diad["fit_params2"]),
        }
        del self.diad["fit_params1"]["x"]
        del self.diad["fit_params2"]["x"]
        # diad split
        self.diad["split"] = abs(
            self.diad["fit_params1"]["center"] - self.diad["fit_params2"]["center"]
        )


class H2O(RamanProcessing):
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
        spline = csaps(xbir, ybir, smooth=smooth)
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

        defaultBir = np.array(
            [
                [100, 270],
                [360, 380],
                [460, 515],
                [555, 560],
                [660, 700],
                [900, 910],
                [930, 940],
                [990, 4000],
            ]
        )

        birs = kwargs.setdefault("birs", defaultBir)
        y = kwargs.get("y", self.spectrumSelect)
        spectrum = self.intensities[y]

        # regions without olivine peaks
        if isinstance(birs, list):
            birs = np.array(birs).reshape((len(birs) // 2, 2))
        xbir, ybir = f._extractBIR(self.x, spectrum, birs)

        # fit spline to olivine free regions of the spectrum
        spline = csaps(xbir, ybir, smooth=smooth)
        self.spectrumSpline = spline(self.x)
        self.olivine = spectrum - self.spectrumSpline

        # Remove part of the spectrum with no olivine peaks
        olivine = self.olivine[self.x < cutoff]
        x = self.x[self.x < cutoff]

        # Get initial guesses
        amplitudes, centers, widths = f._find_peak_parameters(
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

        def sum_GaussLorentz_reshape(x, params, peakAmount):
            "Reshape parameters to use sum_GaussLorentz in least-squares regression"

            values = params.reshape((4, peakAmount))

            return f.sum_GaussLorentz(x, *values)

        # Fit peaks
        residuals = (
            lambda params, x, peakAmount, spectrum: sum_GaussLorentz_reshape(
                x, params, peakAmount, baselevel=0
            )
            - spectrum
        )

        LSfit = least_squares(
            fun=residuals, x0=init_values, bounds=bounds, args=(x, peakAmount, olivine)
        )

        fitParams = LSfit.x.reshape((4, peakAmount))

        self.olivinePeaksFitted = [
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


class olivine(H2O):
    def __init__(self, x, intensity):

        super().__init__(x, intensity)

    def deconvolve(
        self,
        peak_prominence=2,
        fit_window=4,
        noise_threshold=1.4,
        baseline0=False,
        max_iterations=15,
        cutoff=1400,
        **kwargs
    ):

        y = kwargs.get("y", self.spectrumSelect)
        spectrum = self.intensities[y][self.x < cutoff]
        x = self.x[self.x < cutoff]

        _, centers, widths = f._find_peak_parameters(
            x=x, y=spectrum, prominence=peak_prominence
        )
        ranges = f._get_peakFit_ranges(
            centers=centers, half_widths=widths, fit_window=fit_window
        )

        fitted_parameters = []
        for range in ranges:
            xtrim, ytrim = f._trimxy_ranges(x, spectrum, range)
            parameters, *_ = f.deconvolve_curve(
                x=xtrim,
                y=ytrim,
                noise_threshold=noise_threshold,
                prominence=peak_prominence,
                baseline0=baseline0,
                max_iterations=max_iterations,
            )
            fitted_parameters.append(parameters)

        self.deconvolution_parameters = []
        for parameter in zip(*fitted_parameters):
            self.deconvolution_parameters.append(np.concatenate(parameter))

        self.peaks = [
            {"center": i, "amplitude": j, "width": k, "shape": l, "baselevel": m}
            for _, (i, j, k, l, m) in enumerate(zip(*self.deconvolution_parameters))
        ]

        # # Get initial guesses
        # peaks = signal.find_peaks(spectrum, prominence=peak_prominence)

        # amplitudes, centers = spectrum[peaks[0]], x[peaks[0]]
        # widths = signal.peak_widths(spectrum, peaks[0])[0] * abs(np.diff(x).mean())

        # peakAmount = len(centers)
        # shapes = [0.5] * peakAmount

        # init_values = np.concatenate([centers, amplitudes, widths, shapes])

        # # Set boundary conditions
        # leftBoundSimple = [-np.inf, 0, 0, 0]
        # leftBound = np.repeat(leftBoundSimple, peakAmount)

        # rightBoundSimple = [np.inf, np.inf, np.inf, 1]
        # rightBound = np.repeat(rightBoundSimple, peakAmount)

        # bounds = (leftBound, rightBound)

        # def sum_GaussLorentz_reshape(x, params, peakAmount):
        #     "Reshape parameters to use sum_GaussLorentz in least-squares regression"

        #     values = params.reshape((4, peakAmount))

        #     return f.sum_GaussLorentz(x, *values)

        # # Least cost function
        # residuals = (
        #     lambda params, x, peakAmount, spectrum: sum_GaussLorentz_reshape(
        #         x, params, peakAmount, baselevel=0
        #     )
        #     - spectrum
        # )

        # LSfit = least_squares(
        #     fun=residuals, x0=init_values, bounds=bounds, args=(x, peakAmount, spectrum)
        # )

        # fitParams = LSfit.x.reshape((4, peakAmount))

        # self.peaksFitted = [
        #     {"center": i, "amplitude": j, "width": k, "shape": l}
        #     for _, (i, j, k, l) in enumerate(zip(*fitParams))
        # ]
