from ..signal_processing import functions as f
import numpy as np
from warnings import warn
from scipy import signal
import scipy.optimize as opt
from csaps import csaps
from ..signal_processing import curves as c
from ..signal_processing import curve_fitting as cf
from ..signal_processing import deconvolution as d


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

    def smooth(self, smoothType="Gaussian", kernelWidth=9, **kwargs):
        """
        Smoothing by either a moving average or with a Gaussian kernel.
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

    def baselineCorrect(self, baseline_regions=None, smooth_factor=1, **kwargs):
        """
        Baseline correction with fitted natural smoothing splines from csaps

        birs: n x 2 array for n interpolation regions, where each row is [lower_limit, upper_limit]
        smooth: smoothing factor in range [0,1]
        """
        y = kwargs.get("y", self.spectrumSelect)
        spectrum = self.intensities[y]

        if (hasattr(self, "birs")) & (baseline_regions is None):
            baseline_regions = self.birs

        if self.norm:
            warn("run normalisation again to normalise baseline corrected spectrum")

        xbir, ybir = f._extractBIR(self.x, spectrum, baseline_regions)

        max_difference = ybir.max() - ybir.min()
        smooth = 2e-9 * max_difference * smooth_factor

        spline = csaps(xbir, ybir, smooth=smooth)
        self.baseline = spline(self.x)
        self.intensities["BC"] = spectrum - self.baseline

        self.BC = True
        # self.spectrumSelect = intensityDict[self.BC + self.norm]
        self.spectrumSelect = "BC"

    def calculate_noise(self, baseline_regions=None, smooth_factor=1e-5):

        if (hasattr(self, "birs")) & (baseline_regions is None):
            baseline_regions = self.birs

        xbir, ybir = f._extractBIR(self.x, self.intensities["BC"], baseline_regions)
        self.noise, _ = f._calculate_noise(xbir, ybir, smooth_factor=smooth_factor)

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
        curveDict = {"GL": c.GaussLorentz, "G": c.Gaussian, "L": c.Lorentzian}

        residuals = lambda params, x, spectrum: curveDict[curve](x, *params) - spectrum

        amplitudes, centers, widths = cf._find_peak_parameters(
            x=self.x, y=spectrum, prominence=peak_prominence
        )

        # Gaussian - Lorentian mixing paramter
        shape = 0.5
        # baselevel should be 0 for baseline corrected spectra
        baselevel = 0

        for i, (amplitude, center, width) in enumerate(
            zip(amplitudes, centers, widths)
        ):
            x_range = cf._get_peakFit_ranges(center, width, fit_window)
            xtrim, ytrim = cf._trimxy_ranges(self.x, spectrum, x_range)
            # trimBool = (self.x > (centers[i] - widths[i] * fit_window)) & (
            #     self.x < (centers[i] + widths[i] * fit_window)
            # )
            # xTrim = self.x[trimBool]
            # intensityTrim = spectrum[trimBool]

            init_values = [amplitude, center, width, baselevel]
            bounds = (-np.inf, np.inf)
            if curve == "GL":
                init_values.append(shape)
                bounds = (
                    [-np.inf, -np.inf, -np.inf, -np.inf, 0],
                    [np.inf, np.inf, np.inf, np.inf, 1],
                )

            fitParams = opt.least_squares(
                fun=residuals,
                x0=init_values,
                bounds=bounds,
                args=(xtrim, ytrim),
            ).x

            params = ["amplitude", "center", "width", "baselevel"]
            if curve == "GL":
                params.append("shape")

            self.peaks[i] = {k: fitParams[j] for j, k in enumerate(params)}

    def deconvolve(
        self,
        peak_prominence=3,
        noise_threshold=1.8,
        threshold_scale=0.1,
        baseline0=True,
        min_amplitude=8,
        min_peak_width=6,
        fit_window=4,
        noise=None,
        max_iterations=5,
        print_output=False,
        **kwargs,
    ):

        y = kwargs.get("y", self.spectrumSelect)
        spectrum = self.intensities[y]
        x = self.x

        if "cutoff" in kwargs:
            cutoff = kwargs.get("cutoff")
            spectrum = spectrum[x < cutoff]
            x = x[x < cutoff]

        _, centers, widths = cf._find_peak_parameters(
            x=x, y=spectrum, prominence=peak_prominence
        )
        ranges = cf._get_peakFit_ranges(
            centers=centers, half_widths=widths, fit_window=fit_window
        )

        # Scale the noise threshold based on max y
        threshold_scaler = (
            lambda y: (
                (y / spectrum.max() * 2 * threshold_scale) + (1 - threshold_scale)
            )
            * noise_threshold
        )

        fitted_parameters = []
        for range in ranges:
            xtrim, ytrim = cf._trimxy_ranges(x, spectrum, range)
            noise_threshold_local = threshold_scaler(ytrim.max())
            if print_output:
                print(
                    f"max y: {ytrim.max()}, range {range}, threshold: {noise_threshold_local}"
                )
            try:
                parameters, *_ = d.deconvolve_signal(
                    x=xtrim,
                    y=ytrim,
                    noise_threshold=noise_threshold_local,
                    baseline0=baseline0,
                    min_peak_width=min_peak_width,
                    min_amplitude=min_amplitude,
                    noise=noise,
                    max_iterations=max_iterations,
                )
                fitted_parameters.append(parameters)
            except:
                warn(f"range {range} skipped.")
                continue

        self.deconvolution_parameters = []
        for parameter in zip(*fitted_parameters):
            self.deconvolution_parameters.append(np.concatenate(parameter))

        self.deconvoluted_peaks = [
            {"center": i, "amplitude": j, "width": k, "shape": l, "baselevel": m}
            for _, (i, j, k, l, m) in enumerate(zip(*self.deconvolution_parameters))
        ]










