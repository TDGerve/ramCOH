import numpy as np
import csaps as cs
import scipy.optimize as opt
from warnings import warn

from ..signal_processing import functions as f
from ..signal_processing import curve_fitting as cf
from ..signal_processing import curves as c
from ..signal_processing import deconvolution as d
from .baseclass import RamanProcessing
from .olivine import Olivine


class H2O(RamanProcessing):
    # Baseline regions
    Si_birs_default = np.array([[20, 250], [640, 655], [800, 810], [1220, 1600]])
    H2O_boundaries_default = [2800, 3850]
    H2O_birs_default = np.array(
        [[1500, H2O_boundaries_default[0]], [H2O_boundaries_default[1], 4000]]
    )
    birs_default = np.concatenate((Si_birs_default, H2O_birs_default))

    def __init__(self, x, y, **kwargs):

        super().__init__(x, y, **kwargs)
        self._processing.update(
            {"long_corrected": False, "interpolated": False, "olivine_corrected": False}
        )
        self.Si_birs = kwargs.get("Si_birs", self.Si_birs_default)
        self.H2O_boundaries = kwargs.get("H2O_boundaries", self.H2O_boundaries_default)
        H2O_birs = np.array(
            [[1500, min(self.H2O_boundaries)], [max(self.H2O_boundaries), 4000]]
        )
        self.birs = np.concatenate((self.Si_birs, H2O_birs))

    def longCorrect(self, T_C=23.0, normalisation="area", **kwargs):

        laser = kwargs.get("laser", self.laser)

        y = kwargs.get("y", self._spectrumSelect)
        spectrum = getattr(self.signal, y)

        long_corrected = f.long_correction(self.x, spectrum, T_C, laser, normalisation)
        setattr(self.signal, "long_corrected", long_corrected)
        # self.LC = 1
        self._spectrumSelect = "long_corrected"
        self._processing["long_corrected"] = True

    def baselineCorrect(self, **kwargs):

        _ = kwargs.pop("baseline_regions", None)
        Si_birs = kwargs.get("Si_birs", self.Si_birs)
        H2O_boundaries = kwargs.get("H2O_boundaries", self.H2O_boundaries)
        H2O_birs = np.array([[1500, min(H2O_boundaries)], [max(H2O_boundaries), 4000]])
        baseline_regions = np.concatenate((Si_birs, H2O_birs))

        return super().baselineCorrect(baseline_regions=baseline_regions, **kwargs)

    def interpolate(self, *, interpolate=[780, 900], smooth=1e-6, **kwargs):

        birs = np.array(
            [[self.x.min(), min(interpolate)], [max(interpolate), self.x.max()]]
        )

        y = kwargs.get("y", self._spectrumSelect)
        spectrum = getattr(self.signal, y)

        xbir, ybir = f._extractBIR(self.x, spectrum, birs)

        # Boolean array for glass only regions; no interference peaks
        for i, region in enumerate(birs):
            if i == 0:
                glassIndex = (self.x > region[0]) & (self.x < region[1])
            else:
                glassIndex = glassIndex | ((self.x > region[0]) & (self.x < region[1]))
        # regions with interference peaks
        interpolate_index = ~glassIndex

        spline = cs.csaps(xbir, ybir, smooth=smooth)
        self.spectrum_spline = spline(self.x)
        # Interpolated residual
        self.interpolation_residuals = spectrum - self.spectrum_spline

        _, baseline = f._extractBIR(
            self.x[self.x > 350], self.interpolation_residuals[self.x > 350], birs
        )
        noise = baseline.std(axis=None)
        # Add signal noise to the spline
        noise_spline = self.spectrum_spline + np.random.normal(0, noise, len(self.x))

        # only replace interpolated parts of the spectrum
        setattr(self.signal, "interpolated", spectrum.copy())
        self.signal.interpolated[interpolate_index] = noise_spline[interpolate_index]

        # Area of interpolated regions
        self.interpolated_area = np.trapz(
            self.interpolation_residuals[interpolate_index], self.x[interpolate_index]
        )

        self._spectrumSelect = "interpolated"
        self._processing["interpolated"] = True

    def extract_olivine(
        self, olivine_x, olivine_y, *, peak_prominence=10, smooth=1e-6, **kwargs
    ):

        # Set default values
        default_birs = np.array(
            [[0, 780], [900, 4000]]
        )  # np.array([[0, 250], [460, 550], [650, 720], [1035, 4000]])
        birs = kwargs.get("birs", default_birs)
        fit_window = kwargs.get("fit_window", 6)
        noise_threshold = kwargs.get("noise_threshold", 1.5)
        threshold_scale = kwargs.get("threshold_scale", 0.0)
        cutoff_high = 1100
        cutoff_low = 700

        y = kwargs.get("y", self._spectrumSelect)
        spectrum = getattr(self.signal, y)

        xbir, ybir = f._extractBIR(self.x, spectrum, birs)

        # fit spline to olivine free regions of the spectrum
        spline = cs.csaps(xbir, ybir, smooth=smooth)
        spectrum_spline = spline(self.x)
        self.spectrum_spline = spectrum_spline.copy()
        # Signal attributed to olivine interference
        olivine_interference = spectrum - spectrum_spline
        self.olivine_interference = olivine_interference.copy()

        # Calculate noise level
        noise_area = (self.x > 1250) & (self.x < 2000)
        noise = olivine_interference[noise_area].std(axis=None)

        # Remove part of the spectrum with no major olivine peaks
        trim = (self.x > cutoff_low) & (self.x < cutoff_high)
        olivine_trim = olivine_interference[trim]
        x = self.x[trim]

        # Deconvolute the major olivine peaks
        olivine_fit = RamanProcessing(x, olivine_trim)
        # print("fitting interference")ol
        olivine_fit.deconvolve(
            peak_prominence=peak_prominence,
            noise_threshold=1.5,
            threshold_scale=0.0,
            min_amplitude=6,
            min_peak_width=6,
            fit_window=6,
            noise=noise,
            max_iterations=3,
        )

        olivine_main_peaks = olivine_fit.deconvolution_parameters

        # Deconvolute host crystal spectrum
        olivine = Olivine(olivine_x, olivine_y)
        olivine.baselineCorrect()
        olivine.calculate_noise()
        # print("fitting host")
        olivine.deconvolve(
            y="baseline_corrected",
            fit_window=fit_window,
            noise_threshold=noise_threshold,
            threshold_scale=threshold_scale,
        )

        stepsize = abs(np.diff(self.x).mean())
        x_olivine_peaks = np.arange(700, 1200, int(stepsize * 10))
        interference_max = c.sum_GaussLorentz(
            x_olivine_peaks, *olivine_fit.deconvolution_parameters
        ).max()
        host_max = c.sum_GaussLorentz(
            x_olivine_peaks, *olivine.deconvolution_parameters
        ).max()
        self.olivine_scale = host_max / interference_max

        self.olivine = c.sum_GaussLorentz(self.x, *olivine.deconvolution_parameters)

        olivine_corrected = spectrum - (self.olivine / self.olivine_scale)
        setattr(self.signal, "olivine_corrected", olivine_corrected)
        self._spectrumSelect = "olivine_corrected"
        self._processing["olivine_corrected"] = True

        self.olivinePeaks = [
            {"center": i, "amplitude": j, "width": k, "shape": l, "baselevel": m}
            for _, (i, j, k, l, m) in enumerate(zip(*olivine_main_peaks))
        ]

    def calculate_SiH2Oareas(self, **kwargs):

        if not hasattr(self.signal, "baseline_corrected"):
            raise RuntimeError("run baseline correction first")

        water_left = self.birs[-2][1]
        water_right = self.birs[-1][0]

        spectrum = getattr(self.signal, "baseline_corrected")
        SiArea = np.trapz(
            spectrum[(self.x > 150) & (self.x < 1400)],
            self.x[(self.x > 150) & (self.x < 1400)],
        )
        H2Oarea = np.trapz(
            spectrum[(self.x > water_left) & (self.x < water_right)],
            self.x[(self.x > water_left) & (self.x < water_right)],
        )
        self.SiH2Oareas = SiArea, H2Oarea
