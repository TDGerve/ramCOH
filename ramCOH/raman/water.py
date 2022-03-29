import numpy as np
import csaps as cs
import scipy.optimize as opt
from warnings import warn
from ..signal_processing import functions as f
from ..signal_processing import curve_fitting as cf
from ..signal_processing import curves as c
from ..signal_processing import deconvolution as d
from .. import raman as ram


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

    def interpolate(self, interpolate, smooth=1e-6, **kwargs):
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
        self.intensities["interpolated"][interpolate_index] = self.spectrumSpline[
            interpolate_index
        ]

        # Area of olivine spectrum
        self.olivineArea = np.trapz(
            self.olivine[interpolate_index], self.x[interpolate_index]
        )

        self.spectrumSelect = "interpolated"
        self.interplated = True

    def extract_olivine(
        self, olivine_x, olivine_y, peak_prominence=6, smooth=1e-6, **kwargs
    ):

        # Set default values
        default_birs = np.array(
            [[0, 250], [460, 550], [650, 720], [1035, 4000]]
        )  # [900, 910],
        birs = kwargs.get("birs", default_birs)
        fit_window = kwargs.get("fit_window", 8)
        noise_threshold = kwargs.get("noise_threshold", 1.5)
        threshold_scale = kwargs.get("threshold_scale", 0.)
        cutoff_high = 1400
        cutoff_low = 700

        y = kwargs.get("y", self.spectrumSelect)
        spectrum = self.intensities[y]

        xbir, ybir = f._extractBIR(self.x, spectrum, birs)

        # fit spline to olivine free regions of the spectrum
        spline = cs.csaps(xbir, ybir, smooth=smooth)
        spectrum_spline = spline(self.x)
        self.spectrum_spline = spectrum_spline
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
        olivine_fit = ram.RamanProcessing(x, olivine_trim)
        olivine_fit.deconvolve(
            peak_prominence=peak_prominence,
            noise_threshold=noise_threshold,
            threshold_scale=threshold_scale,
            min_amplitude=6,
            min_peak_width=6,
            fit_window=fit_window,
            noise=noise,
            max_iterations=3,
        )

        self.olivine_main_peaks = olivine_fit.deconvolution_parameters

        interference_max = c.sum_GaussLorentz(self.x, *self.olivine_main_peaks).max()
        # Deconvolute host crystal spectrum
        olivine = ram.olivine(olivine_x, olivine_y)
        olivine.baselineCorrect()
        olivine.calculate_noise()
        olivine.deconvolve(noise_threshold=noise_threshold, min_amplitude=2)

        main_peaks_interference = np.sort(olivine_fit.deconvolution_parameters[1])[-2:]
        main_peaks_host = np.sort(olivine.deconvolution_parameters[1])[-2:]
        self.olivine_scale = (main_peaks_host / main_peaks_interference).mean()

        self.olivine = c.sum_GaussLorentz(self.x, *olivine.deconvolution_parameters)
        # Scale host crystal specutrum to the interference
        # self.olivine_scale = self.olivine.max() / interference_max
        self.intensities["olivine_corrected"] = spectrum - (
            self.olivine / self.olivine_scale
        )

        self.olivinePeaks = [
            {"center": i, "amplitude": j, "width": k, "shape": l, "baselevel": m}
            for _, (i, j, k, l, m) in enumerate(zip(*self.olivine_main_peaks))
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
