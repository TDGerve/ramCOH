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
    birs = np.array([[20, 250], [640, 655], [800, 810], [1220, 2800], [3750, 4000]])

    def __init__(self, x, y):

        super().__init__(x, y)
        self.processing.update(
            {"long_corrected": False, "interpolated": False, "olivine_corrected": False}
        )

    def longCorrect(self, T_C=23.0, normalisation="area", **kwargs):

        laser = kwargs.get("laser", self.laser)

        y = kwargs.get("y", self.spectrumSelect)
        spectrum = getattr(self.signal, y)

        long_corrected = f.long_correction(self.x, spectrum, T_C, laser, normalisation)
        setattr(self.signal, "long_corrected", long_corrected)
        # self.LC = 1
        self.spectrumSelect = "long_corrected"
        self.processing["long_corrected"] = True

    def interpolate(self, *, interpolate=[780, 900], smooth=1e-6, **kwargs):

        birs = np.array(
            [[self.x.min(), min(interpolate)], [max(interpolate), self.x.max()]]
        )

        y = kwargs.get("y", self.spectrumSelect)
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

        # Fit spline to baseline regions

        ##########
        # DON'T FIT THE SPLINE ON THE FIRST 250 RAMAN SHIFTS
        #########
        
        spline = cs.csaps(xbir, ybir, smooth=smooth)
        self.spectrum_spline = spline(self.x)
        # Interpolated residual
        self.interpolation_residuals = spectrum - self.spectrum_spline

        _, baseline = f._extractBIR(self.x[self.x > 350], self.interpolation_residuals[self.x > 350], birs)
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

        self.spectrumSelect = "interpolated"
        self.processing["interplated"] = True

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

        y = kwargs.get("y", self.spectrumSelect)
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
        olivine_fit = ram.RamanProcessing(x, olivine_trim)
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
        olivine = ram.olivine(olivine_x, olivine_y)
        olivine.baselineCorrect()
        olivine.calculate_noise()
        # print("fitting host")
        olivine.deconvolve(
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
        self.spectrumSelect = "olivine_corrected"
        self.processing["olivine_corrected"] = True

        self.olivinePeaks = [
            {"center": i, "amplitude": j, "width": k, "shape": l, "baselevel": m}
            for _, (i, j, k, l, m) in enumerate(zip(*olivine_main_peaks))
        ]

    def calculate_SiH2Oareas(self, **kwargs):

        if not hasattr(self.signal, "baseline_corrected"):
            raise RuntimeError("run baseline correction first")

        spectrum = getattr(self.signal, "baseline_corrected")
        SiArea = np.trapz(
            spectrum[(self.x > 150) & (self.x < 1400)],
            self.x[(self.x > 150) & (self.x < 1400)],
        )
        H2Oarea = np.trapz(
            spectrum[(self.x > 3000) & (self.x < 3900)],
            self.x[(self.x > 3000) & (self.x < 3900)],
        )
        self.SiH2Oareas = SiArea, H2Oarea
