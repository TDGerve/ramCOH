from typing import List, Optional
from warnings import warn

import csaps as cs
import numpy as np
import scipy.optimize as opt
import scipy.signal as sig

from ..signal_processing import curve_fitting as cf
from ..signal_processing import curves as c
from ..signal_processing import deconvolution as d
from ..signal_processing import functions as f


class Signal:
    def __init__(self, x, y):
        self.x = x
        self.raw = y
        self._names: List[str] = ["raw"]

    def add(self, name, values: np.ndarray):
        setattr(self, name, values)
        self._names.append(name)

    def set(self, name, values):
        if name not in self.names:
            raise ValueError(f"{name} not in signals")
        setattr(self, name, values)

    def get(self, name):
        return getattr(self, name)

    @property
    def all(self):
        return {name: getattr(self, name) for name in self.names}

    @property
    def names(self):
        return self._names


class RamanProcessing:
    def __init__(self, x, y, laser=532.18):
        x, y = f.trim_sort(x, y)
        self.x = x
        self.signal = Signal(x, y)
        self.noise: Optional[float] = None
        self.laser = laser
        self._processing = {
            "baseline_corrected": False,
            "normalised": False,
            "smoothed": False,
            "interpolated": False,
            "interference_corrected": False,
        }
        self.birs = None
        self._spectrumSelect = "raw"

    @property
    def processing(self):
        return self._processing

    def smooth(self, smoothType="Gaussian", kernelWidth=9, **kwargs):
        """
        Smoothing by either a moving average or with a Gaussian kernel.
        Each application shortens the spectrum by one kernel width
        """

        y = kwargs.get("y", self._spectrumSelect)
        spectrum = self.signal.get(y)

        smooth = f.smooth(spectrum, smoothType, kernelWidth)
        self.signal.add("smooth", smooth)

        # match length of x with length of smoothed signal
        self.x = self.x[(kernelWidth - 1) // 2 : -(kernelWidth - 1) // 2]
        # do the same for any other pre-existing spectra
        for name, value in self.signal.all.items():
            if name != "smooth":
                shortened = value[(kernelWidth - 1) // 2 : -(kernelWidth - 1) // 2]
                self.signal.set(name, shortened)

        self._processing["smoothed"] = True
        self._spectrumSelect = "smooth"

    def baselineCorrect(self, baseline_regions=None, smooth_factor=1, **kwargs):
        """
        Baseline correction with fitted natural smoothing splines from csaps

        birs: n x 2 array for n interpolation regions, where each row is [lower_limit, upper_limit]
        smooth: smoothing factor in range [0,1]
        """
        y = kwargs.get("y", self._spectrumSelect)
        spectrum = self.signal.get(y)

        if (hasattr(self, "birs_default")) & (baseline_regions is None):
            baseline_regions = self.birs_default

        self.birs = baseline_regions

        if "normalised" in self.signal.names:
            warn("run normalisation again to normalise baseline corrected spectrum")

        xbir, ybir = f._extractBIR(self.x, spectrum, baseline_regions)

        # max_difference = abs(ybir.max() - ybir.min())
        smooth = 1e-6 * smooth_factor

        spline = cs.csaps(xbir, ybir, smooth=smooth)
        self.baseline = spline(self.x)
        baseline_corrected = spectrum - self.baseline
        self.signal.add("baseline_corrected", baseline_corrected)
        self.signal.add("baseline", self.baseline)

        self._processing["baseline_corrected"] = True

    def calculate_noise(self, baseline_regions=None):

        if "baseline_corrected" not in self.signal.names:
            raise RuntimeError("Run baseline correction first")

        if (hasattr(self, "birs")) & (baseline_regions is None):
            baseline_regions = self.birs

        _, ybir = f._extractBIR(
            self.x, self.signal.baseline_corrected, baseline_regions
        )

        self.noise = ybir.std(axis=None) * 2

    def normalise(self, **kwargs):

        y = kwargs.get("y", self._spectrumSelect)
        spectrum = getattr(self.signal, y)

        # normalisation to maximum intensity
        normalised = spectrum * 100 / spectrum.max()
        setattr(self.signal, "normalised", normalised)
        self._processing["normalised"] = True
        self._spectrumSelect = "normalised"

    def interpolate(self, *args, interpolate=[[780, 900]], smooth_factor=1, **kwargs):

        interference_corrected = self._processing.get("interference_corrected", False)
        if interference_corrected:
            spectrum = self.signal.get("interference_corrected")
        else:
            spectrum = self.signal.get("raw")

        smooth = smooth_factor * 1e-5
        use = kwargs.get("use", True)

        spectrum_index = None
        for region in enumerate(interpolate):
            if not spectrum_index:
                spectrum_index = region[1] < self.x < region[0]
            else:
                spectrum_index = spectrum_index | (region[1] < self.x < region[0])

        interpolate_index = ~spectrum_index

        xbir = self.x[spectrum_index]
        ybir = spectrum[spectrum_index]

        spline = cs.csaps(xbir, ybir, smooth=smooth)
        self.spectrum_spline = spline(self.x)
        # Interpolated residual
        noise = (spectrum[spectrum_index] - self.spectrum_spline[spectrum_index]).std(
            axis=None
        ) * 2

        # Add signal noise to the spline
        noise_spline = self.spectrum_spline + np.random.normal(0, noise, len(self.x))

        # only replace interpolated parts of the spectrum
        self.signal.add("interpolated", spectrum.copy())
        self.signal.interpolated[interpolate_index] = noise_spline[interpolate_index]

        if use:
            self._spectrumSelect = "interpolated"
            self._processing["interpolated"] = True

    def fitPeaks(self, peak_prominence=3, fit_window=12, curve="GL", **kwargs):

        y = kwargs.get("y", self._spectrumSelect)
        spectrum = self.signal.get(y)
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
        *,
        peak_height,
        residuals_threshold=0.9,
        baseline0=True,
        min_amplitude=2,
        min_peak_width=4,
        fit_window=4,
        noise=None,
        max_iterations=5,
        print_output=False,
        **kwargs,
    ):

        y = kwargs.get("y", self._spectrumSelect)
        spectrum = self.signal.get(y)
        x = self.x

        if "cutoff" in kwargs:
            cutoff = kwargs.get("cutoff")
            spectrum = spectrum[x < cutoff]
            x = x[x < cutoff]

        # smooth_spectrum = sig.savgol_filter(spectrum, window_length=50, polyorder=2)
        peak_prominence = peak_height + (noise / 2)

        _, centers, widths = cf._find_peak_parameters(
            x=x,
            y=spectrum,
            prominence=peak_prominence,
            height=peak_height,
            width=6,
        )
        ranges = cf._get_peakFit_ranges(
            centers=centers, half_widths=widths, fit_window=fit_window
        )

        # Scale the noise threshold based on max y
        # threshold_scaler = (
        #     lambda y: (
        #         (y / spectrum.max() * 2 * threshold_scale) + (1 - threshold_scale)
        #     )
        #     * noise_threshold
        # )

        fitted_parameters = []
        residual = 0
        total_length = 0
        for i, range in enumerate(ranges):
            xtrim, ytrim = cf._trimxy_ranges(x, spectrum, range)
            # noise_threshold_local = threshold_scaler(ytrim.max())

            print(f"processing range {i:02d}/{len(ranges):02d}\r")
            # try:
            parameters, residual_local = d.deconvolve_signal(
                x=xtrim,
                y=ytrim,
                # noise_threshold=noise_threshold_local,
                residuals_threshold=residuals_threshold,
                baseline0=baseline0,
                min_peak_width=min_peak_width,
                min_amplitude=min_amplitude,
                noise=noise,
                max_iterations=max_iterations,
            )
            fitted_parameters.append(parameters)
            residual += residual_local * len(xtrim)
            total_length += len(xtrim)
            # except:
            #     warn(f"range {range} skipped.")

        residual = residual / total_length

        self.deconvolution_parameters = []
        for parameter in zip(*fitted_parameters):
            self.deconvolution_parameters.append(np.concatenate(parameter))

        self.deconvoluted_peaks = [
            {"center": i, "amplitude": j, "width": k, "shape": l, "baselevel": m}
            for _, (i, j, k, l, m) in enumerate(zip(*self.deconvolution_parameters))
        ]
