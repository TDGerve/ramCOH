from typing import List, Optional, Tuple
from warnings import warn

import csaps as cs
import numpy as np
import scipy.interpolate as itp
import scipy.optimize as opt

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
        if name in self._names:
            self.set(name, values)
            return
        setattr(self, name, values)
        self._names.append(name)

    def set(self, name, values):
        if name not in self.names:
            raise ValueError(f"{name} not in signals")
        setattr(self, name, values)

    def get(self, name):
        array = getattr(self, name, None)
        if array is not None:
            array = array.copy()
        return array

    @property
    def all(self):
        return {name: self.get(name) for name in self.names}

    def interpolate_spectrum(self, old_x, old_y):
        interpolate = itp.interp1d(
            old_x, old_y, bounds_error=False, fill_value="extrapolate"
        )
        return interpolate(self.x)

    def set_with_interpolation(self, name, x, y):
        new_y = self.interpolate_spectrum(x, y)
        self.add(name, new_y)

    @property
    def names(self):
        return self._names

    def remove(self, names):
        for name in names:
            if name not in self._names:
                continue
            delattr(self, name)
            self._names.remove(name)


class RamanProcessing:
    def __init__(self, x, y, laser=532.18):
        x, y = f.trim_sort(x, y)
        self.x = x
        self.signal = Signal(x, y)
        self.noise: Optional[float] = None
        self.laser = laser
        self._processing = {
            "raw": True,
            "interference_corrected": False,
            "interpolated": False,
        }
        self.birs = None
        self.peaks = []

    @property
    def processing(self):
        return self._processing

    @property
    def _spectrumSelect(self):
        spectra = ("interference_corrected", "interpolated")
        selection = "raw"
        for key in spectra:
            selection = key if self._processing[key] else selection
        return selection

    def _set_processing(self, types: List[str], values: List[bool]):
        for t, val in zip(types, values):
            try:
                _ = self._processing.get(t)
                self._processing[t] = val
            except KeyError:
                warn(message=f"key '{t}' not found")

    def smooth(self, smoothType="Gaussian", kernelWidth=9, apply=False, **kwargs):
        """
        Smoothing by either a moving average or with a Gaussian kernel.
        Each application shortens the spectrum by one kernel width
        """

        y = kwargs.get("y", self._spectrumSelect)
        spectrum = self.signal.get(y)

        smooth = f.smooth(spectrum, smoothType, kernelWidth)

        if not apply:
            return smooth

        self.signal.set(self._spectrumSelect, smooth)

        # match length of x with length of smoothed signal
        self.x = self.x[(kernelWidth - 1) // 2 : -(kernelWidth - 1) // 2]
        # do the same for any other pre-existing spectra
        for name, value in self.signal.all.items():
            if name == self._spectrumSelect:
                continue
            shortened = value[(kernelWidth - 1) // 2 : -(kernelWidth - 1) // 2]
            self.signal.set(name, shortened)

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

        xbir, ybir = f._extractBIR(self.x, spectrum, baseline_regions)

        # max_difference = abs(ybir.max() - ybir.min())
        smooth = 1e-6 * smooth_factor

        spline = cs.csaps(xbir, ybir, smooth=smooth)
        self.baseline = spline(self.x)
        baseline_corrected = spectrum - self.baseline

        self.signal.add("baseline_corrected", baseline_corrected)
        self.signal.add("baseline", self.baseline)

    def calculate_noise(self, baseline_regions=None):

        baseline_corrected = self.signal.get("baseline_corrected")
        if baseline_corrected is None:
            raise RuntimeError("Run baseline correction first")

        if (hasattr(self, "birs")) & (baseline_regions is None):
            baseline_regions = self.birs

        _, ybir = f._extractBIR(self.x, baseline_corrected, baseline_regions)

        self.noise = ybir.std(axis=None) * 2

    def normalise(self, apply=False, **kwargs):

        y = kwargs.get("y", self._spectrumSelect)
        spectrum = getattr(self.signal, y)

        # normalisation to maximum intensity
        normalised = spectrum * 100 / spectrum.max()

        if not apply:
            return normalised

        self.signal.set(self._spectrumSelect, normalised)

    def interpolate(
        self,
        *args,
        interpolate=List[Tuple],
        smooth_factor=1,
        add_noise=True,
        output=False,
        **kwargs,
    ):
        y = kwargs.get("y", self._spectrumSelect)
        spectrum = self.signal.get(y)
        x = self.x

        use = kwargs.get("use", True)

        interpolate_index = f._extractBIR_bool(x, interpolate)
        spectrum_index = ~interpolate_index

        interpolated_x, interpolated_y, spline = f._interpolate_section(
            x, spectrum, interpolate, smooth_factor
        )

        if add_noise:
            noise = (spectrum[spectrum_index] - spline(spectrum_index)).std(
                axis=None
            ) * 2
            interpolated_y = interpolated_y + np.random.normal(
                0, noise, len(interpolated_y)
            )

        interpolated_spectrum = f.add_interpolation(
            spectrum, interpolate_index, interpolated_y
        )
        self.signal.add("interpolated", interpolated_spectrum)
        if use:
            self._processing["interpolated"] = True

        if output:
            return interpolated_x, interpolated_y

    def fitPeaks(self, peak_prominence=3, fit_window=12, curve="GL", **kwargs):

        y = kwargs.get("y", self._spectrumSelect)
        spectrum = self.signal.get(y)
        # clear old peaks
        self.peaks = []
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

            self.peaks.append({k: fitParams[j] for j, k in enumerate(params)})

    def deconvolve(
        self,
        *,
        peak_height,
        residuals_threshold=10,
        baseline0=True,
        min_amplitude=1,
        min_peak_width=4,
        fit_window=4,
        noise=None,
        max_iterations=5,
        **kwargs,
    ):

        y = kwargs.get("y", self._spectrumSelect)
        spectrum = self.signal.get(y)
        x = self.x

        if "cutoff" in kwargs:
            cutoff = kwargs.get("cutoff")
            spectrum = spectrum[x < cutoff]
            x = x[x < cutoff]
        # clear old peaks
        self.peaks = []

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

        fitted_parameters = []
        residual = 0
        total_length = 0
        for i, range in enumerate(ranges):
            xtrim, ytrim = cf._trimxy_ranges(x, spectrum, range)
            # noise_threshold_local = threshold_scaler(ytrim.max())

            print(f"processing range {i+1:02d}/{len(ranges):02d}\n")
            try:
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
            except IndexError:
                warn(f"range {[int(i) for i in range]}skipped.")

        residual = residual / total_length

        deconvolution_parameters = [np.concatenate(p) for p in zip(*fitted_parameters)]
        self.signal.add(
            name="deconvoluted", values=c.sum_GaussLorentz(x, *deconvolution_parameters)
        )

        self.peaks = [
            {"center": i, "amplitude": j, "width": k, "shape": l, "baselevel": m}
            for _, (i, j, k, l, m) in enumerate(zip(*deconvolution_parameters))
        ]
