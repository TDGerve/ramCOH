"""
===============
Generic classes
===============
The baseclass module provides generic classes for processing Raman data and storing spectral data
"""

from typing import Annotated, Dict, List, Literal, Optional, Tuple, Union
from warnings import warn

import csaps as cs
import numpy as np
import numpy.typing as npt
import scipy.interpolate as itp
import scipy.optimize as opt

from ramCOH.signal_processing import curve_fitting as cf
from ramCOH.signal_processing import curves as c
from ramCOH.signal_processing import deconvolution as d
from ramCOH.signal_processing import functions as f

arrayNx2 = Annotated[npt.NDArray, Literal["N", 2]]


class Signal:
    """Container for spectral data

    Parameters
    ----------
    x   :   array
        1D array with x-axis data
    y   :   array
        1D array with intensity data

    Attributes
    ----------
    x   :   array of float
        x-axis data
    raw :   array of float
        unprocessed intensity data
    names   :   list of str
        names of all current spectra
    all : dict
        all current spectra as name: values
    """

    def __init__(self, x: npt.NDArray, y: npt.NDArray):
        self.x = x
        self.raw = y
        self._names: List[str] = ["raw"]

    @property
    def names(self):
        return self._names

    @property
    def all(self) -> Dict:
        return {name: self.get(name) for name in self.names}

    def add(self, name: str, values: npt.NDArray) -> None:
        """
        Add a new spectrum
        """
        if name in self._names:
            self.set(name, values)
            return
        setattr(self, name, values)
        self._names.append(name)

    def set(self, name, values: npt.NDArray) -> None:
        """
        Set the values of a spectrum
        """
        if name not in self.names:
            raise ValueError(f"{name} not in signals")
        setattr(self, name, values)

    def get(self, name: str) -> npt.NDArray:
        """
        Get the values of a spectrum
        """
        array = getattr(self, name, None)
        if array is not None:
            array = array.copy()
        return array

    def interpolate_spectrum(
        self, old_x: npt.NDArray, old_y: npt.NDArray
    ) -> npt.NDArray:
        """
        Linear interpolation a spectrum to match its x-axis with :py:attr:`~ramCOH.raman.baseclass.Signal.x`

        """
        interpolate = itp.interp1d(old_x, old_y, bounds_error=False, fill_value=0.0)
        return interpolate(self.x)

    def set_with_interpolation(self, name: str, x: npt.NDArray, y: npt.NDArray) -> None:
        """
        Add a new spectrum with interpolated intensities.

        Interpolation is calculated with :py:meth:`~ramCOH.raman.baseclass.Signal.interpolate_spectrum`
        """
        new_y = self.interpolate_spectrum(x, y)
        self.add(name, new_y)

    def remove(self, names):
        for name in names:
            if name not in self._names:
                continue
            delattr(self, name)
            self._names.remove(name)


class RamanProcessing:
    """Generic class for processing Raman spectral data

    Parameters
    ----------
    x   :   array
        1D array with x-axis data
    y   :   array of float or int
        1D array with intensity data
    laser   : float, optional
        laser wavelenghth in nm

    Attributes
    ----------
    x : array
        1D array with x-axis data
    signal : Signal
        container with all spectral data
    noise   :   float
        calculated average noise on the baseline corrected spectrum
    laser   :   float, optional
        laser wavelength in nm
    processing  :   dict
        dictionary of bool indicating which data treatments have been applied
    birs    : ndarray
        (n, 2) shaped array with left and right boundaries of the last used baseline interpolation regions
    peaks   :   list of dict
        list of best-fit parameters of peaks fitted to the spectrum by either deconvolution or simple peak fitting
    _spectrumSelect :   str
        default spectrum selected for processing. The selection hierarchy is raw -> interference_corrected -> interpolated,
        where the last available spectrum is selected

    """

    def __init__(self, x: npt.NDArray, y: npt.NDArray, laser: Optional[float] = None):
        x, y = f.trim_sort(x, y)
        self.x = x
        self.signal = Signal(x, y)

        self.laser = laser
        self._processing = {
            "raw": True,
            "interference_corrected": False,
            "interpolated": False,
        }

        self.noise: Optional[float] = None
        self.birs: Optional[arrayNx2[float]] = None
        self.peaks: Optional[List[Dict]] = None

    @property
    def processing(self) -> dict:
        return self._processing

    @property
    def _spectrumSelect(self) -> str:
        spectra = ("interference_corrected", "interpolated")
        selection = "raw"
        for key in spectra:
            selection = key if self._processing[key] else selection
        return selection

    def _set_processing(self, types: List[str], values: List[bool]) -> None:
        for t, val in zip(types, values):
            try:
                _ = self._processing.get(t)
                self._processing[t] = val
            except KeyError:
                warn(message=f"key '{t}' not found")

    def smooth(self, type="gaussian", kernel_width=9, inplace=False, **kwargs) -> None:
        """
        Smoothing with either a moving average or with a Gaussian kernel.
        Note that each application shortens the spectrum by one kernel width.
        The raw spectrum will be used if keyword argment ``y`` is not set.

        If inplace is set to True, the smoothed spectrum will overwrite the original spectrum.

        Parameters
        ----------
        type    :   str
            Smoothing kernel type: 'gaussian' or 'moving_average'
        kernel_width    :   int, default 9
            Size of the smoothing kernel in steps along the x-axis
        inplace   :   bool, default False
            Return the smoothed array
        **kwargs    :   dict, optional
            Optional keyword arguments, see Other parameters

        Other Parameters
        -------------------
        y   :   str, Optional
            name of the spectrum to be treated

        """

        y = kwargs.get("y", "raw")
        spectrum = self.signal.get(y)

        smooth = f.smooth(y=spectrum, type=type, kernel_width=kernel_width)

        x_shortened = self.x[(kernel_width - 1) // 2 : -(kernel_width - 1) // 2]

        if not inplace:
            return x_shortened, smooth

        self.signal.set(y, smooth)

        # match length of x with length of smoothed signal
        self.x = x_shortened
        # do the same for any other pre-existing spectra
        for name, value in self.signal.all.items():
            if name == self._spectrumSelect:
                continue
            shortened = value[(kernel_width - 1) // 2 : -(kernel_width - 1) // 2]
            self.signal.set(name, shortened)

    def baselineCorrect(self, baseline_regions=None, smooth_factor=1, **kwargs) -> None:
        """
        Baseline correction with natural smoothing splines from :py:func:`csaps:csaps.csaps` fitted to interpolation regions.

        Results are stored in :py:attr:`~ramCOH.raman.baseclass.RamanProcessing.signal`.


        Parameters
        ----------
        baseline_regions    :   ndarray, optional
            (n, 2) array for n interpolation regions, where each row is [lower_limit, upper_limit]
        smooth_factor: float, default 1
            baseline smoothing factor between 0-1. As the value approaches 0, the baseline becomes linear.
            It is passed to the smooth parameter of :py:func:`~csaps:csaps.csaps` as 1e-6 * smooth_factor
        **kwargs    :   dict, optional
            Optional keyword arguments, see Other parameters

        Other Parameters
        ----------------
        y   :   str, Optional
            name of the spectrum to be treated
        """
        y = kwargs.get("y", self._spectrumSelect)
        spectrum = self.signal.get(y)

        if hasattr(self, "birs_default") & (baseline_regions is None):
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

    def calculate_noise(self, baseline_regions: Optional[npt.NDArray] = None) -> None:
        """
        Calculate noise on a baseline corrected spectrum.

        Noise is calculated within baseline interpolation regions as
        2 standard deviations on the baseline corrected spectrum.

        Results are stored in :py:attr:`~ramCOH.raman.baseclass.RamanProcessing.noise`

        Other parameters
        ----------------
        baseline_regions    : ndarray, optional
            (n, 2) shapped array with regions where noise will be calculated. If not set, the last set baseline interpolation regions will be used.
        """

        baseline_corrected = self.signal.get("baseline_corrected")
        if baseline_corrected is None:
            raise RuntimeError("Run baseline correction first")

        if (self.birs is not None) & (baseline_regions is None):
            baseline_regions = self.birs

        _, ybir = f._extractBIR(self.x, baseline_corrected, baseline_regions)

        self.noise = ybir.std(axis=None) * 2

    def normalise(self, inplace=False, **kwargs):
        """
        Normalise spectrum to maximum intensity :math:`\\times` 100. The raw spectrum will be used if keyword argment ``y`` is not set.

        If inplace is set to True, the smoothed spectrum will overwrite the original spectrum.

        Parameters
        ----------
        inplace :   bool, default False
            set normalised spectrum inplace
        **kwargs    :   dict, optional
            Optional keyword arguments, see Other parameters

        Other Parameters
        ----------------
        y   :   str, Optional
            name of the spectrum to be treated
        """

        y = kwargs.get("y", "raw")
        spectrum = getattr(self.signal, y)

        # normalisation to maximum intensity
        normalised = spectrum * 100 / spectrum.max()

        if not inplace:
            return normalised

        self.signal.set(y, normalised)

    def interpolate(
        self,
        interpolate: List[Tuple[float, float]],
        smooth_factor=1,
        add_noise=True,
        output=False,
        use=False,
        **kwargs,
    ) -> None:
        """
        Interpolate across one or more regions

        Interpolated signal is calculated fitting smoothing splines from :py:func:`csaps:csaps.csaps` to the rest of the spectrum.

        Results are stored in :py:attr:`~ramCOH.raman.baseclass.RamanProcessing.signal`

        Parameters
        ----------
        interpolate :   list of lists
            list of interpolation regions in pairs of [upper limit, lower limits]
        smooth_factor   :   float, default 1
            Smoothes the interpolation; as it approaches 0 the interpolation becomes linear.
            Passed on to the smooth parameter in :py:func:`~csaps:csaps.csaps` as smooth_factor * 1e-5
        add_noise   :   bool, default True
            add spectrum noise to the interpolated sections
        use :   bool, default False
            set interpolated spectrum as source for further processing
        **kwargs    :   dict, optional
            Optional keyword arguments, see Other parameters

        Other Parameters
        ----------------
        y   :   str, Optional
            name of the spectrum to be treated
        output  :   bool, default False
            return interpolated sections as as tuple(x, y)

        """
        y = kwargs.get("y", self._spectrumSelect)
        spectrum = self.signal.get(y)
        x = self.x

        interpolate_index = f._extractBIR_bool(x, interpolate)
        spectrum_index = ~interpolate_index

        interpolated_x, interpolated_y, spline = f._interpolate_section(
            x, spectrum, interpolate, smooth_factor
        )

        if add_noise:
            for section in interpolate:
                left, right = section
                window = f._extractBIR_bool(x, [[left, right]])
                noise_window = (
                    f._extractBIR_bool(x, [[left - 50, right + 50]]) & ~window
                )
                local_noise = (spectrum[noise_window] - spline(x[noise_window])).std(
                    axis=None
                )

                mask = (interpolated_x > left) & (interpolated_x < right)
                interpolated_y[mask] = interpolated_y[mask] + np.random.normal(
                    0, local_noise, sum(mask)
                )

        if use:
            interpolated_spectrum = f.add_interpolation(
                spectrum, interpolate_index, interpolated_y
            )
            self.signal.add("interpolated", interpolated_spectrum)
            self._processing["interpolated"] = True

        if output:
            return interpolated_x, interpolated_y

    def fit_peaks(self, peak_prominence=3, fit_window=12, curve="GL", **kwargs) -> None:
        """
        Find the best fit curves for peaks in the spectrum.

        Does not take into account peak overlap,
        use :py:meth:`~ramCOH.raman.baseclass.RamanProcessing.deconvolve` instead if you expect overlapping peaks.
        Initial guesses for peak centers, amplitudes and widths are made with :py:func:`scipy:scipy.signal.find_peaks`

        Results are stored in :py:attr:`~ramCOH.raman.baseclass.RamanProcessing.peaks` as a list of dictionaries with fitted parameters for each individual peak.



        Parameters
        ----------
        peak_prominence : float or int, default 3
            minimum peak prominence of fitted peaks, passed to :py:func:`find_peaks`
        fit_window  :   int, default 12
            Intervals across which peaks are fitted range from (peak center - (fit_window * half width)) to (peak center + (fit_window * half width))
        curve   :   str, default 'GL'
            curve shape. Options are:

            * 'GL' for mixed Gaussian-Lorentzian/pseudo-Voigt (default),
            * 'G' for Gaussian and
            * 'L' for Lorentzian
        **kwargs    :   dict, optional
            Optional keyword arguments, see Other parameters

        Other Parameters
        ----------------
        y   :   str, Optional
            name of the spectrum to be treated

        """

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
        residuals_threshold: float = 10,
        baseline0: bool = True,
        min_amplitude: Union[int, float] = 3,
        min_peak_width: Union[int, float] = 4,
        fit_window: int = 4,
        max_iterations: int = 5,
        noise: Optional[float] = None,
        inplace=True,
        **kwargs,
    ) -> None:
        """
        Deconvolve the spectrum into its constituent peaks.

        Initial guesses for peak centers, amplitudes and widths are made with :py:func:`scipy:scipy.signal.find_peaks`.
        The spectrum is subdivided into multiple windows with *width* = ``fit_window``\ :math:`\\times` *guessed width*,
        where overlapping windows are merged. Within each window :py:func:`mixed Gaussian-Lorentzian <ramCOH.signal_processing.curves.GaussLorentz>`
        peaks are iteratively fitted to the spectrum. With each new iteration an additional peak is summed with previous peaks.
        Iterations are stopped when ``max_iteraton`` is reached or when the residuals have improved less then ``residuals_threshold``.


        Results are stored in :py:attr:`~ramCOH.raman.baseclass.RamanProcessing.peaks`
        as a list of dictionaries with fitted parameters for each individual peak.


        Parameters
        ----------
        peak_height :   float or int
            minimum absolute peak height for initial guesses
        residuals_threshold : float or int, default 10
            fit iterations are stopped when the root-mean squared error has decreased less than ``residuals_threshold``\% compared to the previous iteration.
        baseline0   :   bool, default True
            fix the baselevel at 0
        min_amplitude   : int or float, default 3
            minium amplitude of fitted peaks as a factor of noise on y.
        min_peak_width  : int, default 8
            minimum full width of fitted peaks in x-axis steps
        fit_window  :   int, default 4
            width parameter of individual fit frames. Actual width is calculated as ``fit_window``\ :math:`\\times` *guessed full width*.
        max_iterations  : int, default 5
            maximum fit iterations per window. Each iteration a new peak is added and ``max_iterations``
        noise   :   float, optional
            average noise on the spectrum. If no value is given it will be calculated with :py:attr:`~ramCOH.raman.baseclass.RamanProcessing.calculate_noise`
        inplace :   bool, default True
            return fitted peaks when False
        **kwargs    :   dict, optional
            Optional keyword arguments, see Other parameters

        Other Parameters
        ----------------
        y   :   str, Optional
            name of the spectrum to be treated
        x_min   :   float or int, optional
            x-axis lower limit of deconvolved spectrum
        x_max   :   float or int, optional
            x-axis upper limit of deconvolved spectrum
        """

        y = kwargs.get("y", self._spectrumSelect)
        spectrum = self.signal.get(y)
        x = self.x

        if noise is None:
            self.calculate_noise()
            noise = self.noise

        if x_max := kwargs.get("x_max", None):
            spectrum = spectrum[x < x_max]
            x = x[x < x_max]
        if x_min := kwargs.get("x_min", None):
            spectrum = spectrum[x > x_min]
            x = x[x > x_min]

        # clear old peaks
        if self.peaks is not None:
            self.peaks = []

        # convert to half width (used by scipy.signal.find_peaks)
        min_peak_width = min_peak_width / 2

        # smooth_spectrum = sig.savgol_filter(spectrum, window_length=50, polyorder=2)
        peak_prominence = peak_height + (noise / 2)

        _, centers, widths = cf._find_peak_parameters(
            x=x,
            y=spectrum,
            prominence=peak_prominence,
            height=peak_height,
            width=min_peak_width,
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
        peaks = [
            {"center": i, "amplitude": j, "width": k, "shape": l, "baselevel": m}
            for _, (i, j, k, l, m) in enumerate(zip(*deconvolution_parameters))
        ]

        if not inplace:
            return peaks

        self.peaks = peaks
