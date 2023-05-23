"""
=============
CO2
=============
The CO\ :sub:`2`\  module provides a Raman processing class for for processing CO\ :sub:`2`\ Raman data
"""
from typing import Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt

from ramCOH.raman.baseclass import RamanProcessing
from ramCOH.raman.baseline_regions import default_birs
from ramCOH.signal_processing import curves as c


class CO2(RamanProcessing):
    """
    A subclass of :py:class:`~ramCOH.raman.baseclass.RamanProcessing`, extended with methods for fitting the CO\ :sub:`2` Fermi diad.

    Attributes
    ----------
    diad    :   tuple of dictionaries
        fitted peak parameters of the diad peaks
    diad_split  :   float
        splitting of the Fermi diad in cm-1

    """

    diad: Optional[Tuple[Dict, Dict]] = None
    diad_split: Optional[float] = None

    birs_default = default_birs["CO2"]

    def FermiDiad(
        self, peak_height=20, fit_window=20, min_peak_width=4, **kwargs
    ):  # peak_prominence=40
        """
        Fit the Fermi diad with mixed Gaussian-Lorentzian curves.

        The spectrum is fitted with peaks with :py:attr:`~ramCOH.raman.baseclass.RamanProcessing.deconvolve`.
        The peak with the higest amplitude at wavenumbers <1330cm\ :sup:`-1` is used as the lower diad peak and the peak
        with the highest amplitude at wavenumbers >1330cm\ :sup:`-1` as the upper diad peak. Diad splitting is calculated as the
        absolute difference between the fitted centers of the diad peaks.

        Results are stored in :py:attr:`~ramCOH.raman.baseclass.RamanProcessing.peaks`, :py:attr:`~ramCOH.raman.CO2.CO2.diad` and :py:attr:`~ramCOH.raman.CO2.CO2.diad_split`.

        Analyses of gas+liquid CO\ :sub:`2` mixtures also result in mixed diad peaks,
        be careful with interpreting diad splitting if this is the case.


        Parameters
        ----------
        peak_height :   float or in, default 20
            minimum absolute height of peaks included in initial guesses, passed to :py:func:`~scipy:scipy.signal.find_peaks`
        fit_window  :   int, default 20
            width parameter of the x-axis window within which peaks are fitted. Actual width is calculated as ``fit_window`` :math:`\\times` *guessed width*.
            passed to :py:meth:`~ramCOH.raman.baseclass.RamanProcessing.deconvolve`
        min_peak_width  : int, default 4
            minimum full width of fitted peaks in x-axis steps
            assed to :py:meth:`~ramCOH.raman.baseclass.RamanProcessing.deconvolve`
        **kwargs    :   dict, optional
            Optional keyword arguments, passed to :py:attr:`~ramCOH.raman.baseclass.RamanProcessing.deconvolve`

        """

        if "baseline_corrected" not in self.signal.names:
            raise RuntimeError("run baseline correction first")

        self.peaks = self.deconvolve(
            y="baseline_corrected",
            peak_height=peak_height,
            fit_window=fit_window,
            min_peak_width=min_peak_width,
            baseline0=True,
            inplace=False,
            x_min=1200,
            x_max=1450,
            **kwargs,
        )

        lower = [peak for peak in self.peaks if peak["center"] < 1330]
        upper = [peak for peak in self.peaks if peak["center"] > 1330]

        self.diad = []
        for region in (lower, upper):
            amplitude_sort = np.argsort([peak["amplitude"] for peak in region])
            peaks_sorted = np.array(region)[amplitude_sort]
            self.diad.append(peaks_sorted[-1])
            # TODO give a warning when a mixed vapour-liquid signal is suspected

        self.diad_split = abs(self.diad[0]["center"] - self.diad[1]["center"])

    def diad_curves(self, window=8, stepsize=0.5) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Get diad curves from fitted parameters

        Parameters
        ----------
        window  :   int, default 8
            width parameter of the x-axis window within which curves are calculated. Actual width is calculated as ``window`` :math:`\\times` *peak width*.
        stepsize    : float
            x-axis stepsize

        Returns
        -------
        npt.NDArray, npt.NDArray
            (x, 2) shaped arrays with columns x, y for each diad peak.

        """

        if self.diad is None:
            raise AttributeError("Run 'FermiDiad()' first!")

        results = []

        for peak in self.diad:
            width = peak["width"]

            start = peak["center"] - window * width
            stop = peak["center"] + window * width

            x = np.arange(start=start, stop=stop, step=stepsize)
            y = c.GaussLorentz(x=x, **peak)

            results.append(np.vstack([x, y]))

        return results
