"""
======
Neon
======
The neon module provides a Raman processing class for processing neon emission spectra.
"""

import itertools as it
from typing import Optional

import numpy as np
import numpy.typing as npt

from ..signal_processing import functions as f
from .baseclass import RamanProcessing
from .baseline_regions import default_birs


class Neon(RamanProcessing):
    """
    A subclass of :py:class:`~ramCOH.raman.baseclass.RamanProcessing`, extended with methods for Raman streching and offset correction.

    Attributes
    ----------
    emission_peaks  :   array of float
        theoretical neon emission lines (cm\ :sup:`-1`)
    observed_peaks    :   array of float
        observed neon emission lines (cm\ :sup:`-1`)
    correction_factor   :   float
        stretching correction factor (cm\ :sup:`-1`)
    offset  :   float
        offset correction factor (cm\ :sup:`-1`)
    """

    # Baseline regions
    birs_default = default_birs["neon"]

    emission_peaks: Optional[npt.NDArray] = None
    observed_peaks: Optional[npt.NDArray] = None

    correction_factor: Optional[float] = None
    offset: Optional[float] = None

    def __init__(self, x: npt.NDArray, y: npt.NDArray, laser: float):

        super().__init__(x=x, y=y, laser=laser)

    def deconvolve(
        self,
        peak_height=2,
        residuals_theshold=10,
        min_amplitude=8,
        min_peak_width=2,
        fit_window=4,
        max_iterations=3,
        noise=None,
        **kwargs,
    ):
        """
        :meta private:
        """
        return super().deconvolve(
            peak_prominence=peak_height,
            residuals_threshold=residuals_theshold,
            baseline0=True,
            min_amplitude=min_amplitude,
            min_peak_width=min_peak_width,
            fit_window=fit_window,
            noise=noise,
            max_iterations=max_iterations,
            **kwargs,
        )

    def neonCorrection(
        self, left_nm=565.666, right_nm=574.83, search_window=6, **kwargs
    ):
        """
        Calculate stretching and offset correction factors between observed and theoretical Neon emission lines.

        Compare theoretical neon emission lines near the calibration lines ``left_nm`` and ``right_nm`` with
        observed peaks. Correction for stretching is calculated as the ratio of the inter calibration line distances of
        theoretical and observed lines. Offset correction is calculated as the difference between the observed and theoretical position of ``left_nm``.


        Results are stored in :py:attr:`~ramCOH.raman.neon.Neon.correction_factor` and :py:attr:`~ramCOH.raman.neon.Neon.offset`.

        Parameters
        ----------
        left_nm :   float, default 565.666
            position of the left calibration line in nm
        right_nm :   float, default 574.83
            position of the right calibration line in nm
        search_window   :   int
            Half-width (cm\ :sup:`-1`) of the peak search window around theoretical emission lines .
        """
        laser = kwargs.get("laser", self.laser)

        if self.peaks is None:
            raise NameError("peaks not found, run fitPeaks first")

        neon_emission = f.neonEmission(laser=laser)
        left = np.round(
            np.float(
                neon_emission.iloc[:, 4][
                    np.isclose(left_nm, neon_emission.iloc[:, 1], atol=0.001)
                ]
            ),
            2,
        )
        right = np.round(
            np.float(
                neon_emission.iloc[:, 4][
                    np.isclose(right_nm, neon_emission.iloc[:, 1], atol=0.001)
                ]
            ),
            2,
        )

        # All emission lines within spectrum limits
        neon_emission_trimmed = np.array(
            neon_emission.iloc[:, 4][
                (neon_emission.iloc[:, 4] > self.x.min())
                & (neon_emission.iloc[:, 4] < self.x.max())
            ]
        )

        # theoretical emission line positions with a match found in spectrum
        self.emission_peaks = np.array([])
        # measured emission line positions
        self.observerd_peaks = np.array([])

        for i, j in self.peaks.items():
            peak = j["center"]
            emission_check = np.isclose(
                peak, neon_emission_trimmed, atol=search_window, rtol=0
            )

            if emission_check.sum() == 1:
                self.emission_peaks = np.append(
                    self.emission_peaks, neon_emission_trimmed[emission_check].tolist()
                )
                self.observerd_peaks = np.append(self.observerd_peaks, peak)
            elif emission_check.sum() > 1:
                raise RuntimeError(
                    f"multiple emission line fits found at {peak: .2f} cm$^{-1}$"
                )

        # find correction factor for the calibration lines
        if not np.isin([left, right], np.round(self.emission_peaks, 2)).sum() == 2:
            raise RuntimeError("calibration lines not found in spectrum")

        # boolean array for the location of left and right calibration lines
        calibration_lines = np.array(
            np.isclose(left, self.emission_peaks, atol=0.001)
            + np.isclose(right, self.emission_peaks, atol=0.001)
        )
        # boolean to index
        calibration_lines = list(
            it.compress(range(len(calibration_lines)), calibration_lines)
        )
        # indices for differenced array
        calibration_lines = np.unique(calibration_lines - np.array([0, 1]))

        self.correction_factor = np.float(
            np.sum(np.diff(self.emission_peaks)[calibration_lines])
            / np.sum(np.diff(self.observerd_peaks)[calibration_lines])
        )
        self.offset = np.float(
            self.observerd_peaks[np.isclose(left, self.observerd_peaks, atol=10)]
            - self.emission_peaks[np.isclose(left, self.emission_peaks, atol=0.1)]
        )
