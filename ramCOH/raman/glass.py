"""
======
Glass
======
The glass module provides a Raman processing class for processing spectra of silicate glasses, with tailored algorithms for quantifying their water content.
"""

from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import scipy.optimize as opt

from ramCOH.raman.baseclass import RamanProcessing
from ramCOH.raman.baseline_regions import default_birs
from ramCOH.signal_processing import functions as f


class Glass(RamanProcessing):
    """
    A subclass of :py:class:`~ramCOH.raman.baseclass.RamanProcessing`, extended with methods for quantifying water contents of silicate glasses

    Parameters
    ----------
    x   :   array
        1D array with x-axis data
    y   :   array of float or int
        1D array with intensity data
    **kwargs    :   dict, optional
            Optional keyword arguments, see Other parameters

    Other parameters
    ----------------
    laser   :   float, optional
        laser wavelength in nm

    Attributes
    ----------
    birs_default    :   ndarray
        (n, 2) shaped array with left and right boundaries of default baseline interpolation regions
    Si_SNR  : float
        silicate region signal:noise ratio
    H2O_SNR :   float
        water region signal:noise ratio
    SiH2Oareas  : tuple(float, float)
        Area underneath peaks in the silicate and water regions

    """

    # baseline regions
    birs_default = default_birs["glass"]

    def __init__(self, x: npt.NDArray, y: npt.NDArray, **kwargs):

        super().__init__(x, y, **kwargs)
        self._processing.update({"long_corrected": False})

        self.H2O_SNR: Optional[float] = None
        self.Si_SNR: Optional[float] = None

        self.SiH2Oareas: Optional[Tuple[float, float]] = None

    def longCorrect(
        self, T_C=23.0, normalisation=False, inplace=True, **kwargs
    ) -> None:
        """
        Long correction of Raman signal intensities

        Correction for temperature and excitation line effects\ [1]_ according to:

        .. math::
            I = I_{obs} * \left\{ \\nu^{3}_{0} \left[ 1 - exp(-hc\\nu/kT) \\right] \\nu / (\\nu_{0} - \\nu)^{4} \\right\}

        where:

        * :math:`I` = corrected intensity
        * :math:`I_{obs}` = observed intensity
        * :math:`\\nu_{0}` = wavenumber of the incident laser (:math:`m^{-1}`)
        * :math:`\\nu` = measured wavenumber (:math:`m^{-1}`)
        * :math:`T` = temperature in degrees Kelvin

        with constants:

        * :math:`h` = Planck constant
        * :math:`k` = Boltzmann constant
        * :math:`c` = speed of light

        With constant values taken from :py:mod:`scipy:scipy.constants`.
        Results are stored in :py:attr:`~ramCOH.raman.baseclass.RamanProcessing.signal`.

        Parameters
        ----------
        T_C :   float, default 23.0
            temperature during analysis in degrees celsius
        normalisation   :   bool, default False
            normalise Long corrected spectrum to total area
        inplace :   bool, default True
            return Long corrected spectrum when False
        **kwargs    :   dict, optional
            Optional keyword arguments, see Other parameters

        Other parameters
        ----------------
        y   :   str, Optional
            name of the spectrum to be treated


        References
        ----------
        .. [1] Long, D.A. (1977) Raman Spectroscopy, 276 p. MacGraw-Hill, New York.
        """

        laser = kwargs.get("laser", self.laser)
        if not laser:
            raise ValueError("laser wavelength not set!")

        y = kwargs.get("y", self._spectrumSelect)
        spectrum = self.signal.get(y)

        long_corrected = f.long_correction(
            x=self.x,
            intensities=spectrum,
            T_C=T_C,
            laser=laser,
            normalisation=normalisation,
        )
        if inplace:
            self.signal.add("long_corrected", long_corrected)
            return

        return long_corrected

    def calculate_SNR(self) -> None:
        """
        Calculate signal to noise ratios for silicate and raman peaks

        Noise is calculated with self.calculate_noise and
        maxima within the silicate and water regions are used as signals.

        silicate and water regions are set with :py:meth:`~ramCOH.raman.glass.Glass._get_Si_H2O_regions` and
        results are stored in :py:attr:`~ramCOH.raman.glass.Glass.Si_SNR` and :py:attr:`~ramCOH.raman.glass.Glass.H2O_SNR`.


        """

        if self.noise is None:
            self.calculate_noise()

        (Si_left, Si_right), (water_left, water_right) = self._get_Si_H2O_regions()
        Si_range = (self.x > Si_left) & (self.x < Si_right)
        water_range = (self.x > water_left) & (self.x < water_right)

        self.Si_SNR = max(self.signal.baseline_corrected[Si_range]) / self.noise
        self.H2O_SNR = max(self.signal.baseline_corrected[water_range]) / self.noise

    def subtract_interference(
        self,
        interference: npt.NDArray,
        interval: Tuple[Union[float, int], Union[float, int]],
        smoothing: float,
        inplace=True,
        use=False,
        **kwargs
    ) -> None:
        """
        Subtract interfering signals

        Glass and interfering signal are unmixed by minimising the difference between the unmixed
        spectrum and an interpolated, ideal spectrum. Interpolated signal is calculated across ``interval``
        with smoothing splines from :py:func:`csaps:csaps.csaps`,
        The interpolation region should be a region with intefering peaks bracketed by unaffected signal.

        Results are stored in :py:attr:`~ramCOH.raman.baseclass.RamanProcessing.signal`. as *interference_corrected*


        Parameters
        ----------
        interference    :   array
            intensities of intefering signal, x-axis must match with original spectrum
        interval    :   tuple of float
            lower and upper limit of minimisation interval
        smoothing   :   float
            smoothing of interpolation across minimisation interval
        inplace :   bool, default True
            return unmixed spectrum if False
        use :   bool, default False
            set unmixed spectrum as source for further processing
        **kwargs    :   dict, optional
            Optional keyword arguments, see Other parameters

        Other parameters
        ----------------
        y   :   str, Optional
            name of the spectrum to be treated


        """

        boundary_left, boundary_right = interval
        x = self.x

        spectrum = self.signal.get("raw")
        _, spline_interval, _ = f._interpolate_section(
            x, spectrum, interpolate=[interval], smooth_factor=smoothing
        )

        scaling = opt.root(
            f._root_interference,
            x0=[0.2, 0],
            args=(
                x,
                interference,
                spectrum,
                spline_interval,
                [boundary_left, boundary_right],
            ),
        ).x

        scale, shift = scaling
        shift = int(shift)
        interference = f.shift_spectrum(interference, shift)
        interference = interference * scale

        spectrum_corrected = spectrum - interference

        if inplace:
            self.signal.add("interference_corrected", spectrum_corrected.copy())
            if use:
                # self._spectrumSelect = "interference_corrected"
                self._processing["interference_corrected"] = True
            return

        return spectrum_corrected

    def calculate_SiH2Oareas(self) -> Tuple[float, float]:
        """
        Calculate areas underneath peaks in the silicate and water regions

        Areas are calculated by trapezoidal integration of regions set by :py:meth:`~ramCOH.raman.glass.Glass._get_Si_H2O_regions`
        Results are stored in :py:attr:`~ramCOH.raman.glass.Glass.SiH2Oareas`.

        Returns
        -------
        float, float
            silicate region area, water region area

        """

        if "baseline_corrected" not in self.signal.names:
            raise RuntimeError("run baseline correction first")

        (Si_left, Si_right), (water_left, water_right) = self._get_Si_H2O_regions()
        Si_range = (self.x > Si_left) & (self.x < Si_right)
        water_range = (self.x > water_left) & (self.x < water_right)

        spectrum = self.signal.get("baseline_corrected")
        SiArea = np.trapz(spectrum[Si_range], self.x[Si_range])
        H2Oarea = np.trapz(spectrum[water_range], self.x[water_range])

        self.SiH2Oareas = SiArea, H2Oarea

        return self.SiH2Oareas

    def _get_Si_H2O_regions(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Limits of the silicate region are calculated from the upper boundary of the lowest wavenumber
        baseline interpolation region (bir) and the last bir boundary < 1500 cm-1.

        Limits of the water region are calculated from the first bir boundary > 1500 cm-1
        and the lower limit of the highest wavenumber bir.

        If not enough birs are set, the silicate region is set to 200-1400 cm-1 and the water region to 2000-4000 cm-1

        Returns
        -------
        Tuple[float, float], Tuple[float, float]
            silicate region boundaries, water region boundaries


        :meta public:
        """

        bir_boundaries = self.birs.flatten()

        Si_left = bir_boundaries[1]
        try:
            Si_right = bir_boundaries[bir_boundaries < 1500][-1]
        except IndexError:
            Si_right = 1400

        try:
            water_left = bir_boundaries[bir_boundaries > 1500][0]
        except IndexError:
            water_left = 2000

        water_right = bir_boundaries[-2]

        Si_range = (self.x > Si_left) & (self.x < Si_right)
        water_range = (self.x > water_left) & (self.x < water_right)

        if sum(Si_range) == 0:
            Si_left, Si_right = 200, 1400
        if sum(water_range) == 0:
            water_left, water_right = 2000, 4000

        return (Si_left, Si_right), (water_left, water_right)
