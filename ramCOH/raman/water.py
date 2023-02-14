from typing import Tuple, Union

import numpy as np
import numpy.typing as npt
import scipy.optimize as opt

from ..signal_processing import functions as f
from .baseclass import RamanProcessing


class H2O(RamanProcessing):
    # Baseline regions
    birs_default = np.array(
        [[200, 300], [640, 655], [800, 810], [1220, 2300], [3750, 4000]]
    )

    def __init__(self, x, y, **kwargs):

        super().__init__(x, y, **kwargs)
        self._processing.update({"long_corrected": False})

    def longCorrect(self, T_C=23.0, normalisation="area", inplace=True, **kwargs):

        laser = kwargs.get("laser", self.laser)

        y = kwargs.get("y", self._spectrumSelect)
        spectrum = self.signal.get(y)

        long_corrected = f.long_correction(self.x, spectrum, T_C, laser, normalisation)
        if inplace:
            self.signal.add("long_corrected", long_corrected)
            return

        return long_corrected

    def calculate_SNR(self):

        if self.noise is None:
            self.calculate_noise()

        (Si_left, Si_right), (water_left, water_right) = self._get_Si_H2O_regions()
        Si_range = (self.x > Si_left) & (self.x < Si_right)
        water_range = (self.x > water_left) & (self.x < water_right)

        self.Si_SNR = max(self.signal.baseline_corrected[Si_range]) / self.noise
        self.H2O_SNR = max(self.signal.baseline_corrected[water_range]) / self.noise

    def subtract_interference(
        self,
        interference,
        interval: Tuple[Union[float, int], Union[float, int]],
        smoothing,
        inplace=True,
        use=False,
        **kwargs
    ):

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

    def calculate_SiH2Oareas(self, **kwargs):

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

    def _get_Si_H2O_regions(self):

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
