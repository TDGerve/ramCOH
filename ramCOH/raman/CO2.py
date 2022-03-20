from . import general as ram
from ..signal_processing import curve_fitting as cf
from ..signal_processing import curves as c
import numpy as np


class CO2(ram.RamanProcessing):

    birs = np.array(
        [[1000, 1260], [1270, 1275], [1300, 1375], [1396, 1402], [1418, 1500]]
    )

    def FermiDiad(self, peak_prominence=40, fit_window=8, **kwargs):

        y = kwargs.get("y", self.spectrumSelect)
        spectrum = self.intensities[y]
        self.diad = {}

        # fit parameters for diad
        self.diad["fit_params1"], self.diad["fit_params2"] = cf.diad(
            x=self.x,
            intensities=spectrum,
            peak_prominence=peak_prominence,
            fit_window=fit_window,
        )

        # diad curves
        self.diad["peak1"] = {
            "x": self.diad["fit_params1"]["x"],
            "y": c.GaussLorentz(**self.diad["fit_params1"]),
        }
        self.diad["peak2"] = {
            "x": self.diad["fit_params2"]["x"],
            "y": c.GaussLorentz(**self.diad["fit_params2"]),
        }
        del self.diad["fit_params1"]["x"]
        del self.diad["fit_params2"]["x"]
        # diad split
        self.diad["split"] = abs(
            self.diad["fit_params1"]["center"] - self.diad["fit_params2"]["center"]
        )
