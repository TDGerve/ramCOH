import os
import ramCOH as ram
import pandas as pd
import numpy as np
import json


def import_birs():
    with open("birs.json") as f:
        birs_all = json.load(f)

    birs = []
    labels = []

    for name, bir in birs_all.items():
        if bir["use"]:
            labels.append(name)
            birs.append(np.array(bir["birs"]))

    return labels, birs



class data_processing:

    smooth_factor = 1

    # Get baseline interpolation regions
    Si_birs_labels, Si_birs = import_birs()

    def __init__(self, files, settings, type="H2O"):
        # Parse files
        separator = settings.name_separator.get()
        self.files = files
        self.names = ()
        for file in files:
            name = os.path.basename(file)
            if separator not in name:
                self.names = self.names + tuple([name])
            else:
                self.names = self.names + tuple([name[: name.find(separator)]])
        self.spectra = {}
        self.laser = settings.laser.get()
        self.model = getattr(ram, type)

        # Create dataframe to store outputs
        self.results = pd.DataFrame({"name": self.names})
        self.results["SiArea"] = np.zeros(self.results.shape[0])
        self.results["H2Oarea"] = np.zeros(self.results.shape[0])
        self.results["rWS"] = np.zeros(self.results.shape[0])

        # Create dataframe to store processing parameters
        self.processing = pd.DataFrame({"name": self.names, "interpolate": False})
        self.processing["interpolate_left"] = int(780)
        self.processing["interpolate_right"] = int(900)
        self.processing["Si_bir"] = int(0)
        self.processing["water_left"] = int(2800)
        self.processing["water_right"] = int(3850)

        # # Get baseline interpolation regions
        # self.Si_birs_labels, self.Si_birs = import_birs()

    def preprocess(self):
        for i, f in enumerate(self.files):
            x, y = np.genfromtxt(f, unpack=True)
            self.spectra[i] = self.model(x, y, laser=self.laser)
            self.spectra[i].longCorrect()
            self.spectra[i].baselineCorrect(smooth_factor=1)
            self.spectra[i].calculate_SiH2Oareas()
            Si_area, H2O_area = self.spectra[i].SiH2Oareas
            self.results.loc[i, ["SiArea", "H2Oarea"]] = Si_area, H2O_area
            self.results.loc[i, "rWS"] = H2O_area/ Si_area


    def batch_recalculate(self, save=True):
        for i, sample in self.spectra.items():

            H2O_left, H2O_right = self.processing.loc[i,["water_left", "water_right"]]
            H2O_bir = np.array([[1500, H2O_left], [H2O_right, 4000]])
            Si_birs_select = int(self.processing.loc[i, "Si_bir"])
            Si_bir = self.Si_birs[Si_birs_select]
            birs = np.concatenate((Si_bir, H2O_bir))

            sample.baselineCorrect(baseline_regions=birs, smooth_factor=1)
            sample.calculate_SiH2Oareas()
            Si_area, H2O_area = sample.SiH2Oareas

            if save:
                self.results.loc[i, ["SiArea", "H2Oarea"]] = Si_area, H2O_area
                self.results.loc[i, "rWS"] = H2O_area/ Si_area