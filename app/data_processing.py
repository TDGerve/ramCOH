import os
import ramCOH as ram
import pandas as pd
import numpy as np

class data_processing:

    # Silicate baseline interpolation regions
    Si_bir_0 = np.array([[20, 250], [640, 655], [800, 810], [1220, 1600]])
    Si_bir_1 = np.array([[20, 250], [640, 700], [1220, 1600]])
    Si_birs = [Si_bir_0, Si_bir_1]
    Si_birs_labels = ["2 birs", "1 bir"]

    def __init__(self, files, settings, type="H2O"):
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
        self.data = pd.DataFrame({"name": self.names})
        self.data["SiArea"] = np.zeros(self.data.shape[0])
        self.data["H2Oarea"] = np.zeros(self.data.shape[0])
        self.data["rWS"] = np.zeros(self.data.shape[0])

        # Create dataframe to store processing parameters
        self.processing = pd.DataFrame({"name": self.names, "interpolate": False})
        self.processing["interpolate_left"] = int(780)
        self.processing["interpolate_right"] = int(900)
        self.processing["Si_bir"] = int(0)
        self.processing["water_left"] = int(2800)
        self.processing["water_right"] = int(3850)

    def preprocess(self):
        for i, f in enumerate(self.files):
            x, y = np.genfromtxt(f, unpack=True)
            self.spectra[i] = self.model(x, y, laser=self.laser)
            self.spectra[i].longCorrect()
            self.spectra[i].baselineCorrect()
            self.spectra[i].calculate_SiH2Oareas()