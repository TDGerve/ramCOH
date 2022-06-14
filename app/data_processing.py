import os
import ramCOH as ram
import pandas as pd
import numpy as np

class data_processing:
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
        self.data = pd.DataFrame()
        self.data["name"] = self.names
        self.data["SiArea"] = np.zeros(self.data.shape[0])
        self.data["H2Oarea"] = np.zeros(self.data.shape[0])
        self.data["rWS"] = np.zeros(self.data.shape[0])

    def preprocess(self):
        for name, f in zip(self.names, self.files):
            x, y = np.genfromtxt(f, unpack=True)
            self.spectra[name] = self.model(x, y, laser=self.laser)
            self.spectra[name].longCorrect()
            self.spectra[name].baselineCorrect()
            self.spectra[name].calculate_SiH2Oareas()