

from sympy import interpolate


class current_sample():

    def __init__(self, dataset, index):
        self.index = index
        self.spectra = dataset.spectra[index]
        self.name = dataset.names[index]
        # Baseline interpolation region settings
        self.Si_birs_select = dataset.processing.loc[index, "Si_bir"]
        self.H2O_left = dataset.processing.loc[index, "water_left"]
        self.H2O_right = dataset.processing.loc[index, "water_right"]

        # Interpolation settings
        self.interpolate = dataset.processing.loc[index, "interpolate"]
        self.interpolate_left = dataset.processing.loc[index, "interpolate_left"]
        self.interpolate_right = dataset.processing.loc[index, "interpolate_right"]
        # Calculated areas
        self.Si_area, self.H2O_area = self.spectra.SiH2Oareas

    def save_sample_settings(self, dataset):

        dataset.processing.loc[
            self.index, "Si_bir"
        ] = self.Si_birs_select
        dataset.processing.loc[
            self.index, ["water_left", "water_right"]
        ] = round(self.H2O_left, -1), round(self.H2O_right, -1)

        dataset.results.loc[self.index, ["SiArea", "H2Oarea"]] = (
            self.Si_area,
            self.H2O_area,
        )
        dataset.results.loc[self.index, "rWS"] = (
            self.H2O_area / self.Si_area
        )

    def recalculate_areas(self):
        self.spectra.calculate_SiH2Oareas()
        self.Si_area, self.H2O_area = self.spectra.SiH2Oareas
