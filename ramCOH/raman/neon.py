from .baseclass import RamanProcessing
import numpy as np
from ..signal_processing import functions as f
import itertools as it


class neon(RamanProcessing):
    # Baseline regions
    birs_default = np.array(
        [
            [1027, 1108],
            [1118, 1213],
            [1221, 1303],
            [1312, 1390],
            [1401, 1435],
            [1450, 1464],
        ]
    )


    def deconvolve(
        self,
        *,
        peak_prominence=2,
        noise_threshold=10,
        threshold_scale=0.3,
        baseline0=True,
        min_amplitude=8,
        min_peak_width=0.5,
        fit_window=4,
        noise=None,
        max_iterations=3,
        **kwargs,
    ):
        return super().deconvolve(
            peak_prominence=peak_prominence,
            noise_threshold=noise_threshold,
            threshold_scale=threshold_scale,
            baseline0=baseline0,
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
        laser = kwargs.get("laser", self.laser)

        if not hasattr(self, "peaks"):
            raise NameError("peaks not found, run fitPeaks first")

        neonEmission = f.neonEmission(laser=laser)
        left = np.round(
            np.float(
                neonEmission.iloc[:, 4][
                    np.isclose(left_nm, neonEmission.iloc[:, 1], atol=0.001)
                ]
            ),
            2,
        )
        right = np.round(
            np.float(
                neonEmission.iloc[:, 4][
                    np.isclose(right_nm, neonEmission.iloc[:, 1], atol=0.001)
                ]
            ),
            2,
        )

        # All emission lines within spectrum limits
        neonEmissionTrim = np.array(
            neonEmission.iloc[:, 4][
                (neonEmission.iloc[:, 4] > self.x.min())
                & (neonEmission.iloc[:, 4] < self.x.max())
            ]
        )

        # theoretical emission line positions with a match found in spectrum
        self.peakEmission = np.array([])
        # measured emission line positions
        self.peakMeasured = np.array([])

        for i, j in self.peaks.items():
            peak = j["center"]
            emissionCheck = np.isclose(peak, neonEmissionTrim, atol=search_window)

            if emissionCheck.sum() == 1:
                self.peakEmission = np.append(
                    self.peakEmission, neonEmissionTrim[emissionCheck].tolist()
                )
                self.peakMeasured = np.append(self.peakMeasured, peak)
            elif emissionCheck.sum() > 1:
                print(
                    "multiple emission line fits foundfor peak: "
                    + str(round(peak, 2))
                    + " cm-1"
                )

        # find correction factor for the calibration lines
        if np.isin([left, right], np.round(self.peakEmission, 2)).sum() == 2:
            # boolean array for the location of left and right calibration lines
            calibration_lines = np.array(
                np.isclose(left, self.peakEmission, atol=0.001)
                + np.isclose(right, self.peakEmission, atol=0.001)
            )
            # boolean to index
            calibration_lines = list(
                it.compress(range(len(calibration_lines)), calibration_lines)
            )
            # indices for differenced array
            calibration_lines = np.unique(calibration_lines - np.array([0, 1]))

            self.correctionFactor = np.float(
                np.sum(np.diff(self.peakEmission)[calibration_lines])
                / np.sum(np.diff(self.peakMeasured)[calibration_lines])
            )
            self.offset = np.float(
                self.peakMeasured[np.isclose(left, self.peakMeasured, atol=10)]
                - self.peakEmission[np.isclose(left, self.peakEmission, atol=0.1)]
            )

        else:
            print("calibration lines not found in spectrum")
