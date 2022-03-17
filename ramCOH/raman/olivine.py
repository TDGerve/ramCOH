from . import water as h
import numpy as np


class olivine(h.H2O):
    # Baseline regions
    birs = np.array(
        [[100, 185], [260, 272], [370, 380], [470, 515], [660, 700], [1100, 4000]]
    )

    def __init__(self, x, intensity):

        super().__init__(x, intensity)

    def deconvolve(
        self,
        peak_prominence=4,
        noise_threshold=1.6,
        threshold_scale=0.2,
        min_amplitude=4,
        min_peak_width=6,
        fit_window=4,
        max_iterations=5,
        cutoff=1400,
        **kwargs,
    ):
        super().deconvolve(
            min_peak_width=min_peak_width,
            peak_prominence=peak_prominence,
            noise_threshold=noise_threshold,
            threshold_scale=threshold_scale,
            min_amplitude=min_amplitude,
            fit_window=fit_window,
            max_iterations=max_iterations,
            cutoff=cutoff,
            **kwargs,
        )
