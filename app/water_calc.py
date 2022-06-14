from tkinter import ttk
import tkinter as tk
from RangeSlider import RangeSliderH
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np


class water_calc(ttk.Frame):
    """
    widgets needed:
        - silicate region birs radiobuttons
        - water region birs scale
        - Store button
    """

    def __init__(self, parent, app, *args, **kwargs):

        super().__init__(parent, *args, **kwargs)
        self.app = app
        # Frame settings
        self.rowconfigure(0, weight=1)
        self.rowconfigure(7, weight=1)
        self.columnconfigure(0, weight=1)

        ##### WIDGETS #####
        # Main plot

        # Rangeslider for H2O birs
        left = tk.DoubleVar()
        right = tk.DoubleVar()
        slider = RangeSliderH(
            self,
            variables=[left, right],
            padX=11,
            font_family=app.font,
            bgColor=app.bgClr,
        )
        slider.grid(column=0, row=7, columnspan=5, rowspan=2, sticky=("nesw"))

        for child in self.winfo_children():
            child.grid_configure(padx=5, pady=5)

        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # INITIATE PLOT
        self.fig, (self.ax1, self.ax2) = plt.subplots(
            2, 1, figsize=(4, 6), constrained_layout=True, dpi=100
        )
        self.ax1.set_title("Silicate region")
        self.ax1.set_xlabel(" ")
        self.ax1.set_ylabel("Intensity (arbitr. units)")
        self.ax1.set_yticks([])
        self.ax1.set_xlim(150, 1400)

        self.ax2.set_title("H$_2$O  region")
        self.ax2.set_yticks([])
        self.ax2.set_xlim(2700, 4000)
        self.ax2.set_xlabel("Raman shift cm$^{-1}$")

        self.fig.patch.set_facecolor(app.bgClr_plt)
        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.get_tk_widget().grid(
            row=0, column=0, rowspan=6, columnspan=5, sticky=("nesw")
        )
        self.canvas.draw()

    def update_plot_sample(self, sample):
        data = self.app.data.spectra[sample]
        y_max_Si = np.max(data.signal.long_corrected[data.x < 1400]) * 1.2
        y_max_h2o = np.max(data.signal.long_corrected[data.x > 2500]) * 1.2

        self.ax1.set_ylim(0, y_max_Si * 1.05)
        self.ax2.set_ylim(0, y_max_h2o)

        for ax in (self.ax1, self.ax2):
            # Remove old plotted lines
            for i, line in enumerate(ax.get_lines()):
                # ax.lines.pop(i)
                line.remove()

        self.fig.canvas.draw_idle()

        for ax in (self.ax1, self.ax2):
            # Long corrected
            ax.plot(data.x, data.signal.long_corrected, color=self.colors[0], linewidth=1.2)
            # Baseline corrected
            ax.plot(data.x, data.signal.baseline_corrected, color=self.colors[1], linewidth=1.2)
            # Baseline
            ax.plot(data.x, data.baseline, linestyle="dashed", color=self.colors[2], linewidth=1.2)

        self.canvas.draw()