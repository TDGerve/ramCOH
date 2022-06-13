import numpy as np
import tkinter as tk
from tkinter import ttk
from RangeSlider.RangeSlider import RangeSliderH
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import ramCOH as ram


class main_window:
    def __init__(self, root, *args, **kwargs):
        """
        Widgets needed:
        - menubar:
        load files dialog window
            process file names
        export data dialog window

        - notebook:
        olivine subtraction
        olivine interpolation
        main processing
        """
        # Set theme
        style = ttk.Style()
        theme = "default"
        style.theme_use(theme)
        # Grab some theme elements, for passing on to widgets
        self.font = style.lookup(theme, "font")
        self.bgClr = style.lookup(theme, "background")
        self.bgClr_plt = tuple((c / 2 ** 16 for c in root.winfo_rgb(self.bgClr)))


        root.title("ramCOH by T. D. van Gerve")

        ##### CREATE FRAMES #####
        # Create the two main frames
        samples = ttk.Frame(root)
        main_frame = ttk.Frame(root)
        samples.grid(row=0, column=0, columnspan=2, rowspan=8, sticky=("nesw"))
        main_frame.grid(row=0, column=3, rowspan=8, columnspan=6)
        # Let the first row fill the frame
        samples.rowconfigure(0, weight=1)

        # Create tabs inside the main frame
        panels = ttk.Notebook(main_frame)
        baseline = baseline_correction(panels, self)
        interpolate = interpolation(panels)
        subtract = subtraction(panels)
        # Put the frames on the grid
        panels.grid(column=0, row=0, sticky=("nesw"))
        baseline.grid(column=0, row=0, sticky=("nesw"))
        interpolate.grid(column=0, row=0, sticky=("nesw"))
        subtract.grid(column=0, row=0, sticky=("nesw"))
        # Label the notebook tabs
        panels.add(baseline, text="Baseline correction")
        panels.add(interpolate, text="Interpolation")
        panels.add(subtract, text="Crystal correction")

        ##### POPULATE SAMPLES FRAME #####
        # List with all samples
        sample_list = tk.Listbox(samples)
        sample_list.grid(
            column=0, row=0, columnspan=2, rowspan=6, sticky=("nesw")
        )
        # Buttons to move through samples
        ttk.Button(samples, text="Previous").grid(row=6, column=0)
        ttk.Button(samples, text="Next").grid(row=6, column=1)


class baseline_correction(ttk.Frame):
    """
    widgets needed:
        - silicate region birs radiobuttons
        - water region birs scale
        - Store button
    """

    def __init__(self, parent, app, *args, **kwargs):

        super().__init__(parent, *args, **kwargs)
        ##### WIDGETS #####
        # Main plot
        self.plot(app)  

        # Rangeslider for H2O birs
        left = tk.DoubleVar()
        right = tk.DoubleVar()
        slider = RangeSliderH(
            self,
            variables=[left, right],
            padX=12,
            font_family=app.font,
            bgColor=app.bgClr,
        )
        slider.grid(column=2, row=7, columnspan=6, rowspan=2, sticky=("we"))

        for child in self.winfo_children():
            child.grid_configure(padx=5, pady=5)

    def plot(self, app):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 5), constrained_layout=True)
        fig.patch.set_facecolor(app.bgClr_plt)
        chart = FigureCanvasTkAgg(fig, self)
        chart.get_tk_widget().grid(
            row=0, column=2, rowspan=6, columnspan=5, sticky=(tk.N, tk.E, tk.S, tk.W)
        )


class interpolation(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)


class subtraction(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)


class data_processing():

    def __init__(self, files, type="H2O", laser=532.18, name_separator="_"):
        self.files = files
        self.names = [i[:i.find(name_separator)] for i in files]
        self.spectra = {}
        self.laser = laser
        self.model = getattr(ram, type)

    def preprocess(self):
        for name, f in zip(self.names, self.files):
            x, y = np.genfromtxt(f, unpack=True)
            self.spectra[name] = self.model(x, y, laser=self.laser)


def main():

    root = tk.Tk()
    main_window(root)

    root.mainloop()


if __name__ == "__main__":
    main()
