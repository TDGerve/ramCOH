import os, glob
from re import A
import numpy as np
import tkinter as tk
from tkinter import ttk
from RangeSlider.RangeSlider import RangeSliderH
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import ramCOH as ram
import meltInc.plotting as p


# Some plot settings
fontsize = 8
p.layout(
    colors=p.colors.bella,
    axTitleSize=fontsize,
    axLabelSize=fontsize,
    tickLabelSize=fontsize / 1.2,
    fontSize=fontsize,
)



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
        root.tk.call('source', f"{os.getcwd()}/breeze/breeze.tcl")
        theme = "Breeze"
        style.theme_use(theme)
        style.configure(".", font="Verdana")
        # Grab some theme elements, for passing on to widgets
        self.font = style.lookup(theme, "font")
        print(self.font)
        self.bgClr = style.lookup(theme, "background")
        self.bgClr_plt = tuple((c / 2 ** 16 for c in root.winfo_rgb(self.bgClr)))

        root.title("ramCOH by T. D. van Gerve")

        
        # Set some geometries
        root.geometry("800x1000")
        root.resizable(True, True)
        sizegrip = ttk.Sizegrip(root)
        sizegrip.grid(row=0, sticky=("se"))
        root.rowconfigure(0, weight=1)
        root.columnconfigure(3, weight=1)

        ##### INITIATE SETTINGS #####
        self.settings = settings(root, self)

        ##### CREATE MENU BAR #####
        # Prevent menu from tearting off
        root.option_add("*tearOff", False)
        menubar = tk.Menu(root)
        root["menu"] = menubar
        self.menu_file = tk.Menu(menubar)
        menu_settings = tk.Menu(menubar)
        menubar.add_cascade(menu=self.menu_file, label="File")
        menubar.add_cascade(menu=menu_settings, label="Settings")
        # File menu
        self.menu_file.add_command(label="Load data", command=self.load_data)
        self.menu_file.add_command(label="Export data", command=self.export_data)
        # disable data export on intialisation
        self.menu_file.entryconfigure("Export data", state=tk.DISABLED)
        # Settings menu
        menu_settings.add_command(label="Settings", command=self.settings.open_window)
        ##### CREATE FRAMES #####
        # Create the two main frames
        samples = ttk.Frame(root)
        main_frame = ttk.Frame(root)
        samples.grid(row=0, column=0, columnspan=2, rowspan=8, sticky=("nesw"))
        main_frame.grid(row=0, column=3, rowspan=8, columnspan=6, sticky=("nesw"))
        # Let the first row fill the frame
        samples.rowconfigure(0, weight=1)
        samples.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=4)

        # Create tabs inside the main frame
        panels = ttk.Notebook(main_frame)
        self.baseline = baseline_correction(panels, self)
        interpolate = interpolation(panels)
        subtract = subtraction(panels)
        # Put the frames on the grid
        panels.grid(column=0, row=0, sticky=("nesw"))
        self.baseline.grid(column=0, row=0, sticky=("nesw"))
        interpolate.grid(column=0, row=0, sticky=("nesw"))
        subtract.grid(column=0, row=0, sticky=("nesw"))
        # Label the notebook tabs
        panels.add(self.baseline, text="Baseline correction")
        panels.add(interpolate, text="Interpolation")
        panels.add(subtract, text="Crystal correction")
        # Adjust resizability
        panels.rowconfigure(0, weight=1)
        panels.columnconfigure(0, weight=1)


        ##### POPULATE SAMPLES FRAME #####
        # List with all samples
        self.samplesVar = tk.StringVar(value=[])
        self.sample_list = tk.Listbox(
            samples, listvariable=self.samplesVar, selectmode=tk.SINGLE
        )
        self.sample_list.grid(column=0, row=0, columnspan=2, rowspan=6, sticky=("nesw"))
        # Scroll bar for the samples list
        sample_scroll = ttk.Scrollbar(samples, orient=tk.VERTICAL, command=self.sample_list.yview)
        sample_scroll.grid(row=0, column=2, sticky=("ns"))
        self.sample_list["yscrollcommand"] = sample_scroll.set
        # Buttons to move through samples
        ttk.Button(samples, text="Previous", command=self.previous_sample).grid(
            row=6, column=0
        )
        ttk.Button(samples, text="Next", command=self.next_sample).grid(row=6, column=1)

        # Select a sample from the list
        self.sample = None
        self.sample_list.bind(
            "<<ListboxSelect>>",
            lambda event: self.select_sample(self.sample_list.curselection()),
        )

    def load_data(self):
        dirname = tk.filedialog.askdirectory(initialdir=os.getcwd())
        files = glob.glob(os.path.join(dirname, "*.txt"))
        self.data = data_processing(files, self.settings)
        self.data.preprocess()
        self.samplesVar.set(list(self.data.names))
        self.menu_file.entryconfigure("Export data", state=tk.NORMAL)
        # self.sample_list.configure(state=tk.NORMAL)
        self.sample_list.selection_set(first=0)
        self.select_sample(self.sample_list.curselection())
        print(self.sample_list.curselection())
        self.baseline.update_plot_sample(self.sample)

    def select_sample(self, index):
        selection = index[-1]
        self.sample = self.data.names[selection]
        self.baseline.update_plot_sample(self.sample)

    def next_sample(self):
        current = self.sample_list.curselection()[-1]
        total = self.sample_list.size()
        new = current + 1
        if current < total:
            self.sample_list.select_clear(current)
            self.sample_list.selection_set(new)
            self.sample_list.see(new)
            self.select_sample(self.sample_list.curselection())

    def previous_sample(self):
        current = self.sample_list.curselection()[-1]
        new = current - 1
        if current > 0:
            self.sample_list.select_clear(current)
            self.sample_list.selection_set(new)
            self.sample_list.see(new)
            self.select_sample(self.sample_list.curselection())

    def export_data():
        return


class settings:
    """
    Settings window
    """

    class color_palettes:
        flatDesign = plt.cycler(
            color=["#e27a3d", "#344d5c", "#df5a49", "#43b29d", "#efc94d"]
        )
        vitaminC = plt.cycler(
            color=["#FD7400", "#004358", "#FFE11A", "#1F8A70", "#BEDB39"]
        )

        bella = plt.cycler(
            color=["#801637", "#047878", "#FFB733", "#F57336", "#C22121"]
        )

        buddha = plt.cycler(
            color=["#192B33", "#FF8000", "#8FB359", "#FFD933", "#CCCC52"]
        )

        elemental = plt.cycler(
            color=["#E64661", "#FFA644", "#998A2F", "#2C594F", "#002D40"]
        )

        carolina = plt.cycler(
            color=["#73839C", "#2E4569", "#AECCCF", "#D5957D", "#9C7873"]
        )

        fourtyTwo = plt.cycler(
            color=["#2469A6", "#C4E1F2", "#F2E205", "#F2D22E", "#D9653B"]
        )

        terrazaverde = plt.cycler(
            color=["#DFE2F2", "#88ABF2", "#4384D9", "#56BFAC", "#D9B341"]
        )

    def __init__(self, parent, app):
        self.parent = parent
        self.name_separator = tk.StringVar()
        self.laser = tk.DoubleVar()
        self.laser.set(532.18)
        self.name_separator.set("_")
        self.colors = "bella"
        self.scale = 1.2

        parent.tk.call('tk', 'scaling', self.scale)

    def open_window(self):
        popup = tk.Toplevel(self.parent)
        popup.title("Settings")
        window = ttk.Frame(popup)
        window.grid(column=0, row=0, sticky=("nesw"))
        popup.columnconfigure(0, weight=1)
        popup.rowconfigure(4, weight=1)
        # temparary variables
        laser_temp = tk.DoubleVar()
        separator_temp = tk.StringVar()
        # Create labels
        ttk.Label(window, text="Laser wavelength (nm)").grid(row=1, column=0)
        ttk.Label(window, text="File sample name separator").grid(row=3, column=0)
        ttk.Label(window, text="Current").grid(row=0, column=1)
        ttk.Label(window, text="New").grid(row=0, column=2)
        ttk.Label(window, textvariable=self.laser).grid(row=1, column=1)
        ttk.Label(window, textvariable=self.name_separator).grid(row=3, column=1)
        ttk.Label(window, text="Press Enter to store").grid(
            row=4, column=0, columnspan=3, sticky=("nesw")
        )
        # Create entry fields
        laser_entry = ttk.Entry(window, textvariable=laser_temp)
        separator_entry = ttk.Entry(window, textvariable=separator_temp)
        laser_entry.grid(row=1, column=2, sticky=("we"))
        separator_entry.grid(row=3, column=2, sticky=("we"))

        for child in window.winfo_children():
            child.grid_configure(padx=5, pady=5)

        def store_laser(event):
            self.laser.set(laser_temp.get())

        def store_separator(event):
            self.name_separator.set(separator_temp.get())

        laser_entry.bind("<Return>", store_laser)
        separator_entry.bind("<Return>", store_separator)
        # Keep window on top
        popup.attributes("-topmost", True)


class baseline_correction(ttk.Frame):
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


class interpolation(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)


class subtraction(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)


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


def main():

    root = tk.Tk()
    main_window(root)

    root.mainloop()


if __name__ == "__main__":
    main()
