import os, glob
import tkinter as tk
from tkinter import ttk
# Import all app elements
from settings import settings
from data_processing import data_processing
from water_calc import water_calc
from interpolation import interpolation
from subtraction import subtraction

# Some plot settings
import meltInc.plotting as p
fontsize = 6
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
        Main window
        """

        # Set theme
        style = ttk.Style()
        root.tk.call("source", f"{os.getcwd()}/theme/breeze.tcl")
        theme = "Breeze"
        style.theme_use(theme)
        style.configure(".", font="Verdana")
        # Grab some theme elements, for passing on to widgets
        self.font = style.lookup(theme, "font")
        self.bgClr = style.lookup(theme, "background")
        # calculate background color to something matplotlib understands
        self.bgClr_plt = tuple((c / 2 ** 16 for c in root.winfo_rgb(self.bgClr)))

        root.title("ramCOH by T. D. van Gerve")

        # Set some geometries
        root.geometry("800x900")
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
        self.water_calc = water_calc(panels, self)
        interpolate = interpolation(panels)
        subtract = subtraction(panels)
        # Put the frames on the grid
        panels.grid(column=0, row=0, sticky=("nesw"))
        self.water_calc.grid(column=0, row=0, sticky=("nesw"))
        interpolate.grid(column=0, row=0, sticky=("nesw"))
        subtract.grid(column=0, row=0, sticky=("nesw"))
        # Label the notebook tabs
        panels.add(self.water_calc, text="Baseline correction")
        panels.add(interpolate, text="Interpolation")
        panels.add(subtract, text="Crystal correction")
        # Adjust resizability
        panels.rowconfigure(0, weight=1)
        panels.columnconfigure(0, weight=1)

        ##### POPULATE SAMPLES FRAME #####
        # List with all samples
        self.samplesVar = tk.StringVar(value=[])
        self.sample_list = tk.Listbox(
            samples, listvariable=self.samplesVar, selectmode=tk.BROWSE, state=tk.DISABLED
        )
        self.sample_list.grid(column=0, row=0, columnspan=2, rowspan=6, sticky=("nesw"))
        # Scroll bar for the samples list
        sample_scroll = ttk.Scrollbar(
            samples, orient=tk.VERTICAL, command=self.sample_list.yview
        )
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
        self.sample_list.configure(state=tk.NORMAL)
        self.sample_list.selection_set(first=0)
        self.water_calc.initiate_plot(0)

    def select_sample(self, index):
        if index:
            selection = index[-1]
            self.sample = self.data.names[selection]
            self.water_calc.update_plot_sample(selection)

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


def main():

    root = tk.Tk()
    main_window(root)

    root.mainloop()


if __name__ == "__main__":
    main()
