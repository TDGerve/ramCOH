import tkinter as tk
from tkinter import ttk
from RangeSlider.RangeSlider import RangeSliderH
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import ramCOH as ram




class app:

    def __init__(self, root, *args, **kwargs):

        style = ttk.Style()
        theme = "clam"
        style.theme_use(theme)
        self.font = style.lookup(theme, "font")
        self.bgClr = style.lookup(theme, "background")
        self.bgClr_plt = tuple((c/2**16 for c in root.winfo_rgb(self.bgClr)))
        print(self.bgClr_plt)

        self.root = root
        self.root.title("ramCOH by Thomas van Gerve")
        # Create frame
        panels = ttk.Notebook(root)
        main = main_panel(panels, font=self.font, bgClr=self.bgClr, bgClr_plt=self.bgClr_plt)
        interpolate = interpolate_panel(panels)
        subtract = subtract_panel(panels)
        # Put the frames on the grid
        panels.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        main.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        interpolate.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        subtract.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        # Label the notebook tabs
        panels.add(main, text="Main")
        panels.add(interpolate, text="interpolate")
        panels.add(subtract, text="subtract")


class main_panel(ttk.Frame):
    """
    widgets needed:
        - sample scale, or maybe listbox+scrollbar?
        - next-previous sample buttons
        - silicate region birs radiobuttons
        - water region birs scale
        - Store button

        - menubar:
        load files dialog window
        export data dialog window

    - notebook:
        olivine subtraction
        olivine interpolation
        main processing
    """

    def __init__(self, parent, *args, **kwargs):
        font = kwargs.pop("font")
        bgClr = kwargs.pop("bgClr")
        self.bgClr_plt = kwargs.pop("bgClr_plt")
        super().__init__(parent, *args, **kwargs)

        ##### Main tab widgets #####
        # List with all samples
        sample_list = tk.Listbox(self)
        sample_list.grid(
            column=0, row=0, rowspan=6, columnspan=2, sticky=(tk.N, tk.E, tk.S, tk.W)
        )
        
        self.plot()

        # Button to move through samples
        ttk.Button(self, text="Previous").grid(row=6, column=0, rowspan=2)
        ttk.Button(self, text="Next").grid(row=6, column=1, rowspan=2)

        # Rangeslider for H2O birs
        left = tk.DoubleVar()
        right = tk.DoubleVar()
        # slider_left = ttk.Scale(self, variable=left, orient="horizontal", from_=100, to=900)
        # slider_right = ttk.Scale(self, variable=right, orient="horizontal", from_=100, to=900)
        # slider_left.grid(column=2, row=6, columnspan=6, sticky=("swe"))
        # slider_left.set(500)
        # slider_right.grid(column=2, row=7, columnspan=6, sticky=("nwe"))
        # slider_right.set(900)

        slider = RangeSliderH(self, variables=[left, right], padX=12, font_family=font, bgColor=bgClr)
        slider.grid(column=2, row=7, columnspan=6, rowspan=2, sticky=("we"))
        

        for child in self.winfo_children():
            child.grid_configure(padx=5, pady=5)

    def plot(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 5), constrained_layout=True)
        fig.patch.set_facecolor(self.bgClr_plt)
        chart = FigureCanvasTkAgg(fig, self)
        chart.get_tk_widget().grid(
            row=0, column=2, rowspan=6, columnspan=5, sticky=(tk.N, tk.E, tk.S, tk.W)
        )


class interpolate_panel(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)


class subtract_panel(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)




def main():
    root = tk.Tk()    
    app(root)
    root.mainloop()


if __name__ == "__main__":
    main()
