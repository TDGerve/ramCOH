import matplotlib.pyplot as plt
import meltInc.plotting as p
from tkinter import ttk
import os


# Some plot settings
fontsize = 8
p.layout(
    colors=p.colors.bella,
    axTitleSize=fontsize,
    axLabelSize=fontsize,
    tickLabelSize=fontsize / 1.2,
    fontSize=fontsize,
)

class layout:
    """
    layout settings
    """    

    def __init__(self, root):
        # load some rcParams
                    # Set theme
        style = ttk.Style()
        root.tk.call("source", f"{os.getcwd()}/theme/breeze.tcl")
        theme = "Breeze"
        style.theme_use(theme)
        style.configure(".", font="Verdana")
        # Grab some theme elements, for passing on to widgets
        self.font = style.lookup(theme, "font")
        self.background = style.lookup(theme, "background")
        # calculate background color to something matplotlib understands
        self.background_plt = tuple((c / 2 ** 16 for c in root.winfo_rgb(self.bgClr)))

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