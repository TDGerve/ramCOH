from tkinter import ttk
import tkinter as tk
import matplotlib.pyplot as plt


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