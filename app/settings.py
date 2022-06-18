from tkinter import ttk
import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
import json


def import_birs():
    with open("birs.json") as f:
        birs_all = json.load(f)

    birs = {}

    for name, bir in birs_all.items():
        if bir["use"]:
            birs[name] = np.array(bir["birs"])

    return birs





class settings:
    """
    Settings window
    """
    # Raman settings
    laser_wavelength = None

    # File Settings
    name_separator = None

    # Default processing settings
    Si_bir = None
    H2O_left = None
    H2O_right = None
    interpolate_left = None
    interpolate_right = None


    def __init__(self, parent, app):
        self.app = app
        self.parent = parent

        self.import_settings()


        # File settings
        self.name_separator_var = tk.StringVar()
        self.name_separator_var.set(self.name_separator)
        # Raman Settings
        # self.laser_wavelength = 532.18
        self.laser_var = tk.DoubleVar()
        self.laser_var.set(self.laser_wavelength)
        
        
    def import_settings(self):
        with open("settings.json") as f:
            settings = json.load(f)

        for name, value in settings.items():
            setattr(self, name, value)

    def open_general_settings(self):

        font = (self.app.font, "16")

        popup = tk.Toplevel(self.parent)
        popup.title("Settings")
        window = ttk.Frame(popup)
        window.grid(column=0, row=0, sticky=("nesw"))
        popup.columnconfigure(0, weight=1)
        popup.rowconfigure(4, weight=1)
        # temparary variables
        laser_temp = tk.StringVar()
        separator_temp = tk.StringVar()
        # Create labels
        ttk.Label(window, text="Laser wavelength (nm)", font=font).grid(row=1, column=0)
        ttk.Label(window, text="File sample name separator", font=font).grid(row=3, column=0)
        ttk.Label(window, text="Current", font=font).grid(row=0, column=1)
        ttk.Label(window, text="New", font=font).grid(row=0, column=2)
        ttk.Label(window, textvariable=self.laser_var, font=font).grid(row=1, column=1)
        ttk.Label(window, textvariable=self.name_separator_var, font=font).grid(row=3, column=1)
        ttk.Label(window, text="Press Enter to store\nReload spectra to use new laser wavelength", font=font).grid(
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
            try:
                new_laser = float(laser_temp.get())
                self.laser_var.set(new_laser)
                self.laser_wavelength = new_laser
            except ValueError:
                self.parent.bell()
                return


        def store_separator(event):
            self.name_separator_var.set(separator_temp.get())
            
            if self.app.data:
                files = self.app.data.files
                names = ()
                
                names = self.app.data.get_names_from_files(files)
                self.app.data.names = names
                self.app.samplesVar.set(list(names))



        laser_entry.bind("<Return>", store_laser)
        separator_entry.bind("<Return>", store_separator)
        # Keep window on top
        popup.attributes("-topmost", True)