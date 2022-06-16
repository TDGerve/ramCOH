from tkinter import ttk
import tkinter as tk
import matplotlib.pyplot as plt


class settings:
    """
    Settings window
    """

    def __init__(self, parent, app):
        self.app = app
        self.parent = parent
        self.name_separator = tk.StringVar()
        self.laser = tk.DoubleVar()
        self.laser.set(532.18)
        self.name_separator.set("_")

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
        ttk.Label(window, textvariable=self.laser, font=font).grid(row=1, column=1)
        ttk.Label(window, textvariable=self.name_separator, font=font).grid(row=3, column=1)
        ttk.Label(window, text="Press Enter to store", font=font).grid(
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
                self.laser.set(new_laser)
            except ValueError:
                self.parent.bell()

        def store_separator(event):
            self.name_separator.set(separator_temp.get())

        laser_entry.bind("<Return>", store_laser)
        separator_entry.bind("<Return>", store_separator)
        # Keep window on top
        popup.attributes("-topmost", True)