import tkinter as tk
from tkinter import ttk
import ramCOH as ram


class FeetToMeters:
    def __init__(self, root):

        root.title("Feet to Meters")

        s = ttk.Style()
        s.configure(
            "Danger.TFrame",
            background="red",
            borderwidth=5,
            relief="raised",
            font="TkFixedFont",
        )

        mainframe = ttk.Frame(root, style="Danger.TFrame", padding="3 3 12 12")
        mainframe.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        self.feet = tk.StringVar()
        feet_entry = ttk.Entry(mainframe, width=7, textvariable=self.feet)
        feet_entry.grid(column=2, row=1, sticky=(tk.W, tk.E))
        self.meters = tk.StringVar()

        ttk.Label(mainframe, textvariable=self.meters).grid(
            column=2, row=2, sticky=(tk.W, tk.E)
        )
        ttk.Button(mainframe, text="Calculate", command=self.calculate).grid(
            column=3, row=3, sticky=tk.W
        )

        ttk.Label(mainframe, text="feet").grid(column=3, row=1, sticky=tk.W)
        ttk.Label(mainframe, text="is equivalent to").grid(column=1, row=2, sticky=tk.E)
        ttk.Label(mainframe, text="meters").grid(column=3, row=2, sticky=tk.W)

        for child in mainframe.winfo_children():
            child.grid_configure(padx=5, pady=5)

        feet_entry.focus()
        root.bind("<Return>", self.calculate)

    def calculate(self, *args):
        try:
            value = float(self.feet.get())
            self.meters.set(int(0.3048 * value * 10000.0 + 0.5) / 10000.0)
        except ValueError:
            pass


root = tk.Tk()
FeetToMeters(root)
root.mainloop()
