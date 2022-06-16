from tkinter import DoubleVar, ttk
import tkinter as tk
from RangeSlider import RangeSliderH
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from data_processing import data_processing


class water_calc(ttk.Frame):
    """
    widgets needed:
        - silicate region birs radiobuttons
        - water region birs scale
        - Store button
    """
    sample = None
    sample_index = None
    # Baseline regions
    Si_birs_select = None
    H2O_left = None
    H2O_right = None
    # Peak areas
    Si_area = None
    H2O_area = None
    

    def __init__(self, parent, app, *args, **kwargs):

        super().__init__(parent, *args, **kwargs)
        self.app = app
        # Frame settings
        self.rowconfigure(0, weight=1)
        # self.rowconfigure(7, weight=1)
        self.columnconfigure(0, weight=4)
        self.columnconfigure(1, weight=1)

        plot_frame = ttk.Frame(self)
        buttons_frame = ttk.Frame(self)
        plot_frame.grid(row=0, column=0, rowspan=8, sticky=("nesw"))
        buttons_frame.grid(row=0, column=1, rowspan=8, sticky=("nesw"))
        # Plot frame geometry
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=4)
        # Button frame geometry
        buttons_frame.rowconfigure(0, weight=1)
        buttons_frame.rowconfigure(4, weight=4)
        buttons_frame.rowconfigure(2, weight=3)
        for i in (3, 5, 6):
            buttons_frame.rowconfigure(i, weight=1)
        buttons_frame.rowconfigure(7, weight=2)
        buttons_frame.rowconfigure(8, weight=2)

        # buttons_frame.columnconfigure(0, weight=1
        for i in range(3):
            buttons_frame.columnconfigure(i, weight=1)

        widget_width = 220
        ###### RADIOBUTTONS #####
        self.bir_var = DoubleVar()
        self.bir_0 = ttk.Radiobutton(
            buttons_frame, text="2 BIRs", variable=self.bir_var, value=0, command=self.update_Si_birs, state=tk.DISABLED
        )
        self.bir_1 = ttk.Radiobutton(
            buttons_frame, text="1 BIR", variable=self.bir_var, value=1, command=self.update_Si_birs, state=tk.DISABLED
        )
        self.bir_0.grid(row=0, column=0, columnspan=2, sticky=("ews"), padx=5, pady=5)
        self.bir_1.grid(row=1, column=0, columnspan=2, sticky=("new"), padx=5, pady=5)

        ###### AREA DISPLAY WIDGETS #####
        label = ttk.Label(text="Calculated area", font=(self.app.font, 16, "bold"))
        areas_frame = ttk.Labelframe(
            buttons_frame, labelwidget=label, width=widget_width, height=100
        )
        areas_frame.grid(row=4, column=0, columnspan=2, sticky=("ew"))
        # force the size of all child widgets to the frame width
        areas_frame.grid_propagate(0)
        for i in range(2):
            areas_frame.rowconfigure(i, weight=1)
            areas_frame.columnconfigure(i, weight=1)
        # Text widgets
        font = (self.app.font, 14, "italic")
        subfont = (self.app.font, 8, "italic")
        Si_label = tk.Text(areas_frame, width=7, height=1, font=font)
        H2OSi_label = tk.Text(areas_frame, width=7, height=1, font=font)
        H2O_label = tk.Text(areas_frame, width=7, height=1, font=font)
        Si_label.grid(row=0, column=0, sticky=("sw"))
        H2O_label.grid(row=1, column=0, sticky=("sw"))
        H2OSi_label.grid(row=2, column=0, sticky=("sw"))
        # Si text
        Si_label.insert(tk.END, "Si")
        Si_label.configure(state=tk.DISABLED)
        # H2O text
        H2O_label.insert(tk.END, "H2O")
        H2O_label.tag_add("subscript", "1.1", "1.2")
        H2O_label.tag_configure("subscript", offset=-2, font=subfont)
        H2O_label.configure(state=tk.DISABLED)
        # H2O/Si text
        H2OSi_label.insert(tk.END, "H2O/Si")
        H2OSi_label.tag_add("subscript", "1.1", "1.2")
        H2OSi_label.tag_configure("subscript", offset=-2, font=subfont)
        H2OSi_label.configure(state=tk.DISABLED)
        # Label widgets
        self.Si_var = tk.StringVar()
        self.H2O_var = tk.StringVar()
        self.H2OSi_var = tk.StringVar()
        label_width = 8
        self.Si_area_label = ttk.Label(
            areas_frame, textvariable=self.Si_var, font=font, width=label_width
        )
        self.H2O_area_label = ttk.Label(
            areas_frame, textvariable=self.H2O_var, font=font, width=label_width
        )
        self.H2OSi_area_label = ttk.Label(
            areas_frame, textvariable=self.H2OSi_var, font=font, width=label_width
        )
        # Text widgets for units
        Si_units = tk.Text(areas_frame, width=5, height=1, font=font)
        H2O_units = tk.Text(areas_frame, width=5, height=1, font=font)
        # Grid widgets
        self.Si_area_label.grid(row=0, column=1, sticky=("es"))
        self.H2O_area_label.grid(row=1, column=1, sticky=("es"))
        self.H2OSi_area_label.grid(row=2, column=1, sticky=("es"))
        Si_units.grid(row=0, column=2, sticky=("ws"))
        H2O_units.grid(row=1, column=2, sticky=("ws"))
        # Units text
        for text in (Si_units, H2O_units):
            text.insert(tk.END, "x 10-2")
            text.tag_add("superscript", "1.4", "1.7")
            text.tag_configure("superscript", offset=8, font=subfont)
            text.configure(state=tk.DISABLED)
        # Separator line
        ttk.Separator(areas_frame, orient=tk.HORIZONTAL).grid(row=0, column=0, columnspan=3, sticky=("sew"))
        ttk.Separator(areas_frame, orient=tk.HORIZONTAL).grid(row=1, column=0, columnspan=3, sticky=("sew"))

        ###### BASELINE SMOOTHING WIDGETS #####
        baseline_label = ttk.Label(text="Baseline", font=(self.app.font, 16, "bold"))
        baseline_frame = ttk.Labelframe(
            buttons_frame, labelwidget=baseline_label, width=widget_width, height=80
        )
        baseline_frame.grid(row=5, column=0, columnspan=2, sticky=("nesw"))
        baseline_frame.grid_propagate(0)
        # Configure geometry
        baseline_frame.rowconfigure(0, weight=1)
        baseline_frame.rowconfigure(1, weight=1)
        baseline_frame.columnconfigure(0, weight=1)
        baseline_frame.columnconfigure(1, weight=1)
        # Spinbox widget for baseline smoothing
        self.smoothing_var = tk.StringVar()
        self.smoothing_var.set(1)
        smoothing_label = ttk.Label(baseline_frame, text="Smoothing", font=font)
        # Validate on changing focus
        self.smoothing_spinbox = ttk.Spinbox(
            baseline_frame,
            from_=0.1,
            to=10,
            increment=0.1,
            validate="focus",
            validatecommand=(self.register(self.validate_smoothing), "%P"),
            invalidcommand=(self.register(self.invalid_smoothing), "%P"),
            textvariable=self.smoothing_var,
            takefocus=1,
            width=10,
        )
        baseline_set = ttk.Button(baseline_frame, text="Set", command=self.set_baseline_smoothing)
        smoothing_label.grid(row=0, column=0, sticky=("w"))
        baseline_set.grid(row=1, column=0, columnspan=2, sticky=("ew"))
        self.smoothing_spinbox.grid(row=0, column=1, sticky=("e"))
        # Leave some space between the frame and the widgets
        for child in baseline_frame.winfo_children():
            child.grid_configure(padx=5, pady=5)


        ###### SAVE BUTTON #####
        self.save_button = ttk.Button(buttons_frame, text="Save sample", command=self.save_sample, state=tk.DISABLED)
        self.save_button.grid(
            row=8, column=0, columnspan=2, sticky=("ew"), padx=5, pady=5
        )

        ##### PLOT CANVAS #####
        # Plot colors
        self.colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        # Create plot canvas
        self.fig, (self.ax1, self.ax2) = plt.subplots(
            2, 1, figsize=(3, 7), constrained_layout=True, dpi=80
        )
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(
            row=0, column=0, rowspan=8, columnspan=5, sticky=("nesw")
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

        self.fig.canvas.draw()

        # Object to store lines to be dragged
        self._dragging_line = None
        # Store the id of the H2O bir being dragged, 0 for left, 1 for right
        self._dragging_line_id = None

        self.raw_spectra = []
        self.baselines = []
        self.corrected = []

        for child in self.winfo_children():
            child.grid_configure(padx=5, pady=5)

    def validate_smoothing(self, value):
        """
        Return False if the value is not numeric and reset the validate command if not.
        Resetting validate is neccessary, because tkinter disables validation after changing 
        the variable through the invalidate command in order to stop an infinte loop.

        If the value is numerical clip it to 0, 10
        """
        try:
            value_clipped = np.clip(float(value), 0, 100)
            self.smoothing_var.set(value_clipped)
            valid = True
        except ValueError:
            valid = False
        if not valid:
            # self.bell()
            self.smoothing_spinbox.after_idle(lambda: self.smoothing_spinbox.config(validate="focus"))
        return valid

    def invalid_smoothing(self, value):
        self.smoothing_var.set(1)


    def set_baseline_smoothing(self):
        smoothing = float(self.smoothing_var.get())
        self.app.data.smooth_factor = smoothing
        self.recalculate_baseline()

    def save_sample(self):

        self.app.data.processing.loc[self.sample_index, "Si_bir"] = self.Si_birs_select
        self.app.data.processing.loc[self.sample_index, ["water_left", "water_right"]] = round(self.H2O_left, -1), round(self.H2O_right, -1)

        self.app.data.results.loc[self.sample_index, ["SiArea", "H2Oarea"]] = self.Si_area, self.H2O_area
        self.app.data.results.loc[self.sample_index, "rWS"] = self.H2O_area/ self.Si_area


    def initiate_plot(self, index):
        """
        Docstring
        """

        self.sample_index = index

        self.sample = self.app.data.spectra[index]
        self.recalculate_areas()
        self.H2O_left, self.H2O_right = self.app.data.processing.loc[
            index, ["water_left", "water_right"]
        ]
        H2O_bir = np.array([[1500, self.H2O_left], [self.H2O_right, 4000]])
        self.Si_birs_select = int(self.app.data.processing.loc[index, "Si_bir"])
        self.bir_var.set(self.Si_birs_select)

        y_max_Si = np.max(self.sample.signal.long_corrected[self.sample.x < 1400]) * 1.2
        y_max_h2o = np.max(self.sample.signal.long_corrected[self.sample.x > 2500]) * 1.2

        self.ax1.set_ylim(0, y_max_Si * 1.05)
        self.ax2.set_ylim(0, y_max_h2o)

        self.fig.canvas.draw_idle()

        # Plot spectra
        for ax in (self.ax1, self.ax2):
            # Long corrected
            self.raw_spectra.append(
                ax.plot(
                    self.sample.x,
                    self.sample.signal.long_corrected,
                    color=self.colors[0],
                    linewidth=1.2,
                )
            )
            # Baseline
            self.baselines.append(
                ax.plot(
                    self.sample.x,
                    self.sample.baseline,
                    linestyle="dashed",
                    color=self.colors[2],
                    linewidth=1.2,
                )
            )
            # Baseline corrected
            self.corrected.append(
                ax.plot(
                    self.sample.x,
                    self.sample.signal.baseline_corrected,
                    color=self.colors[1],
                    linewidth=1.2,
                )
            )

        # Plot baseline interpolation regions
        # Silicate region
        Si_bir0_polygons = [
            self.ax1.axvspan(
                bir[0], bir[1], alpha=0.3, color="gray", edgecolor=None, visible=False
            )
            for bir in data_processing.Si_bir_0
        ]
        Si_bir1_polygons = [
            self.ax1.axvspan(
                bir[0], bir[1], alpha=0.3, color="gray", edgecolor=None, visible=False
            )
            for bir in data_processing.Si_bir_1
        ]
        self.Si_bir_polygons = [Si_bir0_polygons, Si_bir1_polygons]
        for polygon in self.Si_bir_polygons[self.Si_birs_select]:
            polygon.set_visible(True)
        # Water region
        self.H2O_bir_polygons = [
            self.ax2.axvspan(bir[0], bir[1], alpha=0.3, color="gray") for bir in H2O_bir
        ]
        self.H2O_bir_lines = [
            self.ax2.axvline(x, color="k", linewidth=1, visible=False)
            for x in [self.H2O_left, self.H2O_right]
        ]

        # Connect mouse events to callback functions
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)

        self.canvas.draw()

        # Activate buttons
        self.save_button.configure(state=tk.NORMAL)
        self.bir_0.configure(state=tk.NORMAL)
        self.bir_1.configure(state=tk.NORMAL)

    def update_plot_sample(self, index):
        """
        Docstring
        """
        self.sample_index = index
        self.sample = self.app.data.spectra[index]

        self.H2O_left, self.H2O_right = self.app.data.processing.loc[
            index, ["water_left", "water_right"]
        ]
        self.Si_birs_select = int(self.app.data.processing.loc[index, "Si_bir"])
        self.bir_var.set(self.Si_birs_select)

        y_max_Si = np.max(self.sample.signal.long_corrected[self.sample.x < 1400]) * 1.2
        y_max_h2o = np.max(self.sample.signal.long_corrected[self.sample.x > 2500]) * 1.2

        self.ax1.set_ylim(0, y_max_Si * 1.05)
        self.ax2.set_ylim(0, y_max_h2o)

        for i, _ in enumerate([self.ax1, self.ax2]):
            # Long corrected
            self.raw_spectra[i][0].set_data(
                self.sample.x, self.sample.signal.long_corrected
            )
            # Baseline
            self.baselines[i][0].set_data(self.sample.x, self.sample.baseline)
            # Baseline corrected
            self.corrected[i][0].set_data(
                self.sample.x, self.sample.signal.baseline_corrected
            )

        for line, x in zip(self.H2O_bir_lines, (self.H2O_left, self.H2O_right)):
            line.set_xdata([x, x])

        self.recalculate_baseline()
        self.update_H2O_birs()
        self.update_Si_birs()
        self.recalculate_areas()

    def update_H2O_birs(self):
        """
        Docstring
        """

        polygon_left = np.array(
            [[1500, 0.0], [1500, 1.0], [self.H2O_left, 1.0], [self.H2O_left, 0.0]]
        )
        polygon_right = np.array(
            [[self.H2O_right, 0.0], [self.H2O_right, 1.0], [4000, 1.0], [4000, 0.0]]
        )
        H2O_polygons_new = [polygon_left, polygon_right]
        for polygon_old, polygon_new in zip(self.H2O_bir_polygons, H2O_polygons_new):
            polygon_old.set_xy(polygon_new)

        self.fig.canvas.draw_idle()

    def update_Si_birs(self):
        """
        Docstring
        """
        self.Si_birs_select = int(self.bir_var.get())
 
        for polygon in self.Si_bir_polygons[self.Si_birs_select]:
            polygon.set_visible(True)
        for polygon in self.Si_bir_polygons[abs(self.Si_birs_select - 1)]:
            polygon.set_visible(False)
        self.recalculate_baseline()
        self.recalculate_areas()
        self.fig.canvas.draw_idle()


    def recalculate_baseline(self):
        """
        Docstring
        """
        smooth_factor = self.app.data.smooth_factor

        H2O_bir = np.array(
            [[1500, round(self.H2O_left, -1)], [round(self.H2O_right, -1), 4000]]
        )
        Si_bir = self.app.data.Si_birs[self.Si_birs_select]
        birs = np.concatenate((Si_bir, H2O_bir))

        self.sample.baselineCorrect(baseline_regions=birs, smooth_factor=smooth_factor)
        for i, _ in enumerate([self.ax1, self.ax2]):
            self.baselines[i][0].set_data(self.sample.x, self.sample.baseline)
            self.corrected[i][0].set_data(self.sample.x, self.sample.signal.baseline_corrected)

        self.recalculate_areas()
        self.fig.canvas.draw_idle()

    def recalculate_areas(self):
        """
        Docstring
        """

        self.sample.calculate_SiH2Oareas()
        self.Si_area, self.H2O_area = self.sample.SiH2Oareas
        self.Si_var.set(f"{self.Si_area * 1e2: .3f}")
        self.H2O_var.set(f"{self.H2O_area * 1e2: .3f}")
        self.H2OSi_var.set(f"{(self.H2O_area / self.Si_area): .3f}")

    def _on_click(self, event):
        """
        callback method for mouse click event
        """
        # left click
        if event.button == 1 and event.inaxes in [self.ax2]:
            line = self._find_neighbor_line(event)
            if line:
                self._dragging_line = line

    def _on_release(self, event):
        """
        Callback method for mouse release event
        """
        if event.button == 1 and event.inaxes in [self.ax2] and self._dragging_line:
            new_x = event.xdata
            self.H2O_bir_lines[self._dragging_line_id] = self._dragging_line
            self._dragging_line = None
            self._dragging_line_id = None
            # self._dragging_line.remove()
            id = self._dragging_line_id
            if id == 0:
                self.H2O_left = round(new_x, -1)
            elif id == 1:
                self.H2O_right = round(new_x, -1)
            self.recalculate_baseline()
            self.update_H2O_birs()

    def _on_motion(self, event):
        """
        callback method for mouse motion event
        """
        if self._dragging_line:
            new_x = event.xdata
            y = self._dragging_line.get_ydata()
            self._dragging_line.set_data([new_x, new_x], y)
            # self.fig.canvas.draw_idle()
            id = self._dragging_line_id
            if id == 0:
                if new_x > self.H2O_right:
                    new_x = self.H2O_right - 20
                self.H2O_left = new_x
            elif id == 1:
                if new_x < self.H2O_left:
                    new_x = self.H2O_left + 20
                self.H2O_right = new_x
            self.recalculate_baseline()
            self.update_H2O_birs()

    def _find_neighbor_line(self, event):
        """
        Find lines around mouse position
        :rtype: ((int, int)|None)
        :return: (x, y) if there are any point around mouse else None
        """
        distance_threshold = 10
        nearest_line = None
        for i, line in enumerate(self.H2O_bir_lines):
            x = line.get_xdata()[0]
            distance = abs(event.xdata - x)
            if distance < distance_threshold:
                nearest_line = line
                self._dragging_line_id = i
        return nearest_line
