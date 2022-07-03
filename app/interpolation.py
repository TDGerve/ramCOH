from tkinter import W, ttk
import tkinter as tk
from turtle import settiltangle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


class VerticalNavigationToolbar2Tk(NavigationToolbar2Tk):
    def __init__(self, canvas, window):
        super().__init__(canvas, window, pack_toolbar=False)

    # override _Button() to re-pack the toolbar button in vertical direction
    def _Button(self, text, image_file, toggle, command):
        b = super()._Button(text, image_file, toggle, command)
        b.pack(side=tk.TOP)  # re-pack button in vertical direction
        return b

    # override _Spacer() to create vertical separator
    def _Spacer(self):
        s = tk.Frame(self, width=26, relief=tk.RIDGE, bg="DarkGray", padx=2)
        s.pack(side=tk.TOP, pady=5)  # pack in vertical direction
        return s

    # disable showing mouse position in toolbar
    def set_message(self, s):
        pass


class interpolation(ttk.Frame):
    # plot x limits
    xmin = 250
    xmax = 1400
    interpolation_colors = ["grey", "green"]
    

    # Initiate variables
    raw_spectrum = None
    interpolated = None
    interpolation_region = None
    # Object to store lines to be dragged
    _dragging_line = None
    # Store the id of the interpolation boundary being dragged, 0 for left, 1 for right
    _dragging_line_id = None

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.sample = None

        font = app.font
        fontsize = app.fontsize
        self.colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        plot_frame = ttk.Frame(self)
        toolbar_frame = ttk.Frame(self)
        settings_frame = ttk.Frame(self)
        plot_frame.grid(row=0, column=0, sticky=("nesw"))
        toolbar_frame.grid(row=0, column=1, sticky=("nesw"))
        settings_frame.grid(row=1, column=0, columnspan=2, sticky=("nesw"))
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        settings_frame.rowconfigure(1, weight=1)

        # Create plot canvas
        self.fig, self.ax = plt.subplots(
            figsize=(5.8, 4.5), constrained_layout=True, dpi=80
        )
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=("nesw"))
        # Plot navigation toolbar
        toolbar = VerticalNavigationToolbar2Tk(self.canvas, toolbar_frame)
        # Don't pack 'configure subplots' and 'save figure'
        toolbar.children["!button4"].pack_forget()
        toolbar.children["!button5"].pack_forget()
        toolbar.update()
        toolbar.grid(row=0, column=0, sticky="ne")
        self.fig.canvas.draw()
        # Plot axes
        self.ax.set_xlabel(" ")
        self.ax.set_ylabel("Intensity (arbitr. units)")
        self.ax.set_yticks([])
        self.ax.set_xlim(self.xmin, self.xmax)
        self.fig.patch.set_facecolor(app.bgClr_plt)

        # Widgets with interpolation settings
        # Interpolation checkbox
        self.itp_var = tk.BooleanVar()
        self.checkbox = ttk.Checkbutton(
            settings_frame,
            text="interpolate",
            variable=self.itp_var,
            onvalue=True,
            offvalue=False,
            command=self.interpolate_check
        )
        self.checkbox.grid(row=0, column=0, sticky=("nsw"))

        smoothing_label = ttk.Label(
            settings_frame, text="Interpolation smoothing", font=(font, fontsize)
        )        
        self.smoothing_var = tk.StringVar()
        self.smoothing_var.set(1)
        self.smoothing_spinbox = ttk.Spinbox(
            settings_frame,
            from_=0.1,
            to=10,
            increment=0.1,
            validate="focus",
            validatecommand=(self.register(self.validate_smoothing), "%P"),
            invalidcommand=(self.register(self.invalid_smoothing), "%P"),
            textvariable=self.smoothing_var,
            takefocus=1,
            width=5,
            font=(font, fontsize, "italic"),
        )
        smoothing_set = ttk.Button(
            settings_frame, text="Set", command=self.set_interpolation_smoothing
        )
        interpolation_reset = ttk.Button(settings_frame, text="Reset", command=self.reset_interpolation)
        smoothing_label.grid(row=0, column=2, sticky=("nse"))
        self.smoothing_spinbox.grid(row=0, column=1, sticky=("nsw"))
        smoothing_set.grid(row=1, column=1, columnspan=2, sticky=("n"))
        interpolation_reset.grid(row=0, column=3)

        for child in settings_frame.winfo_children():
            child.grid_configure(padx=10, pady=10)

        for child in self.winfo_children():
            child.grid_configure(padx=5, pady=5)        

    def initiate_plot(self):
        """ 
        
        """
        
        self.sample = self.app.current_sample
        self.old_spectrum = self.sample.spectra._spectrumSelect
        self.smoothing_var.set(self.sample.interpolation_smoothing)
        self.itp_var.set(str(self.sample.interpolate))
        if not hasattr(self.sample.spectra.signal, "interpolated"):
            self.sample.spectra.interpolate(
                interpolate=[self.sample.interpolate_left, self.sample.interpolate_right],
                smooth_factor=self.sample.interpolation_smoothing,
                use=False,
            )

        self.interpolate_lines = [
            self.ax.axvline(x, color="k", linewidth=1, visible=True)
            for x in [self.sample.interpolate_left, self.sample.interpolate_right]
        ]

        # Calculate ymax and set axis limits
        idx_xaxis = np.logical_and(
            self.xmax > self.sample.spectra.x, self.sample.spectra.x > self.xmin
        )
        y_max = np.max(self.sample.spectra.signal.raw[idx_xaxis]) * 1.2
        self.ax.set_ylim(0, y_max * 1.05)
        self.ax.set_xlim(self.xmin, self.xmax)
        # indeces for interpolation
        idx_interpolate = np.logical_and(
            self.sample.interpolate_right > self.sample.spectra.x,
            self.sample.spectra.x > self.sample.interpolate_left,
        )
        # Plot spectra
        (self.raw_spectrum,) = self.ax.plot(
            self.sample.spectra.x,
            self.sample.spectra.signal.raw,
            color=self.colors[0],
            label="raw",
        )
        (self.interpolated,) = self.ax.plot(
            self.sample.spectra.x[idx_interpolate],
            self.sample.spectra.signal.interpolated[idx_interpolate],
            color=self.colors[3],
            alpha=0.6,
            label="interpolated",
            visible=True,
        )
        # plot interpolation region
        self.interpolation_region = self.ax.axvspan(
            self.sample.interpolate_left,
            self.sample.interpolate_right,
            alpha=0.3,
            color=self.interpolation_colors[self.sample.interpolate],
            edgecolor=None,
        )

        # Connect mouse events to callback functions
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)

        self.canvas.draw()

    def update_plot(self):
        """ """
        self.sample = self.app.current_sample
        self.old_spectrum = self.sample.spectra._spectrumSelect
        self.smoothing_var.set(self.sample.interpolation_smoothing)
        self.itp_var.set(str(self.sample.interpolate))
        if not hasattr(self.sample.spectra.signal, "interpolated"):
            self.sample.recalculate_interpolation()

        self.raw_spectrum.set_data(
            self.sample.spectra.x, self.sample.spectra.signal.raw
        )

        for line, x in zip(self.interpolate_lines, (self.sample.interpolate_left, self.sample.interpolate_right)):
            line.set_xdata([x, x])

        # Calculate ymax and set axis limits
        idx_xaxis = np.logical_and(
            self.xmax > self.sample.spectra.x, self.sample.spectra.x > self.xmin
        )
        y_max = np.max(self.sample.spectra.signal.raw[idx_xaxis]) * 1.2
        self.ax.set_ylim(0, y_max * 1.05)
        self.ax.set_xlim(self.xmin, self.xmax)

        self.update_interpolation_regions()
        self.draw_interpolation()

    def update_interpolation_regions(self):
        """ 
        """
        polygon = np.array(
            [
                [self.sample.interpolate_left, 0.0],
                [self.sample.interpolate_left, 1.0],
                [self.sample.interpolate_right, 1.0],
                [self.sample.interpolate_right, 0.0],
            ]
        )
        self.interpolation_region.set_xy(polygon)
        self.interpolation_region.set(color=self.interpolation_colors[self.sample.interpolate])
        self.fig.canvas.draw_idle()

    def draw_interpolation(self):
        """ 
        """
        idx_interpolate = np.logical_and(
            self.sample.interpolate_right > self.sample.spectra.x,
            self.sample.spectra.x > self.sample.interpolate_left,
        )
        self.sample.recalculate_interpolation()

        self.interpolated.set_data(
            self.sample.spectra.x[idx_interpolate],
            self.sample.spectra.signal.interpolated[idx_interpolate],
        )
        self.fig.canvas.draw_idle()

    def interpolate_check(self):
        
        self.sample.interpolate = self.itp_var.get()
        self.interpolation_region.set(color=self.interpolation_colors[self.sample.interpolate])
        self.fig.canvas.draw_idle()

    def save_interpolation(self):
        if self.sample:
            self.sample.save_interpolation_settings()
            if self.sample.interpolate:
                self.sample.spectra._spectrumSelect = "interpolated"
            else:
                self.sample.spectra._spectrumSelect = self.old_spectrum
            self.sample.spectra.longCorrect()

    def reset_interpolation(self):
        # Read old settings
        self.sample.read_interpolation()
        # Redraw complete plot
        self.update_plot()


    def validate_smoothing(self, value):
        """
        Return False if the value is not numeric and reset the validate command if not.
        Resetting validate is neccessary, because tkinter disables validation after changing
        the variable through the invalidate command in order to stop an infinte loop.

        If the value is numerical clip it to 0, 10
        """
        try:
            value_clipped = np.clip(float(value), 0, 1000)
            self.smoothing_var.set(value_clipped)
            valid = True
        except ValueError:
            valid = False
        if not valid:
            # self.bell()
            self.smoothing_spinbox.after_idle(
                lambda: self.smoothing_spinbox.config(validate="focus")
            )
        return valid

    def invalid_smoothing(self, value):
        self.smoothing_var.set(1)

    def set_interpolation_smoothing(self):
        if self.sample:
            smoothing = float(self.smoothing_var.get())
            self.sample.interpolation_smoothing = smoothing
            self.draw_interpolation()

    def _on_click(self, event):
        """
        callback method for mouse click event
        """
        # left click
        if event.button == 1 and event.inaxes in [self.ax]:
            line = self._find_neighbor_line(event)
            if line:
                self._dragging_line = line

    def _on_release(self, event):
        """
        Callback method for mouse release event
        """
        if event.button == 1 and event.inaxes in [self.ax] and self._dragging_line:
            new_x = event.xdata
            self.interpolate_lines[self._dragging_line_id] = self._dragging_line

            # self._dragging_line.remove()
            id = self._dragging_line_id
            if id == 0:
                self.sample.interpolate_left = round(new_x, -1)
            elif id == 1:
                self.sample.interpolate_right = round(new_x, -1)
            self._dragging_line = None
            self._dragging_line_id = None
            # Recalculate and refresh interpolation
            self.update_interpolation_regions()
            self.draw_interpolation()

    def _on_motion(self, event):
        """
        callback method for mouse motion event
        """
        if self._dragging_line:
            new_x = event.xdata
            if new_x:
                # self.fig.canvas.draw_idle()
                id = self._dragging_line_id
                if id == 0:
                    if new_x > self.sample.interpolate_right:
                        new_x = self.sample.interpolate_right - 20
                    self.sample.interpolate_left = new_x
                elif id == 1:
                    if new_x < self.sample.interpolate_left:
                        new_x = self.sample.interpolate_left + 20
                    self.sample.interpolate_right = new_x
                y = self._dragging_line.get_ydata()
                self._dragging_line.set_data([new_x, new_x], y)
                # Recalculate and refresh interpolation
                self.update_interpolation_regions()
                self.draw_interpolation()

    def _find_neighbor_line(self, event):
        """
        Find lines around mouse position
        :rtype: ((int, int)|None)
        :return: (x, y) if there are any point around mouse else None
        """
        distance_threshold = 10
        nearest_line = None
        for i, line in enumerate(self.interpolate_lines):
            x = line.get_xdata()[0]
            distance = abs(event.xdata - x)
            if distance < distance_threshold:
                nearest_line = line
                self._dragging_line_id = i
        return nearest_line
