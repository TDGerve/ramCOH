from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class interpolation(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.sample = None

        # Initiate variables
        self.raw_spectrum = None
        self.interpolated = None


        plot_frame = ttk.Frame(self)
        settings_frame = ttk.Frame(self)
        plot_frame.grid(row=0, column=0, sticky=("nesw"))
        settings_frame.grid(row=1, column=0, sticky=("nesw"))


        self.colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        # Create plot canvas
        self.fig, self.ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True, dpi=80
        )
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(
            row=0, column=0, rowspan=8, columnspan=5, sticky=("nesw")
        )
        self.ax.set_xlabel(" ")
        self.ax.set_ylabel("Intensity (arbitr. units)")
        self.ax.set_yticks([])
        self.ax.set_xlim(150, 1600)

        self.fig.patch.set_facecolor(app.bgClr_plt)

        self.fig.canvas.draw()

        for child in self.winfo_children():
            child.grid_configure(padx=10, pady=10)


    def initiate_plot(self):
        self.sample = self.app.current_sample

        self.interpolate_lines = [
            self.ax.axvline(x, color="k", linewidth=1, visible=False)
            for x in [self.sample.interpolate_left, self.sample.interpolate_right]
        ]


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
            self.H2O_bir_lines[self._dragging_line_id] = self._dragging_line
            self._dragging_line = None
            self._dragging_line_id = None
            # self._dragging_line.remove()
            id = self._dragging_line_id
            if id == 0:
                self.sample.interpolate_left = round(new_x, -1)
            elif id == 1:
                self.sample.interpolate_right = round(new_x, -1)
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
                if new_x > self.sample.interpolate_right:
                    new_x = self.sample.interpolate_right - 20
                self.sample.interpolate_left = new_x
            elif id == 1:
                if new_x < self.sample.interpolate_left:
                    new_x = self.sample.interpolate_left + 20
                self.sample.interpolate_right = new_x
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
