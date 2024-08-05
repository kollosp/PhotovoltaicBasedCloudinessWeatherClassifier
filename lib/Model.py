import numpy as np
from datetime import datetime as dt
from typing import List
from .Optimized import Optimized
from .Plotter import Plotter
from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity

np.set_printoptions(threshold=np.inf)


class ApplyKde:
    def __init__(self, kernel, bandwidth, bins):
        self._kernel = kernel
        self._bandwidth = bandwidth
        self._bins = bins

    def __call__(self, a):
        log_dens = KernelDensity(kernel=self._kernel, bandwidth=self._bandwidth) \
            .fit(a[~np.isnan(a)].reshape(-1, 1)) \
            .score_samples(self._bins)

        return np.exp(log_dens)


class Model:
    def __init__(self,
                 latitude_degrees: float,
                 longitude_degrees: float,
                 x_bins: int = 10,
                 y_bins: int = 10,
                 bandwidth: float = 0.2,
                 window_size: int = None,
                 enable_debug_params: bool = False):
        self._x_bins = x_bins
        self._bandwidth = bandwidth
        self._y_bins = y_bins
        self._latitude_degrees = latitude_degrees
        self._longitude_degrees = longitude_degrees
        self._model_representation = None
        self._elevation_bins = None
        self._overlay = None
        self._heatmap = None
        self._kde = None
        self._enable_debug_params = enable_debug_params
        self._ws = window_size  # if set then fit function performs moving avreage on the input data

    def fit(self, ts: np.ndarray, data: np.ndarray):
        if self._ws is not None:
            data = Optimized.window_moving_avg(data, window_size=self._ws, roll=True)
        # calculate elevation angles for the given timestamps
        elevation = Optimized.elevation(Optimized.from_timestamps(ts), self._latitude_degrees,
                                        self._longitude_degrees) * 180 / np.pi
        # remove negative timestamps
        elevation[elevation <= 0] = 0
        # create assignment series, which will be used in heatmap processing
        days_assignment = Optimized.date_day_bins(ts)
        elevation_assignment, self._elevation_bins = Optimized.digitize(elevation, self._x_bins)
        self._overlay = Optimized.overlay(data, elevation_assignment, days_assignment)
        self._heatmap = np.apply_along_axis(lambda a: np.histogram(a[~np.isnan(a)], bins=40)[0], 0, self._overlay)

        self._heatmap = np.apply_along_axis(lambda a: np.histogram(a[~np.isnan(a)], bins=self._y_bins)[0], 0,
                                            self._overlay)
        self._heatmap = np.apply_along_axis(lambda a: (100 * a / np.nansum(a)).astype(int), 0, self._heatmap)

        r = (0, data.max())
        bins_no = self._y_bins
        bins = np.array([r[0] + (r[1] - r[0]) * i / (bins_no - 1) for i in range(bins_no)]).reshape(-1, 1)  # bins len

        apply_kde = ApplyKde(kernel="gaussian", bandwidth=self._bandwidth, bins=bins)
        self._kde = np.apply_along_axis(apply_kde, 0, self._overlay)
        self._model_representation = np.apply_along_axis(lambda a: bins[np.argmax(a)], 0, self._kde).flatten()
        #
        # plt.show()

        # print("model", self._model_representation.shape)

    def plot(self):
        fig, ax = plt.subplots(3)
        Plotter.plot_overlay(self._overlay, fig=fig, ax=ax[0])
        x = list(range(self._overlay.shape[1]))
        ax[0].plot(x, self._model_representation, color="r")

        # compute mean values
        # mean = np.apply_along_axis(lambda a: np.nanmean(), 0, self._overlay)
        mean = np.nanmean(self._overlay, axis=0)
        mx = np.nanmax(self._overlay, axis=0)
        mi = np.nanmin(self._overlay, axis=0)
        ax[0].plot(x, mean, color="orange")
        ax[0].plot(x, mx, color="orange")
        ax[0].plot(x, mi, color="orange")

        ax[1].imshow(self._heatmap, cmap='Blues', origin='lower')
        ax[2].imshow(self._kde, cmap='Blues', origin='lower')
        return fig, ax

    def predict(self, ts: np.ndarray):
        if self._model_representation is None:
            raise RuntimeError("Model.predict: Use fit method first!")

        elevation = Optimized.elevation(Optimized.from_timestamps(ts), self._latitude_degrees,
                                        self._longitude_degrees) * 180 / np.pi

        return Optimized.model_assign(self._model_representation, self._elevation_bins, elevation, self._enable_debug_params)

    def __str__(self):
        return "Model representation: " + str(self._model_representation) + \
            " len(" + str(len(self._model_representation)) + ")" + \
            "\nBins: " + str(self._elevation_bins) + " len(" + str(len(self._elevation_bins)) + ")"
