import os.path

import matplotlib.pyplot as plt
import numpy as np
from lib.Optimized import Optimized
from lib.Plotter import Plotter
from lib.Dataset import Dataset
from lib import Time as tm
from matplotlib.colors import ListedColormap
from lib.Model import Model
from lib.Utils import Utils
from matplotlib.backends.backend_pdf import PdfPages

import datetime as dt
from datetime import datetime
from matplotlib.dates import DateFormatter
datetime_format = DateFormatter("%Y-%m-%d %H:%M:%S")
time_format = DateFormatter("%H:%M:%S")
date_format = DateFormatter("%Y-%m-%d")

roi = slice(4000+60, 6000) # period to be shown in images

FILENAME = "datasets/dataset_annotated.csv"
ts, production = Dataset.get(filename=FILENAME)
ts_dt = [datetime.fromtimestamp(t) for t in ts]
_, auto_weather = Dataset.get(field=4, filename=FILENAME)
_, weather = Dataset.get(field=7, filename=FILENAME)
weather = weather.astype(int)
latitude_degrees = 51
longitude_degrees = 14

elevation = Optimized.elevation(Optimized.from_timestamps(ts), latitude_degrees, longitude_degrees) * 180 / np.pi

m = Model(latitude_degrees, longitude_degrees, x_bins=40, y_bins=30)
m.fit(ts, production)

# predict data for selected timestamps
expected_from_model = m.predict(ts)  # in-sample prediction - considered as expected production
# expected_from_model = expected_from_model * production.max()/expected_from_model.max()
overall_cloudiness_index = Optimized.overall_cloudiness_index(production, expected_from_model, window_size=tm._5H)
# overall_cloudiness_index_2 = Optimized.overall_cloudiness_index(production, model, window_size=tm._2H)
variability_cloudiness_index = Optimized.variability_cloudiness_index(production, expected_from_model,
                                                                      window_size=6, ma_window_size=tm._1H)

classes_txt = ["Night", "Clear Sunny", "Semi-transparent clouds", "Full Cloud Cover",
                   "Sunny with long-term clouds", "Semi-transparent clouds with long-term clearings", "Full Cloud Cover with long-term clearings",
                   "Sunny with short-term clouds",  "Semi-transparent clouds with short-term clearings", "Full Cloud Cover with short-term clearings"]
classes_txt_abb = ["Night", "Clear Sunny", "Semi-transparent clouds cloudy", "Full cloud Cover",
                   "Sunny w. lt. clouds", "Semi. cloudy w. lt. clearings", "Full cloud c. w. lt. clearings",
                   "Sunny w. st. clouds",  "Semi. cloudy w. st. clearings", "Full cloud c. w. st. clearings"]

def align_zeros(axes):
    ylims_current = {}  # Current ylims
    ylims_mod = {}  # Modified ylims
    deltas = {}  # ymax - ymin for ylims_current
    ratios = {}  # ratio of the zero point within deltas

    for ax in axes:
        ylims_current[ax] = list(ax.get_ylim())
        # Need to convert a tuple to a list to manipulate elements.
        deltas[ax] = ylims_current[ax][1] - ylims_current[ax][0]
        ratios[ax] = -ylims_current[ax][0] / deltas[ax]

    for ax in axes:  # Loop through all axes to ensure each ax fits in others.
        ylims_mod[ax] = [np.nan, np.nan]  # Construct a blank list
        ylims_mod[ax][1] = max(deltas[ax] * (1 - np.array(list(ratios.values()))))
        # Choose the max value among (delta for ax)*(1-ratios),
        # and apply it to ymax for ax
        ylims_mod[ax][0] = min(-deltas[ax] * np.array(list(ratios.values())))
        # Do the same for ymin
        ax.set_ylim(tuple(ylims_mod[ax]))

def plot_production_and_weather():
    fig, ax = plt.subplots(1)
    unique_labels = np.unique(weather)
    ax.xaxis.set_major_formatter(date_format)
    ax.set_ylabel("Power [kW]")
    ax.set_xlabel("Time")
    ax.tick_params(axis='x', labelrotation=45)

    lns1 = ax.plot(ts_dt[roi], production[roi], c="b", label="Production")

    ax2 = ax.twinx()
    color2 = 'tab:green'
    ax2.set_ylabel('Weather', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_yticks(range(len(unique_labels)))
    ax.set_title("Production and annotated weather labels")
    ax2.set_yticklabels([f"#{u}" for u in unique_labels])
    lns2 = ax2.plot(ts_dt[roi], weather[roi], '.', color=color2, label="Labels")

    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs)

    fig.tight_layout()
    return fig


def plot_production_and_auto_weather():
    fig, ax = plt.subplots(1)
    unique_labels = np.unique(auto_weather)
    ax.xaxis.set_major_formatter(date_format)

    lns1 = ax.plot(ts_dt[roi], production[roi], c="b", label="Production")
    ax.set_ylabel("Power [kW]")
    ax.set_xlabel("Time")
    ax.tick_params(axis='x', labelrotation=45)

    ax2 = ax.twinx()
    color2 = 'tab:green'
    ax2.set_ylabel('Weather', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_yticks(range(len(unique_labels)))

    ax.set_title("Production and auto weather labels")
    labels = ["Cloudy", "Heavy Rain", "Sunny", "Snowy"]
    ax2.set_yticklabels(labels)

    lns2 = ax2.plot(ts_dt[roi], auto_weather[roi], '.', color=color2, label="Labels")

    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs)

    fig.tight_layout()
    return fig

def plot_production_and_elevation():
    fig, ax = plt.subplots(1)
    ax.xaxis.set_major_formatter(date_format)

    lns1 = ax.plot(ts_dt[roi], production[roi], c="b", label="Production")
    ax2 = ax.twinx()
    lns2 = ax2.plot(ts_dt[roi], elevation[roi], c="orange", label="Solar elevation")
    align_zeros([ax, ax2])
    ax.set_ylabel("Power [kW]")
    ax2.set_ylabel("Elevation angle [$^\circ$]")
    ax.set_xlabel("Time")
    ax.tick_params(axis='x', labelrotation=45)

    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs)

    fig.tight_layout()
    return fig

def plot_cloudiness_variability_features():

    diff = np.diff(production)
    mx = Optimized.window_max(diff, window_size=tm._1H)
    mi = Optimized.window_min(diff, window_size=tm._1H)

    rois = [roi, slice(roi.start, roi.start + tm.DAY)]
    fig, _ax = plt.subplots(2)
    _tax = [a.twinx() for a in _ax]
    lns, labs = None, None
    for ax, r, tax in zip(_ax, rois, _tax):
        ax.xaxis.set_major_formatter(date_format)
        lns1 = ax.plot(ts_dt[r], production[r], c="b", label="Production")
        ax.set_ylabel("Power [kW]")
        ax.set_xlabel("Time")
        ax.tick_params(axis='x', labelrotation=45)

        lns2 = tax.plot(ts_dt[r], diff[r], c="orange", label="Power growth")
        lns3 = tax.plot(ts_dt[r], mx[r], c="red", label="Local max")
        lns4 = tax.plot(ts_dt[r], mi[r], c="red", label="Local min")
        tax.set_ylabel("Power growth [kW']")
        # ax.set_ylabel("Elevation angle [$^\circ$]")
        align_zeros([ax, tax])

        lns = lns1 + lns2 + lns3 + lns4
        labs = [l.get_label() for l in lns]
        tax.legend(lns, labs)

    tax, r  = _tax[1], rois[1] # add more elements to the second plot
    middle = len(ts_dt[r])//2
    marked_gray = slice(middle, middle + tm._1H)
    lns5 = tax.fill_between(ts_dt[r][marked_gray],  mx[r][marked_gray], mi[r][marked_gray], color="gray", alpha=0.6, label="var_index")
    lns = lns + [lns5]
    labs = [l.get_label() for l in lns]
    tax.legend(lns, labs)
    fig.tight_layout()
    return fig

def plot_cloudiness_overall_features():
    rois = [roi, slice(roi.start, roi.start + tm.DAY)]
    fig, _ax = plt.subplots(2)
    production_ma = Optimized.window_moving_avg(production, window_size=tm._3H, roll=True)
    # _tax = [a.twinx() for a in _ax]
    for ax, r in zip(_ax, rois):
        ax.xaxis.set_major_formatter(date_format)
        ax.plot(ts_dt[r], production[r], c="b", label="Production")
        ax.plot(ts_dt[r], production_ma[r], c="orange", label="Averaged production")
        ax.plot(ts_dt[r], expected_from_model[r], c="red", label="Model prediction")
        ax.plot(ts_dt[r], expected_from_model[r] - production_ma[r], c="pink", label="over_index")
        ax.set_ylabel("Power [kW]")
        ax.set_xlabel("Time")
        ax.tick_params(axis='x', labelrotation=45)
        ax.legend()


    # tax, r  = _tax[1], rois[1] # add more elements to the second plot
    # middle = len(ts_dt[r])//2
    # marked_gray = slice(middle, middle + tm._1H)

    # fig.legend()
    fig.tight_layout()
    return fig

def plot_production_and_features():
    fig, ax = plt.subplots(1)
    ax.xaxis.set_major_formatter(date_format)
    lns0 = ax.plot(ts_dt[roi], production[roi], c="b", label="Production")

    lns1 = ax.plot(ts_dt[roi], overall_cloudiness_index[roi], c="g", label="over_index")

    ax2 = ax.twinx()
    lns2 = ax2.plot(ts_dt[roi], Optimized.window_moving_avg(variability_cloudiness_index[roi], tm._1H, roll=True), c="r", label="var_index")

    align_zeros([ax, ax2])

    ax.set_ylabel("Power [kW]")
    ax.set_xlabel("Time")

    lns = lns0 + lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs)
    ax.tick_params(axis='x', labelrotation=45)

    fig.tight_layout()
    return fig

def plot_label_distribution():
    classes, centroids = Optimized.centroids(variability_cloudiness_index, overall_cloudiness_index, weather)
    fig, ax = Utils.plot_distribution(weather, classes, title='Class distribution', xlabel='Labels',
                                      ylabel='Percentage share [%]', percentage=True)
    return fig

def plot_label_distribution_without_night():
    vci, oci, w =variability_cloudiness_index[elevation >= 0], overall_cloudiness_index[elevation >= 0], weather[elevation >= 0]
    classes, centroids = Optimized.centroids(vci, oci, w)
    fig, ax = Utils.plot_distribution(w, classes, title='Class distribution w/o night', xlabel='Labels',
                                      ylabel='Percentage share [%]', percentage=True)
    return fig

def plot_label_plane():
    classes, centroids = Optimized.centroids(variability_cloudiness_index, overall_cloudiness_index, weather)
    fig, ax = Plotter.plot_scatter(variability_cloudiness_index, overall_cloudiness_index, weather, centroids=centroids,
                            classes_txt=classes_txt,
                            x_label="variability_cloudiness_index", y_label="overall_cloudiness_index")

    fig.tight_layout()
    return fig

def plot_each_label_plane_separately():
    classes, centroids = Optimized.centroids(variability_cloudiness_index, overall_cloudiness_index, weather)
    fig, ax = plt.subplots(3, 3)
    for i,c in enumerate(classes[1:]): # skip class: 0
        _ax = ax[int(i/3), i%3]
        Plotter.plot_scatter(variability_cloudiness_index, overall_cloudiness_index, weather, centroids=centroids,
                             filter_class=[c], ax=_ax, fig=fig, classes_txt=classes_txt, legend=False,
                             x_label="var_index", y_label="over_index")
    fig.tight_layout()
    return fig

if __name__ == "__main__":

    print("start:", ts_dt[0], "end:", ts_dt[-1], "length:", len(ts_dt), "interval:",  ts_dt[1]-ts_dt[0])

    funcs = [
        plot_production_and_weather,
        plot_production_and_auto_weather,
        plot_production_and_elevation,
        plot_cloudiness_variability_features,
        plot_cloudiness_overall_features,
        plot_label_distribution,
        plot_label_distribution_without_night,
        plot_label_plane,
        plot_each_label_plane_separately,
        plot_production_and_features
    ]

    show_or_save = True
    if show_or_save:
        DIR = "cm"
        if not os.path.exists(DIR):
            os.mkdir(DIR)
        pp = PdfPages(os.path.join(DIR, 'paper_images.pdf'))
        for f in funcs:
            pp.savefig(f())
        pp.close()
    else:
        for f in funcs:
            f().show()
        plt.show()



