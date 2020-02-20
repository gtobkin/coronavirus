import numpy as np
from matplotlib.ticker import NullFormatter, FormatStrFormatter, StrMethodFormatter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # for legends
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import datetime
from scipy.optimize import curve_fit

from sklearn.metrics import r2_score
from math import floor

# We'll use this later - x_max (in days elapsed) before ebola in a particular region switched from exponential to sigmoid
# (determined by inspection, 2-3 readings before daily new cases/deaths peaked)
X_CAPS = {
        "cases": { "total": 440, "guinea": 410, "liberia": 435, "sierraleone": 425, },
        "deaths": { "total": 435, "guinea": 400, "liberia": 465, "sierraleone": 420, },
    }


def display_evdefender_plots():
    # Load and select data from CSV
    data = pd.read_csv("data/coronavirus_data.csv")

    # select data for [Jan 27, Feb 2], matching @evdefender's second plot
    window = data[["date", "cases_china_ex", "deaths_china_ex"]][7:14]

    # Fit quadratic and exp models to cases and deaths timeseries; find coeffs and R2s
    x = list(range(len(window)))  # 7 datapoints
    y, y_predicted, popt, r2, label = ({}, {}, {}, {}, {})
    def fquadratic(x, a, b, c):
        return a*x**2 + b*x + c
    def fexp(x, a, b):
        return a*(1+b)**x
    for fittype in ["quad", "exp"]:
        callback = fquadratic if fittype == "quad" else fexp
        y_predicted[fittype], popt[fittype], r2[fittype], label[fittype] = ({}, {}, {}, {})
        for kind in ["cases", "deaths"]:
            y[kind] = window["{}_china_ex".format(kind)].values
            popt[fittype][kind], _ = curve_fit(callback, x, y[kind])
            if fittype == "quad":
                y_predicted[fittype][kind] = [callback(x_i, popt[fittype][kind][0], popt[fittype][kind][1], popt[fittype][kind][2]) for x_i in x]
                label[fittype][kind] = "{}d^2 + {}d + {}".format(round(popt[fittype][kind][0], 2), round(popt[fittype][kind][1], 1), round(popt[fittype][kind][2], 1))
            elif fittype == "exp":
                y_predicted[fittype][kind] = [callback(x_i, popt[fittype][kind][0], popt[fittype][kind][1]) for x_i in x]
                label[fittype][kind] = "{}*(1+{})^d".format(round(popt[fittype][kind][0], 2), round(popt[fittype][kind][1], 4))
            r2[fittype][kind] = r2_score(y[kind], y_predicted[fittype][kind])
            label[fittype][kind] += "; R2 = {}".format(round(r2[fittype][kind], 5))

    # Plot fits for cases and deaths timeseries
    fig, axs = plt.subplots(2, sharex=True)  # two vertically-stacked subplots
    fig.suptitle("Quadratic fits and R2 scores")
    for i, kind in enumerate(["cases", "deaths"]):
        axs[i].scatter(x, y[kind], color='blue')
        axs[i].set(ylabel="Confirmed China {}\nex-HK, Macau, Taipei".format(kind.capitalize()))
        axs[i].grid(axis='y')
        patches = []
        for fittype in ["quad", "exp"]:
            fittype_color = 'red' if fittype == "quad" else 'green'
            axs[i].plot(x, y_predicted[fittype][kind], color=fittype_color)
            patches.append(mpatches.Patch(color=fittype_color, label=label[fittype][kind]))
        axs[i].legend(handles=patches)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(["Jan {}".format(k) for k in range(27, 32)] + ["Feb {}".format(k) for k in range(1, 3)])

    plt.show()
    return r2["quad"]


def display_ebola_plots():
    # Load data from CSV
    data = pd.read_csv("data/ebola_2014_data.csv")
    
    # Convert dates from strings to datetimes
    dates = data["date"].values
    datetimes = [datetime.datetime.strptime(datestr, '%d %b %Y') for datestr in dates]  # date of month, month abbrev., year
    days_elapsed = [(date - min(datetimes)).days for date in datetimes]

    # queue up plots - we'll add scatter plots and fits as we go
    fig, axs = plt.subplots(2, sharex=True)  # two vertically-stacked subplots
    fig.suptitle("Ebola cases and death fits")

    x = days_elapsed

    # compute sigmoid fits for cases and deaths, for all four locs
    def fsigmoid(x, a, b, c):
        return c / (1 + np.exp(-a*(x-b)))
    results = {"popt": {}, "pcov": {}, "labels": {}}
    colors = {
            "total": ["blue", "blue"],
            "guinea": ["red", "red"],
            "liberia": ["green", "green"],
            "sierraleone": ["black", "black"],
            }
    p0 = [[2.2e-02, 2.3e+02, 2.0e+04],
          [1.2e-02, 2.0e+02, 1.1e+04]]  # from observation from previous runs with no p0
    for kind in ["cases", "deaths"]:
        isdeaths = 1 if kind == "deaths" else 0
        results["popt"][kind] = {}
        results["pcov"][kind] = {}
        results["labels"][kind] = {}
        patches = []
        for loc in colors.keys():
            y = data["{}_{}".format(kind, loc)].values
            popt, pcov = curve_fit(fsigmoid, days_elapsed, y, p0 = p0[isdeaths])
            results["popt"][kind][loc] = popt
            results["pcov"][kind][loc] = pcov

            axs[isdeaths].scatter(x, y, color = colors[loc][0])
            y_predicted = [fsigmoid(x_i, popt[0], popt[1], popt[2]) for x_i in x]
            axs[isdeaths].plot(x, y_predicted, color = colors[loc][1])

            label = "{}: {}/(1 + exp({}*(x-{})))".format(loc, round(popt[2], 3), round(popt[0], 3), round(popt[1], 3))
            label += "; R2 = {}".format(round(r2_score(y, y_predicted), 4))
            results["labels"][kind][loc] = label
            patches.append(mpatches.Patch(color=colors[loc][1], label=label))
        axs[isdeaths].legend(handles=patches)

    # Plot cases and death fits
    plt.show()
    return


def display_ebola_quadratic_samples(r2s_evdef):
    # Load data from CSV
    data = pd.read_csv("data/ebola_2014_data.csv")

    # Convert dates from strings to datetimes
    dates = data["date"][::-1].values
    datetimes = [datetime.datetime.strptime(datestr, '%d %b %Y') for datestr in dates]  # date of month, month abbrev., year
    days_elapsed = [(date - min(datetimes)).days for date in datetimes]

    # TODO: this should be 7-long windows
    # for each kind/loc combo, for each 6-long range of in-window reports, compute an R2 for a quadratic fit
    r2s = []
    for kind in ["cases", "deaths"]:
        for loc in ["total", "guinea", "liberia", "sierraleone"]:
            y_full = data["{}_{}".format(kind,loc)][::-1].values
            for i_min in [x for x in range(len(days_elapsed) - 5) if days_elapsed[x] <= X_CAPS[kind][loc] and y_full[x] > 0]:
                x = np.array(days_elapsed[i_min:i_min+6]).reshape(-1,1)
                y = y_full[i_min:i_min+6]
                poly = PolynomialFeatures(degree = 2)
                x_poly = poly.fit_transform(x)
                poly.fit(x_poly, y)
                lin2_reg = LinearRegression().fit(x_poly, y)

                r2 = lin2_reg.score(poly.fit_transform(x), y)
                r2s.append(r2)
    x = np.array(r2s)
    np.random.seed(0)  # make the jitter deterministic
    y = np.random.normal(0, 1, len(x))
    fig, ax = plt.subplots()
    ax.set_xscale('logit')
    ax.set_yticks([])
    ax.scatter(x, y, alpha=0.4)
    ax.set_ylim([-10, 10])
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.4g'))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([floor(min(r2s)*10.0) / 10, 0.9, 0.99, 0.999, 0.9999])
    # add the two points for @evdefender's cases and deaths R2 scores
    x = np.array([r2s_evdef["cases"], r2s_evdef["deaths"]])
    y = np.zeros_like(x)
    ax.scatter(x, y, color='red', marker='^')
    ax.annotate("Feb 2 China\ncases R2", (x[0], y[0]), textcoords="offset pixels", xytext=(0,-40), ha="center", color="red", fontsize=9, weight="bold")
    ax.annotate("Feb 2 China\ndeaths R2", (x[1], y[1]), textcoords="offset pixels", xytext=(0,-40), ha="center", color="red", fontsize=9, weight="bold")
    plt.show()
    return


def display_ebola_quadratic_split_samples(r2s_evdef):
    # Load data from CSV
    data = pd.read_csv("data/ebola_2014_data.csv")

    # Convert dates from strings to datetimes
    dates = data["date"][::-1].values
    datetimes = [datetime.datetime.strptime(datestr, '%d %b %Y') for datestr in dates]  # date of month, month abbrev., year
    days_elapsed = [(date - min(datetimes)).days for date in datetimes]

    for kind in ["cases", "deaths"]:
        for loc in ["total", "guinea", "liberia", "sierraleone"]:
            y_full = data["{}_{}".format(kind,loc)][::-1].values
            # TODO: this should be 7-long windows
            for i_min in [x for x in range(len(days_elapsed) - 5) if days_elapsed[x] <= X_CAPS[kind][loc] and y_full[x] > 0]:
                # color it based on quality (e.g. >=1, >=5, >= 20 entries)
                # repeat analysis above but in a 4x2 (regions x cases/deaths) plot grid, and color-code scatterpoints
        



if __name__ == "__main__":
    r2s_evdef = display_evdefender_plots()
    display_ebola_plots()
    display_ebola_quadratic_samples(r2s_evdef)

































