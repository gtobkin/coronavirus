import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # for legends
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import datetime
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def display_evdefender_plots():
    # Load and select data from CSV
    data = pd.read_csv("data/coronavirus_data.csv")
    
    window = data[["date", "cases_china_ex", "deaths_china_ex"]][7:14]
    
    y_cases = window["cases_china_ex"].values.reshape(-1,1)
    y_deaths = window["deaths_china_ex"].values.reshape(-1,1)
    x = np.array(range(len(y_cases))).reshape(-1,1)
    
    # Fit quadratic model to cases data
    poly_cases = PolynomialFeatures(degree = 2)
    x_poly_cases = poly_cases.fit_transform(x)
    poly_cases.fit(x_poly_cases, y_cases)
    lin2_reg_cases = LinearRegression().fit(x_poly_cases, y_cases)
    
    # Fit quadratic model to deaths data
    poly_deaths = PolynomialFeatures(degree = 2)
    x_poly_deaths = poly_deaths.fit_transform(x)
    poly_deaths.fit(x_poly_deaths, y_deaths)
    lin2_reg_deaths = LinearRegression().fit(x_poly_deaths, y_deaths)
    
    # Calculate coeffs and R2 for both
    r2_cases = lin2_reg_cases.score(poly_cases.fit_transform(x), y_cases)
    coeffs_cases = lin2_reg_cases.coef_
    r2_deaths = lin2_reg_deaths.score(poly_deaths.fit_transform(x), y_deaths)
    coeffs_deaths = lin2_reg_deaths.coef_
    
    # Plot cases and death fits
    fig, axs = plt.subplots(2, sharex=True)  # two vertically-stacked subplots
    fig.suptitle("Quadratic fits and R2 scores")
    
    axs[0].scatter(x, y_cases, color = "blue")
    axs[0].plot(x, lin2_reg_cases.predict(poly_cases.fit_transform(x)), color = 'red')
    axs[0].set(ylabel = "Confirmed China Cases ex-HK, Macau, Taipei")
    label = "{}d^2 + {}d + {}".format(round(coeffs_cases[0][0], 3), round(coeffs_cases[0][1], 3), round(coeffs_cases[0][2], 3))
    label += "; R2 = {}".format(round(r2_cases, 5))
    patch_cases = mpatches.Patch(color='red', label=label)
    axs[0].legend(handles=[patch_cases])
    axs[0].grid(axis='y')
    
    axs[1].scatter(x, y_deaths, color = "blue")
    axs[1].plot(x, lin2_reg_deaths.predict(poly_deaths.fit_transform(x)), color = 'red')
    axs[1].set(xlabel = "Days")
    axs[1].set(ylabel = "Confirmed China Deaths ex-HK, Macau, Taipei")
    label = "{}d^2 + {}d + {}".format(round(coeffs_deaths[0][0], 3), round(coeffs_deaths[0][1], 3), round(coeffs_deaths[0][2], 3))
    label += "; R2 = {}".format(round(r2_deaths, 5))
    patch_deaths = mpatches.Patch(color='red', label=label)
    axs[1].legend(handles=[patch_deaths])
    axs[1].grid(axis='y')
    
    plt.show()


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


def display_ebola_quadratic_samples():
    # Load data from CSV
    data = pd.read_csv("data/ebola_2014_data.csv")

    # Convert dates from strings to datetimes
    dates = data["date"][::-1].values
    datetimes = [datetime.datetime.strptime(datestr, '%d %b %Y') for datestr in dates]  # date of month, month abbrev., year
    days_elapsed = [(date - min(datetimes)).days for date in datetimes]

    # set per-kind, per-loc caps on x to limit ourselves to exponential growth phase; determined by inspection
    # (2-3 readings before report-by-report change in cases/deaths peaked)
    caps = {
            "cases": { "total": 440, "guinea": 410, "liberia": 435, "sierraleone": 425, },
            "deaths": { "total": 435, "guinea": 400, "liberia": 465, "sierraleone": 420, },
        }

    # for each kind/loc combo, for each 6-long range of in-window reports, compute an R2 for a quadratic fit
    r2s = []
    for kind in ["cases", "deaths"]:
        for loc in ["total", "guinea", "liberia", "sierraleone"]:
            y_full = data["{}_{}".format(kind,loc)][::-1].values
            for i_min in [x for x in range(len(days_elapsed) - 5) if days_elapsed[x] <= caps[kind][loc] and y_full[x] > 0]:
                x = np.array(days_elapsed[i_min:i_min+6]).reshape(-1,1)
                y = y_full[i_min:i_min+6]
                poly = PolynomialFeatures(degree = 2)
                x_poly = poly.fit_transform(x)
                poly.fit(x_poly, y)
                lin2_reg = LinearRegression().fit(x_poly, y)

                r2 = lin2_reg.score(poly.fit_transform(x), y)
                r2s.append(r2)
                # coeffs = lin2_reg.coef_
    # logbins = np.geomspace(min(r2s), 1.0, 20)
    # logbins = np.logspace(np.log10(0.99), np.log10(1.0), 50)
    # logbins = np.logspace(-0.01, 0, base = 10000, num = 50)
    plt.hist(r2s, bins=20, range=[0.999, 1.000])
    plt.xscale('log')
    plt.show()
                


if __name__ == "__main__":
    display_evdefender_plots()
    display_ebola_plots()
    display_ebola_quadratic_samples()

































