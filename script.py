import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # for legends
import pandas as pd

# Load and select data from CSV
data = pd.read_csv("data/coronavirus_data.csv")

window = data[["date", "cases_china_ex", "deaths_china_ex"]][7:14]

y_cases = window["cases_china_ex"].values.reshape(-1,1)
y_deaths = window["deaths_china_ex"].values.reshape(-1,1)
x = np.array(range(len(y_cases))).reshape(-1,1)

from sklearn.preprocessing import PolynomialFeatures

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

