import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("data/coronavirus_data.csv")

window = data[["date", "cases_china_ex", "deaths_china_ex"]][7:14]

y_cases = window["cases_china_ex"].values.reshape(-1,1)
y_deaths = window["deaths_china_ex"].values.reshape(-1,1)
x = np.array(range(len(y_cases))).reshape(-1,1)

from sklearn.preprocessing import PolynomialFeatures

poly_cases = PolynomialFeatures(degree = 2)
x_poly_cases = poly_cases.fit_transform(x)
poly_cases.fit(x_poly_cases, y_cases)
lin2_cases = LinearRegression()
lin2_cases.fit(x_poly_cases, y_cases)

poly_deaths = PolynomialFeatures(degree = 2)
x_poly_deaths = poly_deaths.fit_transform(x)
poly_deaths.fit(x_poly_deaths, y_deaths)
lin2_deaths = LinearRegression()
lin2_deaths.fit(x_poly_deaths, y_deaths)

plt.scatter(x, y_cases, color = 'blue')
plt.plot(x, lin2_cases.predict(poly_cases.fit_transform(x)), color = 'red')
plt.title('Polynomial Regression')
plt.xlabel('Days')
plt.ylabel('Confirmed China Cases ex-HK, Macau, Taipei')

plt.show()

from sklearn.metrics import r2_score


# y_cases = window["cases_china_ex"].values
# y_deaths = window["deaths_china_ex"].values
# x = range(len(y_cases))
# 
# z = np.polyfit(x, y_cases, 2)

