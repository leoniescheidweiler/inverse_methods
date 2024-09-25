import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def open_file(filename):
    data = xr.open_dataset(filename)
    date = data["decimal date"].values
    co2 = data["XCO2 [ppm]"].values
    co2_deseasonalized = data["deseasonalized XCO2 [ppm]"].values

    return date, co2, co2_deseasonalized


def plot_data(date, co2, co2_deseasonalized):
    plt.title("Atmospheric CO2 Mixing Ratio for Mauna Loa")
    date = decimal_date_to_datetime_date(date)
    plt.plot(date, co2, label="measured", color="black")
    plt.plot(date, co2_deseasonalized, label="deseasonalized", color="red")
    plt.xlabel("year")
    plt.ylabel("XCO2 / ppm")
    plt.legend()
    plt.show()


def decimal_date_to_datetime_date(decimal_date):
    datetime_date = [datetime(int(d), 1, 1) + timedelta(days=(d%1)*365)
        for d in decimal_date]
    return datetime_date


def fwd_trend(date, acoeff):
    """
        This function provides a polynomial in the variable date
        using the coefficients acoeff.
    """
    # a0 t^0 + a1 t^1 + a2 t^2
    # 0 a0 t^(0-1) + 1 a1 t^(1-1) + 

    fwd = np.zeros(shape=len(date))
    for i, ai in enumerate(acoeff):
        fwd += ai * date**i

    dfwd_dacoeff = np.zeros(shape=(len(date), len(acoeff)))
    for j in range(len(acoeff)):
        dfwd_dacoeff[:, j] = date**j

    return fwd, dfwd_dacoeff


def fwd_season(date, acoeff, m):
    """
        blah blah
    """
    # a cos(2 pi (t - t0) + m/12)
    fwd = np.zeros(shape=len(date))
    for i in range(len(date)):
        # fwd[i] = acoeff[0] * np.cos(2*np.pi*(date[i] - date[0]) + (date[i]%1+m[0]/12)*2*np.pi)
        fwd[i] = acoeff[0] * np.cos(2 * np.pi * (date[i] + m[0]/12))

    dfwd_dacoeff = np.zeros(shape=(len(date), len(acoeff)))
    for i in range(len(date)):
        # dfwd_dacoeff[i, :] = np.cos(2*np.pi*(date[i] - date[0]) + (date[i]%1+m[0]/12)*2*np.pi)
        dfwd_dacoeff[i, :] = np.cos(2 * np.pi * (date[i] + m[0]/12))

    return fwd, dfwd_dacoeff


def main():
    date_abs, co2, co2_deseasonalized = open_file("co2_mm_mlo.nc")
    # plot_data(date_abs, co2, co2_deseasonalized)

    date = date_abs - date_abs[0]

    max_deg = 3
    a_n = max_deg + 1

    fwd, dfwd_dacoeff = fwd_trend(date, np.zeros(a_n))
    y = co2
    y_deseasonalized = co2_deseasonalized
    K = dfwd_dacoeff
    G = np.linalg.inv(K.T @ K) @ K.T
    x = G @ y
    y_est = fwd_trend(date, x)[0]

    plt.title("concentration")
    plt.plot(date_abs, y, label="measured")
    plt.plot(date_abs, y_est, label=f"modelled")
    plt.legend()
    plt.show()

    plt.title("comparison")
    plt.plot(date_abs, y_est - y_deseasonalized, label="modelled - measured")
    plt.legend()
    plt.show()

    plt.title("monthly trend")
    plt.plot(date_abs[1:], np.diff(y_est)/np.diff(date))
    plt.show()

    plt.title("residual")
    plt.plot(date_abs, y - y_deseasonalized, label="measured")
    plt.plot(date_abs, y - y_est, label="modelled")
    plt.legend()
    plt.show()

    plt.title("seasonality")
    res = y - y_est
    plt.plot(date, res, label="residual")
    for m in np.linspace(0, 12, 100):
        fwd, dfwd_dacoeff = fwd_season(date, [0], [m])
        K = dfwd_dacoeff
        G = np.linalg.inv(K.T @ K) @ K.T
        x = G @ res
        print(x)
        y_est = fwd_season(date, x, [m])[0]

        plt.plot(date, y_est, label="modelled residual")
    plt.legend()
    plt.show()


if __name__=="__main__":
    main()
