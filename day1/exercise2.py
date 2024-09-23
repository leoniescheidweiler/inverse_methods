import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import sys


def pdf(xx, yy, cov, E1, E2, D):
    Nx = xx.shape[0]
    Ny = xx.shape[1]
    res = np.empty(shape=(Nx, Ny))
    for i in range(Nx):
        for j in range(Ny):
            x1 = xx[i, j]
            x2 = yy[i, j]
            tmp = np.array((x1-E1, x2-E2))
            norm = 1/((2*np.pi)**D) * np.sqrt(np.linalg.det(cov))
            num = np.exp(-0.5 * tmp @ np.linalg.inv(cov) @ tmp)
            res[i, j] = norm * num
    return res


data1 = xr.open_dataset("bivariate_data1.nc", engine="netcdf4")
data2 = xr.open_dataset("bivariate_data2.nc", engine="netcdf4")
data3 = xr.open_dataset("bivariate_data3.nc", engine="netcdf4")

# print(f"{data1=}\n")
# print(f"{data2=}\n")
# print(f"{data3=}\n")

fig, ax = plt.subplots(2, 3, figsize=(15, 5))
for i, data in enumerate([data1, data2, data3]):
    x1 = data.isel(data_dimension=0).data
    x2 = data.isel(data_dimension=1).data

    N = data.sizes["number_samples"]
    D = data.sizes["data_dimension"]

    # mean and standard deviation
    # x1_mean = 1/N * np.sum(x1)
    # x1_std = np.sqrt(1/N * np.sum((x1 - x1_mean)**2))

    # covariance and correlation NUMBERS
    # cov = 1/N * (np.sum((x1-x1.mean())*(x2-x2.mean())))
    # cov = cov.values
    # corr = cov / (x1.std()*x2.std())
    # corr = corr.values

    # covariance and correlation MATRICES
    # xs = [x1, x2]
    # cov = np.empty(shape=(D, D))
    # for i in range(D):
    #     for j in range(D):
    #         cov[i, j] = 1/N * np.sum((xs[i] - xs[i].mean()) * (xs[j] - xs[j].mean()))

    x = np.linspace(x1.min().values, x1.max().values, 100)
    y = np.linspace(x2.min().values, x2.max().values, 100)
    xx, yy = np.meshgrid(x, y)
    res = pdf(xx, yy, np.cov(x1, x2), x1.mean(), x2.mean(), D)

    ax[0, i].set_title(f"data {i+1}")
    ax[0, i].hexbin(x1, x2, cmap="inferno_r")
    ax[0, i].set_xlim(-2, 12)
    ax[0, i].set_ylim(1, 8.2)

    ax[1, i].contour(xx, yy, res, cmap="inferno_r")
plt.show()
