import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

data = xr.open_dataset("bivariate_data3.nc")
x1 = data.data.isel(data_dimension=0).values
x2 = data.data.isel(data_dimension=1).values
y = x1 + x2

print(f"{np.cov(x1, x1)=}")
print(f"{np.cov(x1, x2)=}")
print(f"{np.cov(y, y)=}")
print(f"{np.var(y)=}")
