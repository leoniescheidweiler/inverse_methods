import xarray as xr
import numpy as np
import matplotlib.pyplot as plt


def main():
    data = xr.open_dataset("hybrid_reference_spectrum_1nm_resolution_c2021-03-04_with_unc.nc")
    print(data)

    wavelength = data["Vacuum Wavelength"]
    irradiance = data["SSI"]

    plt.plot(wavelength, irradiance)
    plt.xlabel("wavelength / nm")
    plt.ylabel("solar irradiance / W m-2 nm-1")
    plt.show()


def watt_to_photon_rate(wavelength, signal):
    # W -> photons s-1

    # wavelength provided in nm, convert to m
    wavelength = wavelength * 1e-9

    planck_constant = 6.6260715e-34  # J s
    light_speed = 299792458  # m s-1

    # energy of a photon given by E = h freq
    # frequency given by freq = c / lambda (possibly)
    # energy of n photons given by E = n h c lambda-1
    # power is energy per time P = n h c lambda-1 t-1
    # therefore number of photons per time is given by
    # n t-1 = P h-1 c-1 lambda
    signal = signal * wavelength / planck_constant / light_speed

    return signal


if __name__ == "__main__":
    main()
