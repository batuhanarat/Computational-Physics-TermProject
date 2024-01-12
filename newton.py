'''
(b) You can find observational data from many low-temperature WDs in the file white dwarf data.csv.
4 Two quantities are listed for each WD: mass (in solar masses M⊙), and base-10 logarithm of the surface gravity
in CGS units (log(g)). The latter can be easily converted to radius using basic Newtonian gravity.
Write a function to read this data, and show all the points in an M vs R plot using solar masses and
average Earth radii as units. Note that this is a .csv file, so reading it with
    Python might be a bit different from reading a plain ASCII file which was the case for
        the Hubble data in the problem sets.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def read_dataset(file_path):
    dataset = pd.read_csv(file_path)
    return dataset

def get_dw_data(file_path):
    dataset = read_dataset(file_path)
    features = ["wdid", "logg", "mass"]

    loggs = dataset[features[1]]
    masses = dataset[features[2]]
    return np.array(loggs),np.array(masses)

def plot_M_vs_R():
    filename = "white_dwarf_data.csv"
    logs,masses = get_dw_data(filename)
    g = 10 ** logs * 1e-2

    ##Constants for conversion
    G = 6.67430 * 1e-11  # in CSV units
    solar_mass = 1.9891 * 1e30  # solarmass
    G_solar = G * solar_mass  # G in scaled units
    radius_earth = 6.38 * 10 ** 6
    n = len(masses)


    # Calculate radius
    radius = np.sqrt(G * masses / g) / radius_earth  # Convert radius to Earth radii

    # Plot
    plt.scatter(radius, masses / solar_mass)  # Convert mass back to solar masses for plotting
    plt.title('White Dwarf Mass vs Radius')
    plt.xlabel('Radius (in average Earth radius)')
    plt.ylabel('Mass (in Solar mass)')
    plt.show()









