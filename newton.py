import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import polyfit
from sympy import symbols, solve

# Constants
G = 6.67430 * 1e-11  # in CSV units
solar_mass = 1.9891 * 1e30  # solarmass
G_solar = G * solar_mass  # G in scaled units
radius_earth = 6.38 * 10 ** 6

def read_dataset(file_path):
    dataset = pd.read_csv(file_path)
    return dataset

def get_dw_data(file_path):
    dataset = read_dataset(file_path)
    features = ["wdid", "logg", "mass"]

    loggs = dataset[features[1]]
    masses = dataset[features[2]]
    return np.array(loggs),np.array(masses)

def plot_M_vs_R(logs,masses ):
    g = 10 ** logs * 1e-2

    # Calculate radius
    radius = np.sqrt(G  * masses * solar_mass / g) / radius_earth  # stores r values

    # Plot
    plt.scatter(radius, masses)  # Convert mass back to solar masses for plotting
    plt.title('White Dwarf Mass vs Radius')
    plt.xlabel('Radius (in average Earth radius)')
    plt.ylabel('Mass (in Solar mass)')
    plt.show()
    return radius,masses

def get_small_dws(radius,masses):
    mass_threshold = 0.5
    mask = masses < mass_threshold
    filtered_radius = radius[mask]
    filtered_masses = masses[mask]
    return filtered_radius,filtered_masses

def calc_n(radius, mass):
    ln_radius = np.log(radius)
    ln_mass = np.log(mass)


    gamma, ln_B = polyfit(ln_radius, ln_mass, 1)
    B = np.exp(ln_B)
    print(f"ln_b {ln_B}, B {B}")
    n_star = (3 - gamma) / (1 - gamma)
    q = symbols('q')
    equation = q / (5 - q)

    # Solve for q
    q_value = solve(equation - n_star, q)
    print(f"n^*={n_star} , q= {q_value}")
    return n_star


def solve_lane_emden(n_value):
    step_size = 0.01  # Increment for each step
    max_iterations = 100000  # Maximum number of iterations

    xi_value = 0.0001  # Initial xi value, close to zero
    theta_value = 1.0  # Initial theta value
    derivative_f = 0.0  # Initial derivative value
    solutions_theta = []
    solutions_xi = []

    for i in range(max_iterations):
        derivative_f -= xi_value ** 2 * theta_value ** n_value * step_size
        theta_value += derivative_f / xi_value ** 2 * step_size
        xi_value += step_size
        solutions_theta.append(theta_value)
        solutions_xi.append(xi_value)

        # Check if theta crosses zero, indicating a solution
        if solutions_theta[i] * solutions_theta[i - 1] < 0:
            break

    # Compute the derivative of theta at xi
    derivative_theta_at_xi = (solutions_theta[i] - solutions_theta[i - 1]) / step_size

    return xi_value, derivative_theta_at_xi

def plot_M_vs_d(filtered_masses,central_density):
    # Plotting the central density versus mass for white dwarfs
    plt.plot(filtered_masses, central_density, 'bs', markersize=1)
    plt.title("Central Density vs Mass for White Dwarfs")
    plt.xlabel("Mass (in Solar Mass units)")
    plt.ylabel("Central Density (rho_c)")
    plt.show()

def main():
    filename = "white_dwarf_data.csv"
    logs, masses = get_dw_data(filename)
    radius, masses = plot_M_vs_R(logs, masses)
    filtered_radius, filtered_masses = get_small_dws(radius, masses)
    n = calc_n(filtered_radius, filtered_masses)
    xi_for_n, derivative_theta_for_n = solve_lane_emden(n)
    print(f"xi_for_n: {xi_for_n} , derivative_theta_for_n {derivative_theta_for_n} ")
    central_density = -(filtered_masses * xi_for_n) / (4 * np.pi * filtered_radius ** 3 * derivative_theta_for_n)
    print(f"central_density: {central_density}")
    plot_M_vs_d(filtered_masses,central_density)





