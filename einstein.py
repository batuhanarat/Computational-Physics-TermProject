import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sympy import symbols, integrate, log, Function



# Constants
k = 100  # Polytrope constant
length_unit = 1.477  # km
solar_mass = 1.989 * 1e30  # kg


def tov_equations(r, y):
    m, v, p, m_p = y
    if r == 0:
        return [0, 0, 0, 0]
    else:
        rho = np.sqrt(p / k)
        dfdr_m = 4 * np.pi * r ** 2 * rho
        dfdr_v = 2 * (m + 4 * np.pi * r ** 3 * p) / (r * (r - 2 * m))
        dfdr_p = -0.5 * (p + rho) * dfdr_v
        dfdr_m_p = 4 * np.pi * np.sqrt(1 / (1 - 2 * (m / r))) * r ** 2 * rho
        return [dfdr_m, dfdr_v, dfdr_p, dfdr_m_p]


def solve_tov(p0_initial, r0_initial, tol=1e-8):
    p0, r0 = p0_initial, r0_initial
    f0 = [0, 0, p0, 0]  # Initial conditions

    while p0 > tol:
        sol = solve_ivp(tov_equations, [0, r0], f0, method='RK45')
        p0 = sol.y[2, -1]
        r0 = r0 * (1.002 if p0 > 0 else 0.998)

    M = sol.y[0, -1]
    M_p = sol.y[3, -1]
    R = sol.t[-1] * length_unit
    rho_c = np.sqrt(sol.y[2, 0] / k) * (solar_mass / (length_unit * 10 ** 3) ** 3)
    delta = -(M - M_p) / M

    return M, R, rho_c, delta


def plot_results(radius, mass, title, xlabel, ylabel):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(radius, mass)
    plt.show()

def part_e():
    M, r, R = symbols('M r R')
    v = Function('v')(r)
    v_prime = 2 * M / (r * (r - 2 * M))
    integral_result = integrate(v_prime, (r, R, r))
    integral_result += v.subs(r, R)  # Add v(R) to the result of the integral
    # Simplfy the expression
    simplified_result = integral_result.simplify()

    print(simplified_result)

def main():
    num_iter = 50
    M_vec, R_vec, rho_c_vec, E_fb_vec = [], [], [], []

    for i in range(num_iter):
        p0_initial = 1e-4 + i * 0.0001
        r0_initial = 10
        M, R, rho_c, delta = solve_tov(p0_initial, r0_initial)

        M_vec.append(M)
        R_vec.append(R)
        rho_c_vec.append(rho_c)
        E_fb_vec.append(delta)

    plot_results(R_vec, M_vec, 'Neutron Star M-R Dependence solution of TOV', 'Radius in km', 'Mass in solar mass')
    plot_results(R_vec, E_fb_vec, 'Neutron Star fractional binding energy vs. Radius', 'Radius in km',
                 'Fractional Binding Energy')
    sorted_indices = np.argsort(rho_c_vec)
    M_vec = np.array(M_vec)[sorted_indices]
    rho_c_vec = np.array(rho_c_vec)[sorted_indices]
    # Find the stable region
    dM = np.diff(M_vec)
    stable_region_index = np.where(dM < 1e-8)[0][0]
    print(f"Stable region mass at index {stable_region_index}: {M_vec[stable_region_index]} solar masses")
    # Dividing solutions into stable and unstable parts
    M_vec_stable = M_vec[stable_region_index:]
    rho_c_vec_stable = rho_c_vec[stable_region_index:]
    M_vec_unstable = M_vec[:stable_region_index + 1]
    rho_c_vec_unstable = rho_c_vec[:stable_region_index + 1]
    max_mass_index = np.argmax(M_vec)
    print(f"Maximum mass point corresponds to a mass of approximately {M_vec[max_mass_index]:.4f} solar masses")

    # Dividing solutions into stable and unstable parts
    M_vec_stable = M_vec[:max_mass_index + 1]
    rho_c_vec_stable = rho_c_vec[:max_mass_index + 1]
    M_vec_unstable = M_vec[max_mass_index:]
    rho_c_vec_unstable = rho_c_vec[max_mass_index:]

    plt.figure()
    plt.title('Neutron Star Mass vs. Central Density')
    plt.xlabel('Central Density (kg/m^3)')
    plt.ylabel('Mass in solar mass')
    plt.plot(rho_c_vec_stable, M_vec_stable, label='stable')
    plt.plot(rho_c_vec_unstable, M_vec_unstable, '--', label='unstable')
    plt.legend()
    plt.show()
    part_e()




