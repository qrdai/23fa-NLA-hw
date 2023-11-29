
import numpy as np
import matplotlib.pyplot as plt

def plot_data_and_polynomial(data_file, coefficients):
    '''
    Example usage:
       coefficients = [a_0, a_1, ..., a_n] # Coefficients of the polynomial
       plot_data_and_polynomial('data.dat', coefficients)
    '''
    # Reading data
    data = np.loadtxt(data_file, skiprows=1)
    x, y = data[:, 0], data[:, 1]

    # Generating polynomial points
    X = np.linspace(min(x), max(x), 500)
    Y = sum(coefficients[i] * X**i for i in range(len(coefficients)))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='red', label='Original Data Points')
    plt.plot(X, Y, color='blue', label='Polynomial Fit')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Polynomial Fit to Data Points')
    plt.legend()
    plt.show()


def main():
    data_file = 'data.dat'
    coefficients_list = (
    (6.420000,),
    (3.668107, 0.993463,),
    (3.086640, 1.778776, -0.156306,),
    (3.179311, 1.520536, -0.031315, -0.015928),
    (2.931133, 2.527478, -0.925597, 0.259223, -0.027234),
    (3.302285, 0.548540, 1.698495, -1.062627, 0.254667, -0.021502),
    (5.124115, -11.206113, 22.871479, -16.879147, 5.881694, -0.969259, 0.060804),
    (6.511887, -23.089442, 54.695144, -53.511730, 26.741864, -7.148960, 0.973092, -0.052993),
    (13.247400, -85.406060, 241.966400, -306.315748, 205.678504, -78.446860, 17.083212, -1.979595, 0.094700),
    (280.120205, -2632.708972, 8366.981270, -12398.140518, 10053.695004, -4830.562494, 1414.511540, -248.059639, 23.953893, -0.979794),
    )
    plot_data_and_polynomial('data.dat', coefficients_list[9])


if __name__ == "__main__":
    main()