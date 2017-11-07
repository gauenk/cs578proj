import matplotlib.pyplot as mpp
import numpy as np


def plot_coefficients(coefficients, output_file = None):
    mpp.bar(range(0, len(coefficients)), np.absolute(coefficients))
    mpp.xlabel('Feature Index')
    mpp.ylabel('Feature Parameter')
    if (output_file != None):
        mpp.savefig(output_file)
    else:
        mpp.show()

coefficients = np.array([1, 2, 0, 0, -1, 0.5, 0])
plot_coefficients(coefficients)
plot_coefficients(coefficients, 'test.png')