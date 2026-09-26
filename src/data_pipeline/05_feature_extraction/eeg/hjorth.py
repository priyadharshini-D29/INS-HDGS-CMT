import numpy as np


def hjorth_parameters(signal):

    first_deriv  = np.diff(signal)
    second_deriv = np.diff(first_deriv)

    var_zero = np.var(signal)
    var_d1   = np.var(first_deriv)
    var_d2   = np.var(second_deriv)

    if var_zero < 1e-12:
        return 0.0, 0.0

    mobility = np.sqrt(var_d1 / var_zero)

    if var_d1 < 1e-12 or mobility < 1e-12:
        return mobility, 0.0

    complexity = np.sqrt(var_d2 / var_d1) / mobility

    return float(mobility), float(complexity)
