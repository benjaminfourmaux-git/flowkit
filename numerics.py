import numpy as np

def second_derivative(values: np.ndarray, dx: float)-> np.ndarray:

    d2 = np.zeros_like(values)

    d2[1:-1] = (values[:-2] - 2*values[1:-1] + values[2:])/dx**2
    return d2