import numpy as np

def second_derivative(values: np.ndarray, dx: float)-> np.ndarray:

    d2 = np.zeros_like(values)

    d2[1:-1] = (values[:-2] - 2*values[1:-1] + values[2:])/dx**2
    return d2


def sine_second_derivative_error(n:int)->tuple[float,float]:

    x = np.linspace(0.0, 2.0 * np.pi, n)
    dx = x[1]-x[0]
    numeric = second_derivative(np.sin(x),dx)
    exact = -np.sin(x)
    max_error = np.max(np.abs(numeric[1:-1] - exact[1:-1]))
    return dx, max_error