import numpy as np


def check_mean_density_validity_assuming_uniform_input(mean_density):
    if not (mean_density >= 0 and mean_density <= 1):
        raise ValueError(
            f"mean_density must be between 0 and 1. mean_density: {mean_density}"
        )

    unique_values = []
    for value in mean_density:
        if not any(np.isclose(value, unique_values, atol=0.02)):
            unique_values.append(value)

    if not (len(unique_values) <= 2):
        raise ValueError(
            f"mean_density must have at most two unique values. mean_density: {mean_density}"
        )
