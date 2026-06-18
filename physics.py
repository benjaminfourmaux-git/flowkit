def reynold_number(
        density: float, velocity: float, length: float, viscosity: float) -> float:
    return density*velocity*length/viscosity