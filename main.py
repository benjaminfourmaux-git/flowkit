from physics import reynolds_number

re = reynolds_number(1000.0, 2.0, 0.05, 1.0e-3)
print(f"Reynolds = {re:.3f}")