import numpy as np
import matplotlib.pyplot as plt


K = 1
beta = 8/3
min_u = ((-8*K)/(6*beta)) ** (1 / 3) # or (-8*K/6/beta) ** (1 / 3)
small_sigma = (3/8) * beta * (np.abs(min_u)**2) + K/np.abs(min_u)
delta = 0