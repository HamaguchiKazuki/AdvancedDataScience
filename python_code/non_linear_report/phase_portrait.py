from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt


class OscillatorParam():
    def __init__(self, omega_0, beta):
        self.omega_0 = omega_0
        self.beta = beta


def standard_form(X, t):
    osc_param = OscillatorParam(omega_0=1, beta=1)
    x1, x2 = X
    # x^._1 = x_2 = a*cos(w_0*t)
    # x^._2 = -w^2_0 * x_1 + b * x^3_1 = b*sin(w_0*t)
    # return [np.cos(x2), -osc_param.omega_0**2 * np.sin(x1) + osc_param.beta * np.sin(x1)**3]
    # return [x2, -np.sin(x1)]
    return [x1+2*x2, 3*x1+2*x2]


x1 = np.linspace(-2.0, 8.0, 40)
x2 = np.linspace(-2.0, 2.0, 40)

X1, X2 = np.meshgrid(x1, x2)

t = 0

# make a vector
u, v = np.zeros(X1.shape), np.zeros(X2.shape)

# get meshgrid matrix
meshgrid_row_idx, meshgrid_col_idx = X1.shape

for row in range(meshgrid_row_idx):
    for col in range(meshgrid_col_idx):
        # x1-x2 plane
        # x1, x2: start point u,v:end point
        u[row, col], v[row, col] = standard_form(
            [X1[row, col], X2[row, col]], t)

fig, ax = plt.subplots(1)
ax.quiver(X1, X2, u, v, color='g')
ax.set(xlim=(-10, 10), ylim=(-3, 4),
       xlabel='$x_1$', ylabel='$x_2$', title='PHASE PORTRAIT of Center-Type Fixed Point')

# for x2_20 in [0, 0.5, 1, 1.5, 2, 2.5]:
#     tspan = np.linspace(0, 50, 200)
#     x2_0 = [0.0, x2_20]
#     x2_start = odeint(standard_form, x2_0, tspan)
#     ax.plot(x2_start[:, 0], x2_start[:, 1], 'b-')  # path
#     ax.plot([x2_start[0, 0]], [x2_start[0, 1]], 'o')  # start
#     ax.plot([x2_start[-1, 0]], [x2_start[-1, 1]], 's')  # end

plt.show()
