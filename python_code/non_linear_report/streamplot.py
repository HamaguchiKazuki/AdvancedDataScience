import numpy as np
import matplotlib.pyplot as plt

w = 3
w_0 = 2
beta = 2

Y, X = np.mgrid[-w:w:100j, -w:w:100j]
U = Y
V = -w_0**2 * X + beta * X**3

x = np.linspace(-100, 100, num=1000)
v_x = (w_0**2 * x**2) /2 - (beta * x**4) / 4


fig, ax = plt.subplots(1)
# speed = np.sqrt(U**2 + V**2)

# Varying color along a streamline
# strm = plt.streamplot(X, Y, U, V, color=speed, linewidth=2, cmap='PiYG')
ax.streamplot(X, Y, U, V, linewidth=1)
ax.set(xlabel="$x_1$", ylabel="$x_2$", title="phase portrait")
# plt.tight_layout()
fig.savefig("phase_portrait.png")
# plt.show()


fig, ax = plt.subplots(1)
ax.plot(x, v_x)
ax.set(xlabel="$x$", ylabel="$V(x)$", title="Potential function")
fig.savefig("potential_function_close_cretical_point.png")