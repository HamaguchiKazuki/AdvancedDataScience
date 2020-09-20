import numpy as np
import matplotlib.pyplot as plt

# v = np.arange(-3.0, 3.0, 0.1)
# u = np.arange(-3.0, 3.0, 0.1)
# h = 1
# K = 1
# beta = 8 / 3
# sigma = 4.5

# H = (3 / 32) * beta * (u ** 2 + v ** 2)**2 - (sigma / 2) * (u ** 2 + v ** 2) - K * u - h
A = 3
B = 4
C = 6
v = np.arange(-3.0, 3.0, 0.1)
x = v**2
y = A*x**2 + B*x + C

plt.plot(x, y)
plt.savefig("img/playground.png")
plt.close('all')

def duffing(var, t, gamma, a, b, F0, omega, delta):
    """
    var = [u, v]
    du/dt = (sigma - 3/8 * beta * (u**2+v**2))*v
    dv/dt = -gamma*p + 2*a*x - 4*b*x**3 + F0*cos(omega*t + delta)
    """
    x_dot = var[1]
    p_dot = -gamma * var[1] + 2 * a * var[0] - 4 * b * var[0]**3 + F0 * np.cos(omega * t + delta)

    return np.array([x_dot, p_dot])

from scipy.integrate import odeint


def f(Y, t):
    y1, y2 = Y
    return [y2, -np.sin(y1)]

y1 = np.linspace(-2.0, 8.0, 20)
y2 = np.linspace(-2.0, 2.0, 20)

for y20 in [0, 0.5, 1, 1.5, 2, 2.5]:
    tspan = np.linspace(0, 100, 200)
    y0 = [0.0, y20] # 初期値問題
    ys = odeint(f, y0, tspan)
    plt.plot(ys.T[0], ys.T[1], '.') # path
    plt.plot([ys[0,0]], [ys[0,1]], 'o') # start
    plt.plot([ys[-1, 0]], [ys[-1, 1]], 'd')  # end

plt.xlim([-2, 8])
plt.savefig('img/phase-portrait-2.png')