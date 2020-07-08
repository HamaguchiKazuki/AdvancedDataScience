import numpy as np
import matplotlib.pyplot as plt
import random
random.seed(42)

w = 3
K = 1
beta = 9/3
min_u = ((-8*K)/(6*beta)) ** (1 / 3) # or (-8*K/6/beta) ** (1 / 3)
cr_sigma = (3 / 8) * beta * (np.abs(min_u) ** 2) + K / np.abs(min_u)
root3_sigma = random.uniform(cr_sigma, 10)
root1_sigma = random.uniform(0, cr_sigma)
delta = 0
print(f"cr sigma: {cr_sigma}")
print(f"root3 sigma: {root3_sigma}")
print(f"root1 sigma: {root1_sigma}")


def phase_portrait(K, beta, sigma, save_name):
    v_solution = [0, 0, 0]
    u_solution = np.roots([3 / 8 * beta, 0, -sigma, -K])
    print(f"v = 0, u solution: {u_solution}")

    ## Y = X2, X = X1
    Y, X = np.mgrid[-w:w:100j, -w:w:100j] ## 周期的な値作ってる
    print(X)
    print()
    print(Y)
    # Phase Portrait
    U = -delta*X - (sigma - 3/8*beta * (X**2 + Y**2))*Y
    V = -delta*Y + (sigma - 3/8*beta * (X**2 + Y**2))*X + K


    # 2D velocytiy param
    # x = np.linspace(-100, 100, num=1000)
    # v_x = (w_0**2 * x**2) /2 - (beta * x**4) / 4


    fig, ax = plt.subplots(1)
    # speed = np.sqrt(U**2 + V**2)
    title = \
        f""" phase portrait
        beta: {beta:.3f} K: {K:.3f}
        sigma: {sigma:.4f}
        v = {v_solution}, 
        u = {u_solution}
        """
    # Varying color along a streamline
    # strm = plt.streamplot(X, Y, U, V, color=speed, linewidth=2, cmap='PiYG')
    ax.streamplot(X, Y, U, V, linewidth=1)
    ax.set(xlabel="$x_1$", ylabel="$x_2$", title=title)
    if save_name == "root1_sigma":
        ax.plot(float(u_solution[0].real), float(v_solution[0].real), marker="d", linestyle='None', label=f"{save_name} fixed point")
    else:
        ax.plot(u_solution, v_solution, marker="d", linestyle='None', label=f"{save_name} fixed point")
    ax.legend()
    plt.tight_layout()
    # plt.show()
    fig.savefig(f"img/{save_name}_phase_portrait.png")


    # fig, ax = plt.subplots(1)
    # ax.plot(x, v_x)
    # ax.set(xlabel="$x$", ylabel="$V(x)$", title="Potential function")
    # fig.savefig("img/potential_function.png")
    return X, Y

X, Y = phase_portrait(K, beta, cr_sigma, save_name="cr_sigma")
_,_=phase_portrait(K, beta, root3_sigma, save_name="root3_sigma")
_,_=phase_portrait(K, beta, root1_sigma, save_name="root1_sigma")