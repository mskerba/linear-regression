import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (ensures 3D projection registered)

def read_data(file):
    try:
        with open(file, "r") as f:
            lines = f.read().strip().split("\n")
        header, rows = lines[0], lines[1:]
        data = [[float(v) for v in r.split(",")] for r in rows if r]
        return data
    except BaseException:
        print("Error during data reading")
        exit()

def read_thetas():
    try:
        thetas = open("thetas", "r")
        theta = thetas.read().split(",")
        theta = [float(i) for i in theta]
        thetas.close()
    except BaseException:
        return [0, 0]

    return theta


def plot_cost_surface_3d(x, y, theta0_bounds=None, theta1_bounds=None, num=120, theta_hat=None):
    """
    Plot the 3D MAE/L1 cost J(theta0, theta1) for y ~ theta0 + theta1 * x.

    Args:
        x, y: 1D array-like of same length
        theta0_bounds: (min, max) for theta0 (y-intercept). If None, auto-computed.
        theta1_bounds: (min, max) for theta1 (slope). If None, auto-computed.
        num: grid resolution per axis (default 120)
        theta_hat: optional (theta0, theta1) to highlight the trained solution
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = x.size
    if m == 0:
        raise ValueError("Empty data.")

    # ---- pick plotting ranges if not provided
    if theta0_bounds is None:
        y_p5, y_p95 = np.percentile(y, [5, 95])
        pad0 = 0.2 * (y_p95 - y_p5 + 1e-9)
        theta0_bounds = (y_p5 - pad0, y_p95 + pad0)

    if theta1_bounds is None:
        # robust slope guess; fall back if x has tiny variance
        xstd = np.std(x) + 1e-12
        ystd = np.std(y) + 1e-12
        try:
            slope_est = np.polyfit(x, y, 1)[0]
        except Exception:
            slope_est = ystd / xstd
        if abs(slope_est) < 1e-12:
            slope_est = ystd / xstd
        span = 6.0 * abs(slope_est) + 1e-6  # wide window around estimate
        theta1_bounds = (slope_est - span/2, slope_est + span/2)

    t0 = np.linspace(theta0_bounds[0], theta0_bounds[1], num)
    t1 = np.linspace(theta1_bounds[0], theta1_bounds[1], num)
    T0, T1 = np.meshgrid(t0, t1)  # shape (num, num)

    # ---- compute MAE cost over the grid (vectorized)
    # predictions per grid cell across samples -> shape (num, num, m)
    pred = T0[..., None] + T1[..., None] * x[None, None, :]
    J = np.mean(np.abs(pred - y[None, None, :]), axis=-1)

    # ---- plot 3D surface
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(T0, T1, J, rstride=1, cstride=1, linewidth=0, antialiased=True, alpha=0.9)
    ax.set_xlabel(r'$\theta_0$')
    ax.set_ylabel(r'$\theta_1$')
    ax.set_zlabel(r'$J(\theta_0,\theta_1)$ (MAE)')
    ax.set_title('MAE Cost Surface for Linear Model')

    # mark trained solution if provided
    if theta_hat is not None:
        th0, th1 = theta_hat
        # compute its cost to place the point properly
        J_hat = np.mean(np.abs((th0 + th1 * x) - y))
        ax.scatter(th0, th1, J_hat, s=60, marker='o')

    plt.tight_layout()
    plt.show()
    return T0, T1, J

def plotting_data(mileages_raw, prices, theta0, theta1):
    try:
        plt.figure()
        plt.title("Data and hypothesis")
        plt.xlabel("Mileage")
        plt.ylabel("Price")
        plt.scatter(mileages_raw, prices, color="blue")
        plt.plot(
        [min(mileages_raw), max(mileages_raw)],
        [   
            theta0 + theta1 * min(mileages_raw),
            theta0 + theta1 * max(mileages_raw),
        ],
        "r",
        )
        plt.show()
    except BaseException as e:
        print("Error during plotting: ", e)
        exit()