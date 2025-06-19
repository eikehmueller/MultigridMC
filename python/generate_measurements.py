"""Generate measurements for 2d and 3d  operator

randomly choose a number of sample points in the unit square
or unit cube. Construct a random covariance matrix for these
measurements.
"""
import itertools
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def visualise_3d(p):
    """Visualise in 3 with plotly

    :arg p: data points
    """
    import plotly.express as px

    df = pd.DataFrame(p, columns=["x", "y", "z"])
    df["type"] = np.asarray(args.nmeas * ["measurement"] + ["sample point"])
    fig = px.scatter_3d(
        df, x="x", y="y", z="z", color="type", color_discrete_sequence=["blue", "red"]
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                nticks=10,
                range=[0, 1],
            ),
            yaxis=dict(
                nticks=10,
                range=[0, 1],
            ),
            zaxis=dict(
                nticks=10,
                range=[0, 1],
            ),
        ),
    )
    fig.show()


def distance_boundary(x):
    """Compute distance between a point and the boundary"""
    dim = x.shape[0]
    return min([min(abs(x[d]), abs(1.0 - x[d])) for d in range(dim)])


def sample_points(n, dim, dmin=0.1):
    """Draw random points in unit square or unit cube

    :arg n: number of sample points to draw
    :arg dim: dimension
    :arg dmin: minimum distance between any pair of points
    """
    rng = np.random.default_rng(seed=2154157)
    points = []
    while len(points) < n:
        x = np.asarray(rng.uniform(low=0.0, high=1.0, size=dim))
        if (distance_boundary(x) > 0.1) and all(
            [np.linalg.norm(x - p) > dmin for p in points]
        ):
            points.append(x)
    return points


def average(n, mu_low, mu_high):
    """Create array of measured average values

    The measured averages are assumed to be uniformly distributed
    in the interval [mu_low,mu_high].

    :arg n: number of measurement points
    :arg mu_low: lower bound on average
    :arg mu_high: upper bound on average
    """
    rng = np.random.default_rng(seed=2513267)
    return rng.uniform(size=n, low=mu_low, high=mu_high)


def variance(n, sigma_low, sigma_high):
    """Create diagonal variance

    The eigenvalues of the matrix are constructed such that they are
    in the range [sigma_min,sigma_max]

    :arg n: size of matrix (= number of meaurements)
    :arg sigma_low: lower bound on eigenvalues
    :arg sigma_high: upper bound on eigenvalues
    """
    rng = np.random.default_rng(seed=2511541)
    Sigma_diag = rng.uniform(low=sigma_low, high=sigma_high, size=n)
    return Sigma_diag


def format_array(a):
    """Convert an array to a form that can be printed

    :arg a: array to convert
    """
    return "[" + ", ".join([f"{x:.16f}" for x in list(a)]) + "]"


nmeas = 8
dim = 2
dmin = 0.2

parser = argparse.ArgumentParser("Specifications")
parser.add_argument(
    "--dim",
    metavar="dim",
    type=int,
    action="store",
    default=dim,
    choices=[2, 3],
    help="dimension",
)

parser.add_argument(
    "--nmeas",
    metavar="nmeas",
    type=int,
    action="store",
    default=nmeas,
    help="number of measurements",
)

args = parser.parse_args()

p = np.asarray(sample_points(args.nmeas + 1, args.dim, dmin))
mean = average(args.nmeas, 1.0, 4.0)
Sigma_diag = variance(args.nmeas, 1.0, 2.0)

# Print results in a format that can be used in the configuration file
print("dim = ", args.dim, ";")
print("n = ", args.nmeas, ";")
print("measurement_locations = ", format_array(p[:-1, :].flatten()), ";")
print("sample_location = ", format_array(p[-1, :].flatten()), ";")
print("mean = ", format_array(mean.flatten()), ";")
print("variance = ", format_array(Sigma_diag.flatten()), ";")

plt.clf()
fig = plt.figure()
if args.dim == 2:
    ax = fig.add_subplot()
    ax.set_aspect("equal")
    ax.plot(p[:-1, 0], p[:-1, 1], linewidth=0, markersize=4, marker="o", color="blue")
    ax.plot(p[-1, 0], p[-1, 1], linewidth=0, markersize=4, marker="o", color="red")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
else:
    ax = fig.add_subplot(projection="3d")
    ax.scatter(p[:-1, 0], p[:-1, 1], p[:-1, 2], color="blue")
    ax.scatter(p[-1, 0], p[-1, 1], p[-1, 2], color="red")
    ax = plt.gca()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    try:
        visualise_3d(p)
    except Exception:
        print("Need to install plotly for 3d visualisation...")
plt.savefig("points.pdf", bbox_inches="tight")
