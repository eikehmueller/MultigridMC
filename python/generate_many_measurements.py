"""Generate lots of measurements for 2d setup

Randomly choose a large number of measurements with a specified mean and
covariance
"""

import argparse
import numpy as np
from matplotlib import pyplot as plt

nmeas = 8
dim = 2
epsilon = 0.01

parser = argparse.ArgumentParser("Specifications")

parser.add_argument(
    "--filename",
    type=str,
    action="store",
    default="measurements.cfg",
    help="file to store results in",
)

parser.add_argument(
    "--nmeas",
    type=int,
    action="store",
    default=nmeas,
    help="number of measurements",
)

parser.add_argument(
    "--epsilon",
    type=float,
    action="store",
    default=epsilon,
    help="measurement error",
)

parser.add_argument(
    "--missing_data",
    action="store_true",
    default=False,
    help="omit data in the centre",
)


args = parser.parse_args()

print(f"Creating measurements file {args.filename}")
print()
print(f" n = {args.nmeas}")
print(f" epsilon = {args.epsilon}")
print(f" missing data = {args.missing_data}")

rng = np.random.default_rng(seed=2718417)

nmodes = 8
coefficients = rng.normal(size=(nmodes, nmodes))


def average(x, y):
    """Evaluate mean field at a certain position

    :arg x: x-coordinate of point at which to evaluate field
    :arg y: y-coordinate of point at which to evaluate field
    """
    F = np.asarray(
        [
            1 / np.sqrt(2) * np.exp(-0.3 * j) * np.sin((j + 1) * np.pi * x)
            for j in range(nmodes)
        ]
    )
    G = np.asarray(
        [
            1 / np.sqrt(2) * np.exp(-0.3 * k) * np.sin((k + 1) * np.pi * y)
            for k in range(nmodes)
        ]
    )
    return np.dot(
        F,
        np.dot(coefficients, G),
    )


v_mean = np.vectorize(average)

h = 0.01
X = np.arange(0, 1 + h, h)
Y = np.arange(0, 1 + h, h)
XX, YY = np.meshgrid(X, Y)
Z = np.asarray(
    [[average(x, y) for x, y in zip(xrow, yrow)] for xrow, yrow in zip(XX, YY)]
)
contours = plt.contourf(X, X, Z, levels=32)
ax = plt.gca()
ax.set_aspect("equal")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

if args.missing_data:
    points = np.empty((args.nmeas + 1, 2))
    radius = 0.25
    x0, y0 = 0.5, 0.5
    j = 0
    while j < args.nmeas + 1:
        p = rng.uniform(size=2, low=0, high=1)
        if (p[0] - x0) ** 2 + (p[1] - y0) ** 2 > radius**2:
            points[j, :] = p
            j += 1
else:
    points = rng.uniform(size=(args.nmeas + 1, 2), low=0, high=1)
mean = np.asarray([average(x, y) for x, y in points])

# find the point closest to the centre of the domain
idx = np.argsort(np.linalg.norm(points - np.asarray([0.5, 0.5]), axis=1))
points = points[idx]
mean = mean[idx]

# measurements
plt.scatter(
    points[1:, 0],
    points[1:, 1],
    c=mean[1:],
    marker="o",
    vmin=np.min(Z),
    vmax=np.max(Z),
    edgecolors="white",
    s=10,
)
# observation
plt.scatter(
    points[0, 0],
    points[0, 1],
    c=mean[0],
    marker="o",
    vmin=np.min(Z),
    vmax=np.max(Z),
    edgecolors="black",
    s=10,
)
# observation
plt.scatter(
    [0.5],
    [0.5],
    c="black",
    marker="x",
    s=10,
)
plt.colorbar(contours)


variance = args.epsilon**2 * np.ones(args.nmeas + 1)

plt.savefig("many_points.pdf", bbox_inches="tight")

# Print results in a format that can be used in the configuration file
with open(args.filename, mode="w", encoding="utf8") as f:
    print(f"// automatically generated measurement file for MGMC", file=f)
    print("", file=f)
    print(f"n = {args.nmeas};", file=f)
    print(f"dim = {dim};", file=f)
    print(f"measurement_locations = ", repr(list(points[1:, :].flatten())), ";", file=f)
    print(f"mean = ", repr(list(mean[1:].flatten())), ";", file=f)
    print(f"variance = ", repr(list(variance[1:].flatten())), ";", file=f)
