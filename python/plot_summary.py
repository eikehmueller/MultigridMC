# Code generated with from ChatGPT (OpenAI GPT-5.3-mini) and reviewed by Eike Mueller
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Style
# -----------------------------
plt.rcParams.update(
    {
        "font.size": 20,
        "axes.titlesize": 24,
        "axes.labelsize": 22,
        "legend.fontsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "lines.linewidth": 3.2,
    }
)

# -----------------------------
# Data
# -----------------------------
n = np.array([16**3, 32**3, 48**3, 64**3])

iact_gibbs = np.array([2.6, 4.7, 10.3, 20.6])
iact_gibbs_err = np.array([0.3, 0.7, 2.0, 5.3])

iact_mgmc = np.array([1.32, 1.20, 1.26, 1.28])
iact_mgmc_err = np.array([0.19, 0.14, 0.17, 0.17])

t_chol = np.array([0.48, 12.58, 82.81, 317.23])
t_gibbs = np.array([1.47, 24.60, 257.20, 1030.82])
t_mgmc = np.array([0.98, 9.17, 39.17, 87.00])

# -----------------------------
# Colors
# -----------------------------
col_gibbs = "C0"
col_chol = "#4C78A8"
col_mgmc = "#D62728"

err_style = dict(capsize=6, elinewidth=3.0, capthick=3.0)

# -----------------------------
# Figure
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# =========================================================
# LEFT: Sampling efficiency
# =========================================================
ax = axes[0]

ax.errorbar(
    n,
    iact_gibbs,
    yerr=iact_gibbs_err,
    color=col_gibbs,
    marker="o",
    markersize=15,
    mfc="white",
    mec=col_gibbs,
    mew=2.5,
    label="Gibbs",
    **err_style,
)

ax.errorbar(
    n,
    iact_mgmc,
    yerr=iact_mgmc_err,
    color=col_mgmc,
    marker="s",
    markersize=15,
    mfc="white",
    mec=col_mgmc,
    mew=2.5,
    label="MGMC (our method)",
    **err_style,
)

ax.axhline(
    1,
    linestyle="--",
    linewidth=3.0,
    color="black",
    alpha=0.85,
    label="independent samples",
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(0.75, 35)

ax.set_xlabel("Problem size (number of unknowns)")
ax.set_ylabel("Integrated Autocorrelation time (IACT)")
ax.set_title("Sampling efficiency")

# -----------------------------
# LEFT annotations (data-space, improved placement)
# -----------------------------
ax.annotate(
    "",
    xy=(1.2 * n[0], 10),
    xytext=(0, -100),  # offset in points
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", lw=3, color="black", mutation_scale=30),
    ha="right",
    va="center",
    fontsize=14,
)

ax.text(
    1.3 * n[0],
    4,
    r"worse",
    color="black",
    fontsize=28,
    rotation=90,
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
)

ax.text(
    4e4,
    6.8,
    "reference",
    color=col_gibbs,
    fontsize=22,
    rotation=30,
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
)

ax.text(
    2e4,
    1.7,
    "MGMC (our method)",
    color=col_mgmc,
    fontsize=22,
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
)

ax.grid(True, which="both", linestyle="--", alpha=0.5)
ax.legend(frameon=True)

# =========================================================
# RIGHT: Computational cost
# =========================================================
ax = axes[1]

ax.annotate(
    "",
    xy=(1.2 * n[0], 100),
    xytext=(0, -100),  # offset in points
    textcoords="offset points",
    arrowprops=dict(arrowstyle="<-", lw=3, color="black", mutation_scale=30),
    ha="right",
    va="center",
    fontsize=14,
)

ax.text(
    1.4 * n[0],
    20,
    r"better",
    color="black",
    fontsize=28,
    rotation=90,
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
)

ax.plot(
    n,
    t_chol,
    color=col_chol,
    marker="^",
    markersize=15,
    mfc="white",
    mec=col_chol,
    mew=2.5,
    label="Cholesky",
)

ax.plot(
    n,
    t_gibbs,
    color=col_gibbs,
    marker="o",
    markersize=15,
    mfc="white",
    mec=col_gibbs,
    mew=2.5,
    label="Gibbs",
)

ax.plot(
    n,
    t_mgmc,
    color=col_mgmc,
    marker="s",
    markersize=15,
    mfc="white",
    mec=col_mgmc,
    mew=2.5,
    label="MGMC (our method)",
)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("Problem size (number of unknowns)")
ax.set_ylabel("Time per independent sample")
ax.set_title("Computational cost")

# -----------------------------
# RIGHT annotations (data-space, consistent placement)
# -----------------------------
ax.annotate(
    "",
    xy=(n[-1], 1.1 * t_mgmc[-1]),
    xytext=(0, 40),  # offset in points
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", lw=6, color="red", mutation_scale=25),
    ha="right",
    va="center",
    fontsize=14,
)

ax.text(
    1.2 * n[-1],
    0.8 * np.sqrt(t_mgmc[-1] * t_chol[-1]),
    r"3x",
    color="red",
    fontsize=36,
    fontweight="bold",
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
)

ax.text(
    4e4,
    100,
    "reference",
    color=col_gibbs,
    fontsize=22,
    rotation=35,
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
)

ax.text(
    3e4,
    2.5,
    "MGMC (our method)",
    color=col_mgmc,
    fontsize=22,
    rotation=25,
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
)

ax.grid(True, which="both", linestyle="--", alpha=0.5)
ax.legend(frameon=True)

# -----------------------------
# Layout
# -----------------------------
plt.tight_layout()
plt.savefig("summary.png", dpi=300, bbox_inches="tight")
