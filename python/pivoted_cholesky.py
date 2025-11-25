import numpy as np
from matplotlib import pyplot as plt
from scipy import special as sp

"""
Tests for pivoted Cholesky factorisation in

     Harbrecht, H., Peters, M. and Schneider, R., 2012:
     On the low-rank approximation by the pivoted Cholesky decomposition
     Applied numerical mathematics, 62(4), pp.428-440
"""


def create_1d_matrix(Lambda, n):
    """Construct 1d discretisation of the following linear operator:

        Au = -laplace u + sigma u

    with periodic boundary conditions

    :arg sigma: coefficient of zero-order term
    :arg n: number of points
    """
    A = np.zeros((n, n))
    h_sq = 1.0 / n**2
    for j in range(n):
        A[j, j] = 2 + h_sq / Lambda**2
        A[j, (j + 1) % n] = -1
        A[j, (j - 1) % n] = -1
    return A


def create_2d_matrix(n, Lambda, sigma):
    """Construct 2d discretisation of the following linear operator:

        Au = -laplace u + sigma u

    with periodic boundary conditions

    :arg n: number of points
    :arg Lambda: correlation length
    :arg sigma: covariance of global measurement
    """
    A = np.zeros((n**2, n**2))
    h_sq = 1 / n**2
    for j in range(n**2):
        A[j, j] = 4 + h_sq / Lambda**2
        A[j, (j + 1) % n**2] = -1
        A[j, (j - 1) % n**2] = -1
        A[j, (j + n) % n**2] = -1
        A[j, (j - n) % n**2] = -1
    v = np.ones(n**2) / n**2
    A += 1 / sigma * np.outer(v, v)
    v = np.zeros(n**2)
    v[7] = 1.0
    A += 1 / sigma * np.outer(v, v)
    return A


def create_2d_correlation(n, Lambda):
    """
    :arg n: number of points
    :arg Lambda: smoothness parameter
    """

    A = np.zeros((n**2, n**2))
    h_sq = 1.0 / n**2
    for j_1 in range(n):
        for j_2 in range(n):
            for k_1 in range(n):
                for k_2 in range(n):
                    dsq = h_sq * ((j_1 - k_1) ** 2 + (j_2 - k_2) ** 2)
                    f = 1 / (Lambda**2 + dsq)
                    A[j_1 * n + j_2, k_1 * n + k_2] = f
    return A


def cholesky_crout(A):
    """Compute pivoted Cholesky L.L^T factorisation of a nxn matrix

    :arg A: matrix A
    """
    n, m = A.shape
    assert n == m  # ensure that we are dealing with a square matrix
    L = np.zeros((n, n))
    diag = np.array(np.diagonal(A))
    for m in range(n):
        L[m, m] = np.sqrt(diag[m])
        for i in range(m + 1, n):
            L[m, i] = (A[m, i] - np.dot(L[:m, m], L[:m, i])) / L[m, m]
            diag[i] -= L[m, i] ** 2
    return L.T


def cholesky_crout_diagonal(A):
    """Compute pivoted Cholesky L.D.L^T factorisation of a nxn matrix

    :arg A: matrix A
    """
    n, m = A.shape
    assert n == m  # ensure that we are dealing with a square matrix
    L = np.zeros((n, n))
    D = np.zeros(n)
    diag = np.array(np.diagonal(A))
    for m in range(n):
        D[m] = diag[m]
        L[m, m] = 1.0
        for i in range(m + 1, n):
            L[m, i] = (A[m, i] - np.dot(L[:m, m], D[:m] * L[:m, i])) / D[m]
            diag[i] -= D[m] * L[m, i] ** 2
    return L.T, D


def pivoted_cholesky(A, tolerance):
    """Compute pivoted Cholesky factorisation of a nxn matrix

    :arg A: matrix A
    """
    n, m = A.shape
    assert n == m  # ensure that we are dealing with a square matrix
    perm = list(range(n))
    L = np.zeros((n, n))
    diag = np.array(np.diagonal(A))
    error = np.linalg.norm(diag, ord=1)
    error0 = error
    rel_error = [1.0]
    for m in range(n):
        i = m + np.argmax(diag[perm[m:]])
        perm[i], perm[m] = perm[m], perm[i]
        L[perm[m], m] = np.sqrt(diag[perm[m]])
        for i in range(m + 1, n):
            L[perm[i], m] = (
                A[perm[i], perm[m]] - np.dot(L[perm[i], :m], L[perm[m], :m])
            ) / L[perm[m], m]
            diag[perm[i]] -= L[perm[i], m] ** 2
        error = np.linalg.norm(diag[perm[m + 1 :]], ord=1)
        rel_error.append(error / error0)
        if error / error0 < tolerance:
            break
    return L, rel_error


def truncated_svd(A, tolerance):
    """Compute truncated SVD"""
    U, S, VT = np.linalg.svd(A, full_matrices=True, hermitian=True)
    rel_error = []
    for j in range(A.shape[0]):
        rel_error.append(np.linalg.norm(A - U[:, :j] @ np.diag(S[:j]) @ VT[:j, :]))
    rel_error = np.asarray(rel_error)
    return rel_error / rel_error[0]


n = 16

sigma = 1.0e-4
tolerance = 1.0e-15
plt.clf()
ax = plt.gca()
ax.set_yscale("log")
for j, Lambda in enumerate([0.1, 0.2, 0.4, 0.8]):
    precision = create_2d_matrix(n, Lambda, sigma)
    cov = np.linalg.inv(precision)

    # cov = create_2d_correlation(n, Lambda)
    L, rel_error = pivoted_cholesky(cov, tolerance)
    # rel_error = truncated_svd(cov, tolerance)

    plt.plot(rel_error, linewidth=2, label=r"$\Lambda=" + f"{Lambda:3.1f}" + r"$")
plt.legend(loc="upper right")
plt.savefig("relative_error.pdf", bbox_inches="tight")
print(np.linalg.norm(L @ L.T - cov))
