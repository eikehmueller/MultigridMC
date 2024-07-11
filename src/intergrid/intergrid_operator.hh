#ifndef INTERGRID_OPERATOR_HH
#define INTERGRID_OPERATOR_HH INTERGRID_OPERATOR_HH
#include <memory>
#include <cmath>
#include <map>
#include <set>
#include <utility>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "lattice/lattice.hh"

/** @file intergrid_operator.hh
 * @brief Header file for intergrid operator classes
 */

/** @class IntergridOperator
 *
 * @brief IntergridOperator that can be used as a base class
 *
 * A linear operator class provides functionality for performing the
 * following operations:
 *
 *  - Restrict a vector on a lattice to a vector on the next-coarser lattice
 *
 *      x^{c} = I_{h}^{2h} x
 *
 *  - Prolongate-add a vector from the next-coarser level
 *
 *      x += I_{2h}^{h} x^{c}
 *
 *  - Coarsen a linear operator to the next-coarser lattice using the
 *    Galerkin triple-product
 *
 *       A^{c} = I_{h}^{2h} A I_{2h}^{h}
 *
 *  - Interpolate a vector to the next-coarser level
 *
 *      x^{c} = tilde{I}_{h}^{2h} x
 *
 * It is assumed that I_{h}^{2h} = (I_{2h}^{h})^T.
 * The sparsity structure of the intergrid operator is assumed to be fixed and is described
 * by a stencil; the entries of the matrices I_{h}^{2h} and I_{2h}^{h} are also assumed to
 * be constant. Note in particular that this implies that each coarse level unknown depends on
 * the fine level unknown in exactly the same way.
 */
class IntergridOperator
{
public:
    /** @brief Type of sparse matrix*/
    typedef Eigen::SparseMatrix<double> SparseMatrixType;
    /** @brief Create a new instance
     *
     * @param[in] lattice_ underlying lattice
     * @param[in] stencil_size_ size of intergrid stencil
     * @param[in] interpolation_stencil_size_ size of interpolation stencil
     */
    IntergridOperator(const std::shared_ptr<Lattice> lattice_,
                      const int stencil_size_,
                      const int interpolation_stencil_size_) : lattice(lattice_),
                                                               stencil_size(stencil_size_),
                                                               interpolation_stencil_size(interpolation_stencil_size_)
    {
        matrix = new double[stencil_size];
        interpolation_matrix = new double[interpolation_stencil_size];
        colidx = new unsigned int[lattice->get_coarse_lattice()->Nvertex * stencil_size];
        interpolation_colidx = new unsigned int[lattice->get_coarse_lattice()->Nvertex * interpolation_stencil_size];
    }

    ~IntergridOperator()
    {
        delete[] matrix;
        delete[] colidx;
    }

    /** @brief Restrict a vector to the next-coarser level
     *
     * Compute x^{c} = I_{h}^{2h} x
     *
     * @param[in] x input vector on current lattice
     * @param[out] x^{c} output vector on next-coarser lattice
     */
    virtual void restrict(const Eigen::VectorXd &x, Eigen::VectorXd &x_coarse)
    {
        std::shared_ptr<Lattice> coarse_lattice = lattice->get_coarse_lattice();

        for (unsigned int ell_coarse = 0; ell_coarse < coarse_lattice->Nvertex; ++ell_coarse)
        {
            double result = 0;
            for (unsigned k = 0; k < stencil_size; ++k)
            {
                unsigned int ell = colidx[ell_coarse * stencil_size + k];
                result += matrix[k] * x[ell];
            }
            x_coarse[ell_coarse] = result;
        }
    }

    /** @brief Prolongate-add a vector from the next-coarser level
     *
     * Compute x += alpha * I_{2h}^{h} x^{c}
     *
     * The prolongation operator does not depend on position, so the operation can be written as:
     *
     * x_i  = x_i + alpha * \sum_j (I_{2h}^h)_{ij} x^{c}_j
     *      = x_i + alpha * \sum_{\sigma} (I_{2h}^h)_{i,2*i+\sigma} x^{c}_{2*i+\sigma}
     *      = x_i + alpha * \sum_{\sigma} P_{\sigma} x^{c}_{2*i+\sigma}
     *
     * where \sigma runs over all possible offsets.
     *
     * @param[in] alpha scale factor alpha
     * @param[in] x_coarse input vector on next-coarser lattice
     * @param[out] x output vector on current lattice
     */
    virtual void prolongate_add(const double alpha,
                                const Eigen::VectorXd &x_coarse,
                                Eigen::VectorXd &x)
    {
        std::shared_ptr<Lattice> coarse_lattice = lattice->get_coarse_lattice();
        for (unsigned int ell_coarse = 0; ell_coarse < coarse_lattice->Nvertex; ++ell_coarse)
        {
            double x_coarse_ell = x_coarse[ell_coarse];
            for (unsigned k = 0; k < stencil_size; ++k)
            {
                unsigned int ell = colidx[ell_coarse * stencil_size + k];
                x[ell] += alpha * matrix[k] * x_coarse_ell;
            }
        }
    };

    /** @brief Interpolate a vector to the next-coarser level
     *
     * Compute x^{c} = tilde{I}_{h}^{2h} x
     *
     * @param[in] x input vector on current lattice
     * @param[out] x^{c} output vector on next-coarser lattice
     */
    virtual void interpolate(const Eigen::VectorXd &x, Eigen::VectorXd &x_coarse)
    {
        std::shared_ptr<Lattice> coarse_lattice = lattice->get_coarse_lattice();

        for (unsigned int ell_coarse = 0; ell_coarse < coarse_lattice->Nvertex; ++ell_coarse)
        {
            double result = 0;
            for (unsigned k = 0; k < interpolation_stencil_size; ++k)
            {
                unsigned int ell = interpolation_colidx[ell_coarse * interpolation_stencil_size + k];
                result += interpolation_matrix[k] * x[ell];
            }
            x_coarse[ell_coarse] = result;
        }
    }

    /** @brief convert restriction operator to a sparse matrix */
    const Eigen::SparseMatrix<double> to_sparse() const
    {
        std::shared_ptr<Lattice> coarse_lattice = lattice->get_coarse_lattice();
        typedef Eigen::Triplet<double> T;
        std::vector<T> triplet_list;
        unsigned int nrow = coarse_lattice->Nvertex;
        unsigned int ncol = lattice->Nvertex;
        unsigned int nnz = stencil_size * nrow;
        triplet_list.reserve(nnz);
        for (unsigned int ell = 0; ell < nrow; ++ell)
        {
            {
                for (int k = 0; k < stencil_size; ++k)
                {
                    triplet_list.push_back(T(ell, colidx[ell * stencil_size + k], matrix[k]));
                }
            }
        }
        SparseMatrixType A_sparse(nrow, ncol);
        A_sparse.setFromTriplets(triplet_list.begin(), triplet_list.end());
        return A_sparse;
    }

protected:
    /** @brief Compute column indices on entire lattice
     *
     * @param[in] shift vector with shift indices
     * @param[in] stencil_size_ size of stencil
     * @param[out] colidx_ptr array with column indices to set
     */
    void compute_colidx(const std::vector<Eigen::VectorXi> shift,
                        const int stencil_size_,
                        unsigned int *colidx_ptr);

    /** @brief underlying lattice */
    const std::shared_ptr<Lattice> lattice;
    /** @brief size of stencil */
    const int stencil_size;
    /** @brief size of stencil for interpolation matrix */
    const int interpolation_stencil_size;
    /** @brief underlying matrix */
    double *matrix;
    /** @brief underlying interpolation matrix */
    double *interpolation_matrix;
    /** @brief indirection map */
    unsigned int *colidx;
    /** @brief indirection map */
    unsigned int *interpolation_colidx;
};

/* ******************** factory classes ****************************** */

/** @brief Intergrid factory base class */
class IntergridOperatorFactory
{
public:
    /** @brief extract a smoother for a given action */
    virtual std::shared_ptr<IntergridOperator> get(std::shared_ptr<Lattice> lattice) = 0;
};

#endif // INTERGRID_OPERATOR_HH