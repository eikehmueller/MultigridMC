#ifndef LINEAR_OPERATOR_HH
#define LINEAR_OPERATOR_HH LINEAR_OPERATOR_HH
#include <memory>
#include <random>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "lattice/lattice.hh"
#include "intergrid/intergrid_operator.hh"

/** @file LinearOperator.hh
 * @brief Header file for LinearOperator classes
 */

/** @class LinearOperator
 *
 * @brief LinearOperator class
 *
 * Represents linear operators of the form
 *
 *  A = A_0 + B \Sigma^{-1} B^T
 *
 *
 * where A, A_0 are n x n matrix, \Sigma is a m x m matrix (with m << n) and
 * B is a n x m matrix.
 */

class LinearOperator
{
public:
    /** @brief Type of sparse matrix*/
    typedef Eigen::SparseMatrix<double> SparseMatrixType;
    typedef Eigen::MatrixXd DenseMatrixType;

    /** @brief Create a new instance
     *
     * @param[in] lattice_ underlying lattice
     * @param[in] m_lowrank_ the dimension of the low-rank space spanned by B
     */
    LinearOperator(const std::shared_ptr<Lattice> lattice_,
                   const unsigned int m_lowrank_ = 0) : lattice(lattice_),
                                                        m_lowrank(m_lowrank_),
                                                        A_sparse(lattice_->M, lattice_->M),
                                                        B(lattice_->M, m_lowrank_),
                                                        Sigma_inv_BT(m_lowrank_, lattice_->M),
                                                        Sigma_inv(m_lowrank_, m_lowrank_)
    {
    }

    /** @brief destructor */
    virtual ~LinearOperator() = default;

    /** @brief Extract underlying lattice */
    std::shared_ptr<Lattice> get_lattice() const
    {
        return lattice;
    }

    /** @brief Apply the linear LinearOperator
     *
     * Compute y = Ax
     *
     * @param[in] x input state
     * @param[out] y output state
     */
    void apply(const Eigen::VectorXd &x, Eigen::VectorXd &y)
    {
        // Apply the sparse matrix A_0
        y = A_sparse * x;
        // Add low-rank correction, if necessary
        if (m_lowrank > 0)
        {
            Eigen::VectorXd Sigma_BT_x = Sigma_inv_BT * x;
            y += B * Sigma_BT_x;
        }
    }

    /** @brief get number of unknowns */
    const unsigned int get_ndof() const { return A_sparse.rows(); }

    /** @brief get the dimension of the low-rank correction */
    const unsigned int get_m_lowrank() const { return m_lowrank; }

    /** @brief Coarsen linear operator to the next-coarser level
     *
     * Compute A^{c} = I_{2h}^{h} A I_{h}^{2h}
     *
     * @param[in] intergrid_operator The intergrid operator to be used
     */
    LinearOperator coarsen(const std::shared_ptr<IntergridOperator> intergrid_operator) const;

    /** @brief Convert to sparse storage format */
    const SparseMatrixType &get_sparse() const { return A_sparse; };

    /** @brief Return B (in sparse storage format) */
    const SparseMatrixType &get_B() const { return B; };

    /** @brief Return Sigma^{-1} (in dense storage format) */
    const DenseMatrixType &get_Sigma_inv() const { return Sigma_inv; };

    /** @brief compute (dense) precision matrix */
    DenseMatrixType precision() const;

    /** @brief compute (dense) covariance matrix */
    DenseMatrixType covariance() const
    {
        return precision().inverse();
    }

protected:
    /** @brief underlying lattice */
    const std::shared_ptr<Lattice> lattice;
    /** @brief dimension of low rank space spanned by B*/
    const unsigned int m_lowrank;
    /** @brief sparse representation of matrix A_0*/
    SparseMatrixType A_sparse;
    /** @brief matrix B */
    SparseMatrixType B;
    /** @brief matrix Sigma^{-1}.B^T */
    SparseMatrixType Sigma_inv_BT;
    /** @brief dense representation of m x m matrix Sigma^{-1} */
    DenseMatrixType Sigma_inv;
};

#endif // LINEAR_OPERATOR_HH
