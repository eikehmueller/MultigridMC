/** @file sor_smoother.cc
 *
 * @brief Implementation of sor_smoother.hh
 */

#include "sor_smoother.hh"

/* Construct a new instance */
SORSmoother::SORSmoother(const std::shared_ptr<LinearOperator> linear_operator_,
                         const double omega_,
                         const int nsmooth_,
                         const Direction direction_) : Base(linear_operator_),
                                                       omega(omega_),
                                                       nsmooth(nsmooth_),
                                                       direction(direction_)
{
    if (linear_operator->get_m_lowrank() > 0)
    {
        B = linear_operator->get_B();
        LinearOperator::SparseMatrixType A_sparse = linear_operator->get_sparse();
        LinearOperator::DenseMatrixType Sigma = linear_operator->get_Sigma().toDenseMatrix();
        // Construct the matrix 1/omega * D + L + L^T
        LinearOperator::SparseMatrixType D(A_sparse.diagonal().asDiagonal());
        LinearOperator::SparseMatrixType A_sparse_diag_scaled = A_sparse + (1. - omega) / omega * D;
        LinearOperator::DenseMatrixType dense_B_bar;
        if (direction == forward)
        {
            // Compute bar(B)_{FW} = (L   + 1/omega * D)^{-1} B ( Sigma + B^T (L   + 1/omega * D)^{-1} B )^{-1}
            LinearOperator::DenseMatrixType LD_inv_B = A_sparse_diag_scaled.triangularView<Eigen::Lower>().solve(B.toDense());
            dense_B_bar = LD_inv_B * (Sigma + B.transpose() * LD_inv_B).inverse();
        }
        else
        {
            // Compute bar(B)_{BW} = (L^T + 1/omega * D)^{-1} B ( Sigma + B^T (L^T + 1/omega * D)^{-1} B )^{-1}
            LinearOperator::DenseMatrixType LTD_inv_B = A_sparse_diag_scaled.triangularView<Eigen::Upper>().solve(B.toDense());
            dense_B_bar = LTD_inv_B * (Sigma + B.transpose() * LTD_inv_B).inverse();
        }
        B_bar = std::make_shared<LinearOperator::SparseMatrixType>(dense_B_bar.sparseView());
    }
};

/** apply SOR smoother */
void SORSmoother::apply(const Eigen::VectorXd &b, Eigen::VectorXd &x) const
{
    for (int k = 0; k < nsmooth; ++k)
    {
        apply_sparse(b, x);
        // Low rank update (apply only if necessary)
        if (linear_operator->get_m_lowrank() > 0)
        {
            auto BT_x = B.transpose() * x;
            x -= (*B_bar) * BT_x;
        }
    }
}

/** apply SOR smoother to sparse part of the matrix*/
void SORSmoother::apply_sparse(const Eigen::VectorXd &b, Eigen::VectorXd &x) const
{
    const LinearOperator::SparseMatrixType &A_sparse = linear_operator->get_sparse();
    const auto row_ptr = A_sparse.outerIndexPtr();
    const auto col_ptr = A_sparse.innerIndexPtr();
    const auto val_ptr = A_sparse.valuePtr();
    const auto diag_ptr = A_sparse.diagonal();
    unsigned int nrow = A_sparse.rows();
    for (int k = 0; k < nsmooth; ++k)
    {
        for (unsigned int ell_ = 0; ell_ < nrow; ++ell_)
        {
            unsigned int ell = (direction == forward) ? ell_ : nrow - 1 - ell_;
            double residual = 0.0;
            for (int k = row_ptr[ell]; k < row_ptr[ell + 1]; ++k)
            {
                unsigned int ell_prime = col_ptr[k];
                residual += val_ptr[k] * x[ell_prime];
            }
            x[ell] += omega * (b[ell] - residual) / diag_ptr[ell];
        }
    }
}
