#include "cholesky_solver.hh"
/** @file cholesky_solver.cc
 *
 * @brief Implementation of cholesky_solver.hh
 */

/** Create a new instance */
CholeskySolver::CholeskySolver(std::shared_ptr<LinearOperator> linear_operator_) : LinearSolver(linear_operator_)
{
    // Compute Cholesky factorisation here
    const SparseMatrixType &A = linear_operator->get_sparse();
    solver = std::make_shared<LLTType>(A);
    if (linear_operator->get_m_lowrank() > 0)
    {
        B = linear_operator->get_B();
        LinearOperator::DenseMatrixType B_dense = B.toDense();
        const DenseMatrixType Sigma = linear_operator->get_Sigma().toDenseMatrix();
        DenseMatrixType Ainv_B(B_dense.rows(), B_dense.cols());
        Eigen::VectorXd y(B_dense.rows());
        for (int j = 0; j < B_dense.cols(); ++j)
        {
            solver->solve(B_dense(Eigen::seq(0, B_dense.rows() - 1), j), y);
            Ainv_B(Eigen::seq(0, B_dense.rows() - 1), j) = y;
        }
        B_bar = Ainv_B * (Sigma + B_dense.transpose() * Ainv_B).inverse();
    }
}

/** Solve the linear system Ax = b */
void CholeskySolver::apply(const Eigen::VectorXd &b, Eigen::VectorXd &x)
{
    // Solve the factorised system
    if (linear_operator->get_m_lowrank() > 0)
    {
        Eigen::VectorXd y(b.size());
        solver->solve(b, y);
        Eigen::VectorXd BTy = B.transpose() * y;
        x = y - B_bar * BTy;
    }
    else
    {
        solver->solve(b, x);
    }
}