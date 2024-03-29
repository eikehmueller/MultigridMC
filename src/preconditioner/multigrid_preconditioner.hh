#ifndef MULTIGRID_PRECONDITIONER_HH
#define MULTIGRID_PRECONDITIONER_HH MULTIGRID_PRECONDITIONER_HH
#include <memory>
#include <vector>
#include <Eigen/Dense>
#include "auxilliary/parameters.hh"
#include "linear_operator/linear_operator.hh"
#include "intergrid/intergrid_operator.hh"
#include "intergrid/intergrid_operator_linear.hh"
#include "solver/linear_solver.hh"
#include "solver/cholesky_solver.hh"
#include "preconditioner.hh"
#include "smoother/smoother.hh"
#include "smoother/sor_smoother.hh"
#include "smoother/ssor_smoother.hh"

/** @file multigrid_preconditioner.hh
 *
 * @brief multigrid preconditioner
 */

/** @class MultigridPreconditioner
 *
 * @brief Preconditioner based on the multigrid algorithm
 */
class MultigridPreconditioner : public Preconditioner
{
public:
    /** @brief Create a new instance
     *
     * @param[in] linear_operator_ underlying linear operator
     * @param[in] params_ multigrid parameters
     */
    MultigridPreconditioner(std::shared_ptr<LinearOperator> linear_operator_,
                            const MultigridParameters params_);

    /** @brief Solve the linear system Ax = b
     *
     * @param[in] b right hand side b
     * @param[out] x solution x
     */
    virtual void apply(const Eigen::VectorXd &b, Eigen::VectorXd &x);

protected:
    /** @brief Recursive solve on a givel level
     *
     * @param[in] level level on which to solve recursively
     */
    void solve(const unsigned int level);

    /** @brief parameters */
    const MultigridParameters params;
    /** @brief coarse level solver */
    std::shared_ptr<LinearSolver> coarse_solver;
    /** @brief linear operators on all levels */
    std::vector<std::shared_ptr<LinearOperator>> linear_operators;
    /** @brief smoothers on all levels */
    std::vector<std::shared_ptr<Smoother>> presmoothers;
    /** @brief smoothers on all levels */
    std::vector<std::shared_ptr<Smoother>> postsmoothers;
    /** @brief intergrid operators on all levels (except the coarsest) */
    std::vector<std::shared_ptr<IntergridOperator>> intergrid_operators;
    /** @brief Solution on each level */
    std::vector<Eigen::VectorXd> x_ell;
    /** @brief RHS on each level */
    std::vector<Eigen::VectorXd> b_ell;
    /** @brief Residual on each level */
    std::vector<Eigen::VectorXd> r_ell;
};

#endif // MULTIGRID_PRECONDITIONER_HH