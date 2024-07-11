#ifndef MULTIGRIDMC_SAMPLER_HH
#define MULTIGRIDMC_SAMPLER_HH MULTIGRIDMC_SAMPLER_HH
#include <memory>
#include <vector>
#include <Eigen/Dense>
#include "linear_operator/linear_operator.hh"
#include "intergrid/intergrid_operator.hh"
#include "intergrid/intergrid_operator_linear.hh"
#include "auxilliary/parallel_random.hh"
#include "auxilliary/parameters.hh"
#include "solver/cholesky_solver.hh"
#include "sampler.hh"
#include "sor_sampler.hh"
#include "ssor_sampler.hh"
#include "cholesky_sampler.hh"

/** @file multigridmc_sampler.hh
 *
 * @brief Multigrid Monte Carlo sampler
 */

/** @class MultigridMCSampler
 *
 * @brief Sampler based on Multigrid Monte Carlo
 */
class MultigridMCSampler : public Sampler
{
public:
    /** @brief Create a new instance
     *
     * @param[in] linear_operator_ underlying linear operator
     * @param[in] rng_ random number generator
     * @param[in] params_ multigrid Monte Carlo parameters
     * @param[in] cholesky_params_ Cholesky parameters (for coarse level Cholesky sampler)
     */
    MultigridMCSampler(std::shared_ptr<LinearOperator> linear_operator_,
                       std::shared_ptr<RandomGenerator> rng_,
                       const MultigridParameters params_,
                       const CholeskyParameters cholesky_params_);

    /** @brief deep copy
     *
     * Create a deep copy of object, while using a specified random number generator
     *
     * @param[in] random number generator to use
     */
    virtual std::shared_ptr<Sampler> deep_copy(std::shared_ptr<RandomGenerator> rng)
    {
        std::shared_ptr<LinearOperator> linear_operator_ = linear_operator->deep_copy();
        return std::make_shared<MultigridMCSampler>(linear_operator_,
                                                    rng,
                                                    params,
                                                    cholesky_params);
    };

    /** @brief Draw a new sample
     *
     * @param[in] f right hand side b
     * @param[out] x solution x
     */
    virtual void apply(const Eigen::VectorXd &f, Eigen::VectorXd &x) const;

protected:
    /** @brief Recursive solve on a givel level
     *
     * @param[in] level level on which to solve recursively
     */
    void sample(const unsigned int level) const;

    /** @brief Compute logarithm of probability density, ignoring normalisation
     *
     * the density is pi(x) = N*exp[-1/2*x^T.A.x + f^T.x], this method computes
     *
     *    -1/2*(x-mu)^T.A.(x-mu)
     *
     * where A.mu = f
     * @param[in] linear_operator linear operator A
     * @param[in] f_rhs right hand side vector f
     * @param[in] x current state x
     */
    double log_probability(const std::shared_ptr<LinearOperator> linear_operator,
                           const Eigen::VectorXd &f_rhs,
                           const Eigen::VectorXd &x) const;

    /** @brief parameters */
    const MultigridParameters params;
    /** @brief Cholesky parameters (for coarse sampler )*/
    const CholeskyParameters cholesky_params;
    /** @brief coarse level solver */
    std::shared_ptr<Sampler> coarse_sampler;
    /** @brief linear operators on all levels */
    std::vector<std::shared_ptr<LinearOperator>> linear_operators;
    /** @brief smoothers on all levels */
    std::vector<std::shared_ptr<Sampler>> presamplers;
    /** @brief smoothers on all levels */
    std::vector<std::shared_ptr<Sampler>> postsamplers;
    /** @brief intergrid operators on all levels (except the coarsest) */
    std::vector<std::shared_ptr<IntergridOperator>> intergrid_operators;
    /** @brief Solution on each level */
    mutable std::vector<Eigen::VectorXd> x_ell;
    /** @brief RHS on each level */
    mutable std::vector<Eigen::VectorXd> f_ell;
    /** @brief Residual on each level */
    mutable std::vector<Eigen::VectorXd> r_ell;
};

#endif // MULTIGRIDMC_SAMPLER_HH