#ifndef DIFFUSION_OPERATOR_2D_HH
#define DIFFUSION_OPERATOR_2D_HH DIFFUSION_OPERATOR_2D_HH

#include <vector>
#include "linear_operator.hh"
#include "lattice/lattice2d.hh"

/** @file diffusion_operator_2d.hh
 *
 * @brief Contains class for diffusion operator in two dimensions
 */

/** @class DiffusionOperator2d
 *
 * Two dimensional diffusion operator
 *
 * This is a discretisation of the linear operator defined by
 *
 *   -div( K(x,y) grad (u(x,y))) + b(x,y) u(x,y)
 *
 * The diffusion coefficient is assumed to be of the form
 *
 *    K(x,y) = alpha_K + beta_K * sin(2 pi x) * sin(2 pi y)
 *
 * and the zero order term is assumed to be
 *
 *    b(x,y) = alpha_b + beta_b * cos(2 pi x) * cos(2 pi y)
 *
 * for some constants alpha_K, beta_K, alpha_b, beta_b.
 *
 */
class DiffusionOperator2d : public LinearOperator
{
public:
    /** @brief Create a new instance
     *
     * Populates matrix entries across the grid
     *
     * @param[in] lattice_ underlying 2d lattice
     * @param[in] rng_ random number generator
     * @param[in] alpha_K first coefficient in diffusion function
     * @param[in] beta_K second coefficient in diffusion function
     * @param[in] alpha_b first coefficient in zero order term
     * @param[in] beta_b second coefficient in zero order term
     * @param[in] m_lowrank_ the dimension of the low rank correction
     */
    DiffusionOperator2d(const std::shared_ptr<Lattice2d> lattice_,
                        const double alpha_K_,
                        const double beta_K_,
                        const double alpha_b_,
                        const double beta_b_);

    /** @brief Diffusion coefficient
     *
     * Evaluates the diffusion coefficient at a given position (x,y) in the domain
     *
     * @param[in] x position in x-direction
     * @param[in] y position in y-direction
     */
    double K_diff(const double x, const double y) const;

    /** @brief Zero order term
     *
     * Evaluates the zero order term at a given position (x,y) in the domain
     *
     * @param[in] x position in x-direction
     * @param[in] y position in y-direction
     */
    double b_zero(const double x, const double y) const;

protected:
    /** @brief First coefficient in diffusion function */
    const double alpha_K;
    /** @brief Second coefficient in diffusion function */
    const double beta_K;
    /** @brief First coefficient in zero order term */
    const double alpha_b;
    /** @brief Second coefficient in zero order term */
    const double beta_b;
};

/** @brief diffusion operator with measurements
 *
 * Assume that we measured data as Y = B^T X + E, where X is drawn from a prior
 * distribution N(xbar,Q^{-1}) and E is draw from an (independent) multivariate normal
 * distribution N(0,Sigma) with covariance Sigma. The conditional distribution of X given y
 * is then a multivariate normal distribution with mean
 *
 *   x_{X|y} = xbar + Q^{-1} B (Sigma + B^T Q^{-1} B)^{-1} (y - B^T xbar)
 *
 * and precision matrix
 *
 *   Q_{X|y} = Q + B Sigma^{-1} B^T.
 *
 */
class MeasuredDiffusionOperator2d : public LinearOperator
{
public:
    /** @brief Create a new instance
     *
     * Populates matrix entries across the grid
     *
     * @param[in] lattice_ underlying 2d lattice
     * @param[in] rng_ random number generator
     * @param[in] measurement_locations_ coordinates of locations where the field is measured
     * @param[in] Sigma_ covariance matrix of measurements
     * @param[in] ignore_measurement_cross_correlations_ ignore all off-diagonal entries in the
     *            covariance matrix Sigma
     * @param[in] measure_global_ measure the average across the entire domain
     * @param[in] sigma_global_ variance of global average measurement
     * @param[in] alpha_K first coefficient in diffusion function
     * @param[in] beta_K second coefficient in diffusion function
     * @param[in] alpha_b first coefficient in zero order term
     * @param[in] beta_b second coefficient in zero order term
     */
    MeasuredDiffusionOperator2d(const std::shared_ptr<Lattice2d> lattice_,
                                const std::vector<Eigen::Vector2d> measurement_locations_,
                                const Eigen::MatrixXd Sigma_,
                                const bool ignore_measurement_cross_correlations_,
                                const bool measure_average_,
                                const double sigma_global_,
                                const double alpha_K_,
                                const double beta_K_,
                                const double alpha_b_,
                                const double beta_b_);

    /** @brief Compute posterior mean
     *
     * @param[in] xbar prior mean
     * @param[in] y measured values
     */
    Eigen::VectorXd posterior_mean(const Eigen::VectorXd &xbar,
                                   const Eigen::VectorXd &y);
};

#endif // DIFFUSION_OPERATOR_2D_HH