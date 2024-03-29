#ifndef MEASURED_OPERATOR_HH
#define MEASURED_OPERATOR_HH MEASURED_OPERATOR_HH

#include <vector>
#include <Eigen/Dense>
#include "auxilliary/parameters.hh"
#include "auxilliary/quadrature.hh"
#include "lattice/lattice.hh"
#include "linear_operator.hh"

/** @file measured_operator.hh
 *
 * @brief Contains class for measured operator
 */

/** @brief linear operator with measurements
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
class MeasuredOperator : public LinearOperator
{
public:
    /** @brief Create a new instance
     *
     * Populates matrix entries across the grid
     *
     * @param[in] base_operator_ underlying linear operator
     * @param[in] params_ measurement parameters
     */
    MeasuredOperator(const std::shared_ptr<LinearOperator> base_operator_,
                     const MeasurementParameters params_);

    /** @brief Create measurement vector in dual space
     *
     * The entries of the vector are given by
     *
     *    w_j = int_{\Omega} f(x) phi_j(x)
     *
     * where phi_j is the j-th finite element basis function and f(x) is a function given by
     *
     *    f(x) = f_{meas}(|x-x_0|/R) if |x-x_0| < R and f(x) = 0 otherwise
     *
     * where f_{meas}(xi) is some square-integrable function defined in the interval [0,1]
     *
     * @param[in] x0 centre x_0 of measurement
     * @param[in] radius radius R of measurement
     */
    Eigen::SparseVector<double> measurement_vector(const Eigen::VectorXd x0, const double radius) const;

protected:
    /** @brief measurement function
     *
     * @param[in] xi point xi in [0,1] at which the function is to be evaluated
     */
    inline double f_meas(const double xi) const { return 1.0; }

    /** @brief volume of the R-sphere in d dimensions
     *
     * Uses the recursive definition for the volume V_d(R)
     * 
     * V_0(R) = 1, V_1(R) = 2*R, V_d(R) = 2 pi/d *R^2 * V_{d-2}(R)
     * 
     * @param[in] radius radius R
     * @param[in] dim dimension d
     */
    double V_sphere(const double radius, const unsigned int dim) const;

    /** @brief Measurement parameters */
    MeasurementParameters params;
    /** @brief underlying linear operator */
    std::shared_ptr<LinearOperator> base_operator;
};

#endif // MEASURED_OPERATOR_HH