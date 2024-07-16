#ifndef SOR_SMOOTHER_HH
#define SOR_SMOOTHER_HH SOR_SMOOTHER_HH
#include <random>
#include <Eigen/Dense>
#include "linear_operator/linear_operator.hh"
#include "smoother.hh"

/** @file sor_smoother.hh
 *
 * @brief Successive overrelaxation smoother
 */

/** @brief Sweep direction */
enum Direction
{
    forward = 1,
    backward = 2
};

/** @class SORSmoother
 *
 * @brief Successive overrelaxation smoother
 *
 * This implements the following iteration
 *
 *   x^{k+1/2} = (L +   1/omega * D)^{-1} (b + (L^T - (1-omega)/omega * D) x^k)
 *   y^{k+1} = B^T x^{k+1/2}
 *   x^{k+1} = x^{k+1/2} - bar(B)_{FW} y^{k+1}
 *
 * for the forward sweep or
 *
 *   x^{k+1/2} = (L^T + 1/omega * D)^{-1} (b + (L - (1-omega)/omega * D) x^k)
 *   y^{k+1}   = B^T x^{k+1/2}
 *   x^{k+1}   = x^{k+1/2} - bar(B)_{BW} y^{k+1}
 *
 * for the backward sweep
 *
 * Here we have that
 *
 *   bar(B)_{FW} = (L   + 1/omega * D)^{-1} B ( Sigma + B^T (L   + 1/omega * D)^{-1} B )^{-1}
 *   bar(B)_{BW} = (L^T + 1/omega * D)^{-1} B ( Sigma + B^T (L^T + 1/omega * D)^{-1} B )^{-1}
 *
 */
class SORSmoother : public Smoother
{
public:
    /** @brief Base type*/
    typedef Smoother Base;
    /** @brief Create a new instance
     *
     * @param[in] linear_operator_ underlying linear operator
     * @param[in] omega_ overrelaxation factor
     * @param[in] nsmooth_ number of smoothing steps
     * @param[in] direction_ sweep direction (forward or backward)
     */
    SORSmoother(const std::shared_ptr<LinearOperator> linear_operator_,
                const double omega_,
                const int nsmooth_,
                const Direction direction_);

    /** @brief Carry out a single SOR-sweep
     *
     * @param[in] b right hand side
     * @param[inout] x vector to which the sweep is applied
     */
    virtual void apply(const Eigen::VectorXd &b, Eigen::VectorXd &x) const;

    /** @brief Carry out a single SOR-sweep on the sparse part of the matrix
     *
     * @param[in] b right hand side
     * @param[inout] x vector to which the sweep is applied
     */
    void apply_sparse(const Eigen::VectorXd &b, Eigen::VectorXd &x) const;

protected:
    /** @brief Overrelaxation factor */
    const double omega;
    /** @brief Number of smoothing steps */
    const int nsmooth;
    /** @brief Sweep direction */
    const Direction direction;
    /** @brief the matrix B that arises in the low-rank update of the linear operator */
    LinearOperator::SparseMatrixType B;
    /** @brief the matrix bar(B)_{FW} or bar(B)_{BW} used on the forward/backward sweeps */
    std::shared_ptr<LinearOperator::SparseMatrixType> B_bar;
};

/* ******************** factory classes ****************************** */

/** @brief SOR smoother factor class */
class SORSmootherFactory : public SmootherFactory
{
public:
    /** @brief Create new instance
     *
     * @param[in] omega_ overrelaxation parameter
     * @param[in] nsmooth_ number of smoothing steps
     * @param[in] direction_ sweep direction (forward or backward)
     */
    SORSmootherFactory(const double omega_,
                       const int nsmooth_,
                       const Direction direction_) : omega(omega_),
                                                     nsmooth(nsmooth_),
                                                     direction(direction_) {}

    /** @brief Destructor */
    virtual ~SORSmootherFactory() {}

    /** @brief Return sampler for a specific  linear operator
     *
     * @param[in] linear_operator_ Underlying linear operator
     */
    virtual std::shared_ptr<Smoother> get(std::shared_ptr<LinearOperator> linear_operator)
    {
        return std::make_shared<SORSmoother>(linear_operator, omega, nsmooth, direction);
    }

private:
    /** @brief overrelaxation parameter */
    const double omega;
    /** @brief number of smoothing steps */
    const int nsmooth;
    /** @brief sweep direction */
    const Direction direction;
};

#endif // SOR_SMOOTHER_HH
