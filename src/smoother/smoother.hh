#ifndef SMOOTHER_HH
#define SMOOTHER_HH SMOOTHER_HH
#include <random>
#include <Eigen/Dense>
#include "linear_operator/linear_operator.hh"

/** @file Smoother.hh
 *
 * @brief Smoothers which can be used in multigrid algorithms
 */

/** @class Smoother
 *
 * @brief Smoother base class */
class Smoother
{
public:
    /** @brief Create a new instance
     *
     * @param[in] linear_operator_ underlying linear operator
     */
    Smoother(const std::shared_ptr<LinearOperator> linear_operator_) : linear_operator(linear_operator_){};

    /** @brief Apply smoother once
     *
     * @param[in] b right hand side
     * @param[inout] x vector to which the sweep is applied
     */
    virtual void apply(const Eigen::VectorXd &b, Eigen::VectorXd &x) const = 0;

protected:
    /** @brief Underlying Linear operator */
    const std::shared_ptr<LinearOperator> linear_operator;
};

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
     * @param[in] direction_ sweep direction (forward or backward)
     */
    SORSmoother(const std::shared_ptr<LinearOperator> linear_operator_,
                const double omega_,
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
    /** @brief Sweep direction */
    const Direction direction;
    /** @brief the matrix B that arises in the low-rank update of the linear operator */
    LinearOperator::DenseMatrixType B;
    /** @brief the matrix bar(B)_{FW} or bar(B)_{BW} used on the forward/backward sweeps */
    std::shared_ptr<LinearOperator::DenseMatrixType> B_bar;
};

/** @class SSORSmoother
 *
 * @brief Symmetric successive overrelaxation smoother with low rank updates
 *
 * This implements the following iteration
 *
 *   x^{k+1/4} = (L +   1/omega * D)^{-1} (b + (L^T - (1-omega)/omega * D) x^k)
 *   y^{k+1/2} = B^T x^{k+1/4}
 *   x^{k+1/2} = x^{k+1/4} - bar(B)_{FW} y^{k+1/2}
 *   x^{k+3/4} = (L^T + 1/omega * D)^{-1} (b + (L - (1-omega)/omega * D) x^{k+1/2})
 *   y^{k+1}   = B^T x^{k+3/4}
 *   x^{k+1}   = x^{k+3/4} - bar(B)_{BW} y^{k+1}
 *
 * Here we have that
 *
 *   bar(B)_{FW} = (L   + 1/omega * D)^{-1} B ( Sigma + B^T (L   + 1/omega * D)^{-1} B )^{-1}
 *   bar(B)_{BW} = (L^T + 1/omega * D)^{-1} B ( Sigma + B^T (L^T + 1/omega * D)^{-1} B )^{-1}
 */
class SSORSmoother : public Smoother
{
public:
    /** @brief Base type*/
    typedef Smoother Base;
    /** @brief Create a new instance
     *
     * @param[in] linear_operator_ underlying linear operator
     * @param[in] omega_ overrelaxation factor
     */
    SSORSmoother(const std::shared_ptr<LinearOperator> linear_operator_,
                 const double omega_) : Base(linear_operator_),
                                        sor_forward(linear_operator_, omega_, forward),
                                        sor_backward(linear_operator_, omega_, backward){};

    /** @brief Carry out a single SOR-sweep
     *
     * @param[in] b right hand side
     * @param[inout] x vector to which the sweep is applied
     */
    virtual void apply(const Eigen::VectorXd &b, Eigen::VectorXd &x) const;

protected:
    /** @brief Forward smoother */
    const SORSmoother sor_forward;
    /** @brief Backward smoother */
    const SORSmoother sor_backward;
};

/* ******************** factory classes ****************************** */

/** @brief Smoother factory base class */
class SmootherFactory
{
public:
    /** @brief extract a smoother for a given action */
    virtual std::shared_ptr<Smoother> get(std::shared_ptr<LinearOperator> linear_operator) = 0;
};

/** @brief SOR smoother factor class */
class SORSmootherFactory : public SmootherFactory
{
public:
    /** @brief Create new instance
     *
     * @param[in] omega_ overrelaxation parameter
     * @param[in] direction_ sweep direction (forward or backward)
     */
    SORSmootherFactory(const double omega_,
                       const Direction direction_) : omega(omega_), direction(direction_) {}

    /** @brief Destructor */
    virtual ~SORSmootherFactory() {}

    /** @brief Return sampler for a specific  linear operator
     *
     * @param[in] linear_operator_ Underlying linear operator
     */
    virtual std::shared_ptr<Smoother> get(std::shared_ptr<LinearOperator> linear_operator)
    {
        return std::make_shared<SORSmoother>(linear_operator, omega, direction);
    }

private:
    /** @brief overrelaxation parameter */
    const double omega;
    /** @brief sweep direction */
    const Direction direction;
};

/** @brief SSOR smoother factor class */
class SSORSmootherFactory : public SmootherFactory
{
public:
    /** @brief Create new instance
     *
     * @param[in] omega_ overrelaxation parameter
     */
    SSORSmootherFactory(const double omega_) : omega(omega_) {}

    /** @brief Destructor */
    virtual ~SSORSmootherFactory() {}

    /** @brief Return sampler for a specific  linear operator
     *
     * @param[in] linear_operator_ Underlying linear operator
     */
    virtual std::shared_ptr<Smoother> get(std::shared_ptr<LinearOperator> linear_operator)
    {
        return std::make_shared<SSORSmoother>(linear_operator, omega);
    }

private:
    /** @brief overrelaxation parameter */
    const double omega;
};

#endif // SMOOTHER_HH