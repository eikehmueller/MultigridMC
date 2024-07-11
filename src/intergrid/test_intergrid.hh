#ifndef TEST_INTERGRID_HH
#define TEST_INTERGRID_HH TEST_INTERGRID_HH

#include <gtest/gtest.h>
#include <random>
#include <Eigen/Dense>
#include "auxilliary/parameters.hh"
#include "lattice/lattice1d.hh"
#include "lattice/lattice2d.hh"
#include "lattice/lattice3d.hh"
#include "intergrid/intergrid_operator_linear.hh"
#include "linear_operator/correlationlength_model.hh"
#include "linear_operator/shiftedlaplace_fem_operator.hh"

/** @brief fixture class for intergrid tests */
class IntergridTest : public ::testing::Test
{
protected:
    /* @brief initialise tests */
    void SetUp() override
    {
        // Lattice sizes - use anisotropic lattice
        unsigned int n = 8;
        lattice_1d = std::make_shared<Lattice1d>(n);
        coarse_lattice_1d = std::make_shared<Lattice1d>(n / 2);
        unsigned int nx = 8;
        unsigned int ny = 8;
        lattice_2d = std::make_shared<Lattice2d>(nx, ny);
        coarse_lattice_2d = std::make_shared<Lattice2d>(nx / 2, ny / 2);
        unsigned int nz = 8;
        lattice_3d = std::make_shared<Lattice3d>(nx, ny, nz);
        coarse_lattice_3d = std::make_shared<Lattice3d>(nx / 2, ny / 2, nz / 2);
        intergrid_operator_1dlinear = std::make_shared<IntergridOperatorLinear>(lattice_1d);
        intergrid_operator_2dlinear = std::make_shared<IntergridOperatorLinear>(lattice_2d);
        intergrid_operator_3dlinear = std::make_shared<IntergridOperatorLinear>(lattice_3d);
    }

    /** @brief return a sample state
     *
     * @param[in] lattice Underlying lattice
     * @param[in] random Initialise with random numbers?
     */
    Eigen::VectorXd get_state(const std::shared_ptr<Lattice> lattice,
                              const bool random = false)
    {
        unsigned int ndof = lattice->Nvertex;
        unsigned int seed = 1212417;
        std::mt19937 rng(seed);
        std::normal_distribution<double> dist(0.0, 1.0);
        Eigen::VectorXd X(ndof);
        if (random)
        {
            for (unsigned int ell = 0; ell < ndof; ++ell)
            {
                X[ell] = dist(rng);
            }
        }
        else
        {
            X.setZero();
        }
        return X;
    }

    /** @brief underlying 1d lattice */
    std::shared_ptr<Lattice1d> lattice_1d;
    /** @brief underlying 1d coarse lattice */
    std::shared_ptr<Lattice1d> coarse_lattice_1d;
    /** @brief underlying 2d lattice */
    std::shared_ptr<Lattice2d> lattice_2d;
    /** @brief underlying 2d coarse lattice */
    std::shared_ptr<Lattice2d> coarse_lattice_2d;
    /** @brief underlying 3d lattice */
    std::shared_ptr<Lattice3d> lattice_3d;
    /** @brief underlying 3d coarse lattice */
    std::shared_ptr<Lattice3d> coarse_lattice_3d;

    /** @brief intergrid operator for linear interpolation in 1d*/
    std::shared_ptr<IntergridOperatorLinear> intergrid_operator_1dlinear;
    /** @brief intergrid operator for linear interpolation in 2d */
    std::shared_ptr<IntergridOperatorLinear> intergrid_operator_2dlinear;
    /** @brief intergrid operator for linear interpolation in 3d */
    std::shared_ptr<IntergridOperatorLinear> intergrid_operator_3dlinear;
};

/** @brief check that prolongating a field in 1d will return the same result as manually interpolating */
TEST_F(IntergridTest, TestProlong1dLinear)
{
    // initial coarse level state
    Eigen::VectorXd X_coarse = get_state(coarse_lattice_1d, true);
    // prolongated state
    Eigen::VectorXd X_prol = get_state(lattice_1d, false);
    // prolongate state
    intergrid_operator_1dlinear->prolongate_add(1.0, X_coarse, X_prol);
    // Manually interpolate linearly
    Eigen::VectorXd X_linear = get_state(lattice_1d, false);
    Eigen::VectorXi shift_right(1);
    shift_right[0] = +1;
    Eigen::VectorXi shift_left(1);
    shift_left[0] = -1;
    for (int ell_coarse = 0; ell_coarse < coarse_lattice_1d->Nvertex; ++ell_coarse)
    {
        unsigned int ell = coarse_lattice_1d->fine_vertex_idx(ell_coarse);
        X_linear[ell] = X_coarse[ell_coarse];
        X_linear[lattice_1d->shift_vertexidx(ell, shift_left)] += 0.5 * X_coarse[ell_coarse];
        X_linear[lattice_1d->shift_vertexidx(ell, shift_right)] += 0.5 * X_coarse[ell_coarse];
    }
    double tolerance = 1.E-12;
    EXPECT_NEAR((X_prol - X_linear).norm(), 0.0, tolerance);
}

/** @brief check that prolongating a field in 2d will return the same result as manually interpolating */
TEST_F(IntergridTest, TestProlong2dLinear)
{
    // initial coarse level state
    Eigen::VectorXd X_coarse = get_state(coarse_lattice_2d, true);
    // prolongated state
    Eigen::VectorXd X_prol = get_state(lattice_2d, false);
    // prolongate state
    intergrid_operator_2dlinear->prolongate_add(1.0, X_coarse, X_prol);
    // Manually interpolate linearly
    Eigen::VectorXd X_linear = get_state(lattice_2d, false);
    Eigen::Vector2i shift_north = {0, +1};
    Eigen::Vector2i shift_south = {0, -1};
    Eigen::Vector2i shift_east = {+1, 0};
    Eigen::Vector2i shift_west = {-1, 0};
    Eigen::Vector2i shift_north_east = {+1, +1};
    Eigen::Vector2i shift_south_east = {+1, -1};
    Eigen::Vector2i shift_north_west = {-1, +1};
    Eigen::Vector2i shift_south_west = {-1, -1};
    for (int ell_coarse = 0; ell_coarse < coarse_lattice_2d->Nvertex; ++ell_coarse)
    {
        unsigned int ell = coarse_lattice_2d->fine_vertex_idx(ell_coarse);
        X_linear[ell] = X_coarse[ell_coarse];
        X_linear[lattice_2d->shift_vertexidx(ell, shift_north)] += 0.5 * X_coarse[ell_coarse];
        X_linear[lattice_2d->shift_vertexidx(ell, shift_south)] += 0.5 * X_coarse[ell_coarse];
        X_linear[lattice_2d->shift_vertexidx(ell, shift_east)] += 0.5 * X_coarse[ell_coarse];
        X_linear[lattice_2d->shift_vertexidx(ell, shift_west)] += 0.5 * X_coarse[ell_coarse];
        X_linear[lattice_2d->shift_vertexidx(ell, shift_north_east)] += 0.25 * X_coarse[ell_coarse];
        X_linear[lattice_2d->shift_vertexidx(ell, shift_south_east)] += 0.25 * X_coarse[ell_coarse];
        X_linear[lattice_2d->shift_vertexidx(ell, shift_north_west)] += 0.25 * X_coarse[ell_coarse];
        X_linear[lattice_2d->shift_vertexidx(ell, shift_south_west)] += 0.25 * X_coarse[ell_coarse];
    }
    double tolerance = 1.E-12;
    EXPECT_NEAR((X_prol - X_linear).norm(), 0.0, tolerance);
}

/** @brief check that restriction is transpose of prolongation.
 *
 * For this to be true, we need that u^{(c)}.r^{(c)} = u.r
 *
 * where u is obtained from u^{(c)} by prolongation and r^{(c)} is
 * obtained from r by restriction.
 */
TEST_F(IntergridTest, TestProlongRestrict2dLinear)
{
    // coarse level state
    Eigen::VectorXd X_coarse = get_state(coarse_lattice_2d, true);
    // prolongated state
    Eigen::VectorXd X_prol = get_state(lattice_2d, false);
    // fine level residual
    Eigen::VectorXd r_fine = get_state(lattice_2d, true);
    // restricted residual
    Eigen::VectorXd r_restr = get_state(coarse_lattice_2d, false);
    // Prolongate state
    intergrid_operator_2dlinear->prolongate_add(1.0, X_coarse, X_prol);
    // Restrict residual
    intergrid_operator_2dlinear->restrict(r_fine, r_restr);
    double tolerance = 1.E-12;
    EXPECT_NEAR(X_coarse.dot(r_restr) - X_prol.dot(r_fine), 0.0, tolerance);
}

/** @brief check that interpolation is inverse of prolongation.
 *
 * For the linear intergrid operator check that the interpolated field
 *
 *    tilde(I)_{h}^{2h} u = u^{c}
 *
 * where u = I_{2n}^{h} u^{c} is obtained from u^{c} by prolongation.
 *
 */
TEST_F(IntergridTest, TestInterpolation2dLinear)
{
    // coarse level state
    Eigen::VectorXd X_coarse = get_state(coarse_lattice_2d, true);
    // prolongated state
    Eigen::VectorXd X_prol = get_state(lattice_2d, false);
    // fine level residual
    Eigen::VectorXd X_interp = get_state(coarse_lattice_2d, false);
    // Prolongate state
    intergrid_operator_2dlinear->prolongate_add(1.0, X_coarse, X_prol);
    // Restrict residual
    intergrid_operator_2dlinear->interpolate(X_prol, X_interp);
    double tolerance = 1.E-12;
    EXPECT_NEAR((X_coarse - X_interp).norm(), 0.0, tolerance);
}

/** @brief check that coarsening the operator works in 2d
 *
 * Coarsening the shifted Laplace operator with constant coefficients should
 * result in the same operator on the next-coarser level
 */

TEST_F(IntergridTest, TestCoarsenOperator2d)
{
    ConstantCorrelationLengthModelParameters correlationmodel_params;
    correlationmodel_params.Lambda = 1.0;
    std::shared_ptr<CorrelationLengthModel> correlationlength_model = std::make_shared<ConstantCorrelationLengthModel>(correlationmodel_params);
    ShiftedLaplaceFEMOperator linear_operator(lattice_2d, correlationlength_model);
    ShiftedLaplaceFEMOperator coarse_operator(coarse_lattice_2d, correlationlength_model);
    LinearOperator coarsened_operator = linear_operator.coarsen(intergrid_operator_2dlinear);
    const double tolerance = 1.E-12;
    EXPECT_NEAR((coarse_operator.get_sparse() - coarsened_operator.get_sparse()).norm(), 0.0, tolerance);
}

/** @brief check that coarsening the operator works in 3d
 *
 * Coarsening the shifted Laplace operator with constant coefficients should
 * result in the same operator on the next-coarser level
 */

TEST_F(IntergridTest, TestCoarsenOperator3d)
{
    ConstantCorrelationLengthModelParameters correlationmodel_params;
    correlationmodel_params.Lambda = 1.0;
    std::shared_ptr<CorrelationLengthModel> correlationlength_model = std::make_shared<ConstantCorrelationLengthModel>(correlationmodel_params);
    ShiftedLaplaceFEMOperator linear_operator(lattice_3d, correlationlength_model);
    ShiftedLaplaceFEMOperator coarse_operator(coarse_lattice_3d, correlationlength_model);
    LinearOperator coarsened_operator = linear_operator.coarsen(intergrid_operator_3dlinear);
    const double tolerance = 1.E-12;
    EXPECT_NEAR((coarse_operator.get_sparse() - coarsened_operator.get_sparse()).norm(), 0.0, tolerance);
}

#endif // TEST_INTERGRID_HH