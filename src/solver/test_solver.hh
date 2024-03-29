#ifndef TEST_SOLVER_HH
#define TEST_SOLVER_HH TEST_SOLVER_HH

#include <gtest/gtest.h>
#include <random>
#include <Eigen/Dense>
#include <Eigen/QR>
#include "lattice/lattice2d.hh"
#include "smoother/ssor_smoother.hh"
#include "intergrid/intergrid_operator_linear.hh"
#include "preconditioner/preconditioner.hh"
#include "preconditioner/multigrid_preconditioner.hh"
#include "solver/cholesky_solver.hh"
#include "solver/loop_solver.hh"
#include "linear_operator/shiftedlaplace_fem_operator.hh"
#include "linear_operator/measured_operator.hh"

/** @brief fixture class for solver tests */
class SolverTest : public ::testing::Test
{
protected:
    /* @brief initialise tests */
    void SetUp() override
    {
        unsigned int nx = 256;
        unsigned int ny = 256;

        unsigned int seed = 1212417;
        std::mt19937 rng(seed);
        std::normal_distribution<double> normal_dist(0.0, 1.0);
        std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

        std::shared_ptr<Lattice2d> lattice = std::make_shared<Lattice2d>(nx, ny);
        unsigned int ndof = lattice->Nvertex;

        PeriodicCorrelationLengthModelParameters correlationlengthmodel_params;
        correlationlengthmodel_params.Lambda_min = 0.12;
        correlationlengthmodel_params.Lambda_max = 0.23;
        std::shared_ptr<CorrelationLengthModel> correlationlengthmodel = std::make_shared<PeriodicCorrelationLengthModel>(correlationlengthmodel_params);

        linear_operator = std::make_shared<ShiftedLaplaceFEMOperator>(lattice,
                                                                      correlationlengthmodel);
        unsigned int n_meas = 10;
        std::vector<Eigen::VectorXd> measurement_locations(n_meas);
        Eigen::VectorXd Sigma_diag(n_meas, n_meas);
        for (int k = 0; k < n_meas; ++k)
        {
            measurement_locations[k] = Eigen::Vector2d({uniform_dist(rng), uniform_dist(rng)});
            Sigma_diag(k) = (1.0 + 2.0 * uniform_dist(rng));
        }
        std::shared_ptr<ShiftedLaplaceFEMOperator> prior_operator = std::make_shared<ShiftedLaplaceFEMOperator>(lattice,
                                                                                                                correlationlengthmodel);
        MeasurementParameters measurement_params;
        measurement_params.n = n_meas;
        measurement_params.measurement_locations = measurement_locations;
        measurement_params.variance = Sigma_diag;
        measurement_params.variance_scaling = 1.E-6;
        measurement_params.radius = 0.05;
        measurement_params.measure_global = false;
        measurement_params.variance_global = 0.0;
        measurement_params.mean_global = 0.0;
        linear_operator_lowrank = std::make_shared<MeasuredOperator>(prior_operator,
                                                                     measurement_params);
        // Create states
        x_exact = Eigen::VectorXd(ndof);
        x = Eigen::VectorXd(ndof);
        for (unsigned int ell = 0; ell < ndof; ++ell)
        {
            x_exact[ell] = normal_dist(rng);
        }

        b = Eigen::VectorXd(ndof);
        linear_operator->apply(x_exact, b);
        b_lowrank = Eigen::VectorXd(ndof);
        linear_operator_lowrank->apply(x_exact, b_lowrank);
    }

protected:
    /** @brief linear operator */
    std::shared_ptr<ShiftedLaplaceFEMOperator> linear_operator;
    /** @brief linear operator */
    std::shared_ptr<MeasuredOperator> linear_operator_lowrank;
    /** @brief exact solution */
    Eigen::VectorXd x_exact;
    /** @brief numerical solution */
    Eigen::VectorXd x;
    /** @brief right hand side */
    Eigen::VectorXd b;
    /** @brief right hand side */
    Eigen::VectorXd b_lowrank;
};

/* Test Cholesky solver
 *
 * Computes b = A.x_{exact} for the diffusion operator and then solves A.x = b for x.
 * We expect that ||x-x_{exact}|| is close to zero.
 */
TEST_F(SolverTest, TestCholesky)
{

    CholeskySolver solver(linear_operator_lowrank);
    solver.apply(b_lowrank, x);
    double error = (x - x_exact).norm() / x_exact.norm();
    double tolerance = 1.E-11;
    EXPECT_NEAR(error, 0.0, tolerance);
}

/* Test Multigrid solver for standard diffusion operator
 *
 * Computes b = A.x_{exact} for the diffusion operator and then solves A.x = b for x.
 * We expect that ||x-x_{exact}||/||x-x_{exact}|| is close to zero.
 */
TEST_F(SolverTest, TestMultigrid)
{
    MultigridParameters multigrid_params;
    multigrid_params.nlevel = 5;
    multigrid_params.smoother = "SSOR";
    multigrid_params.coarse_solver = "Cholesky";
    multigrid_params.npresmooth = 1;
    multigrid_params.npostsmooth = 1;
    multigrid_params.ncoarsesmooth = 1;
    multigrid_params.omega = 1.0;
    multigrid_params.coarse_scaling = 1.0;
    multigrid_params.cycle = 1;
    multigrid_params.verbose = 0;
    std::shared_ptr<MultigridPreconditioner> prec = std::make_shared<MultigridPreconditioner>(linear_operator,
                                                                                              multigrid_params);
    IterativeSolverParameters solver_params;
    solver_params.rtol = 1.0E-13;
    solver_params.atol = 1.0E-12;
    solver_params.maxiter = 100;
    solver_params.verbose = 0;
    LoopSolver solver(linear_operator, prec, solver_params);
    solver.apply(b, x);
    double tolerance = 1.E-10;
    double error = (x - x_exact).norm() / x_exact.norm();
    EXPECT_NEAR(error, 0.0, tolerance);
}

/* Test Multigrid solver
 *
 * Computes b = A.x_{exact} for the measured diffusion operator and then solves A.x = b for x.
 * We expect that ||x-x_{exact}||/||x-x_{exact}|| is close to zero.
 */
TEST_F(SolverTest, TestMultigridLowRank)
{
    MultigridParameters multigrid_params;
    multigrid_params.nlevel = 5;
    multigrid_params.smoother = "SSOR";
    multigrid_params.coarse_solver = "Cholesky";
    multigrid_params.npresmooth = 1;
    multigrid_params.npostsmooth = 1;
    multigrid_params.ncoarsesmooth = 1;
    multigrid_params.omega = 1.0;
    multigrid_params.coarse_scaling = 1.0;
    multigrid_params.cycle = 1;
    multigrid_params.verbose = 0;
    std::shared_ptr<MultigridPreconditioner> prec = std::make_shared<MultigridPreconditioner>(linear_operator_lowrank,
                                                                                              multigrid_params);
    IterativeSolverParameters solver_params;
    solver_params.rtol = 1.0E-13;
    solver_params.atol = 1.0E-11;
    solver_params.maxiter = 100;
    solver_params.verbose = 0;
    LoopSolver solver(linear_operator_lowrank, prec, solver_params);
    solver.apply(b_lowrank, x);
    double tolerance = 1.E-10;
    double error = (x - x_exact).norm() / x_exact.norm();
    EXPECT_NEAR(error, 0.0, tolerance);
}

#endif // TEST_SOLVER_HH
