#ifndef TEST_SAMPLER_HH
#define TEST_SAMPLER_HH TEST_SAMPLER_HH

#include <utility>
#include <gtest/gtest.h>
#include <random>
#include <Eigen/Dense>
#include <Eigen/QR>
#include "config.h"
#include "auxilliary/parameters.hh"
#include "lattice/lattice1d.hh"
#include "lattice/lattice2d.hh"
#include "linear_operator/correlationlength_model.hh"
#include "linear_operator/linear_operator.hh"
#include "linear_operator/shiftedlaplace_fem_operator.hh"
#include "linear_operator/measured_operator.hh"
#include "intergrid/intergrid_operator_linear.hh"
#include "sampler/sampler.hh"
#include "sampler/cholesky_sampler.hh"
#include "sampler/ssor_sampler.hh"
#include "sampler/multigridmc_sampler.hh"

/** @class TestOperator1d
 *
 * @brief Simple linear operator used for testing
 *
 * The sparse matrix is an n x n matrix given by
 *
 *
 *            [  6 -1                 ]
 *            [ -1  6 -1              ]
 *            [    -1  6 -1           ]
 * A_sparse = [       -1  6 -1        ]
 *            [            ....       ]
 *            [               -1 6 -1 ]
 *            [                 -1  6 ]
 *
 * The low rank update is optional. If it is used, the B that defines it is given by
 *
 * B^T = [ 0 0 0 1 0 0 ... 0 ]
 *       [ 0 0 0 0 1 0 ... 0 ]
 *
 *
 * with Sigma = [4.2  0.0]
 *              [0.0  9.3]
 */
class TestOperator1d : public LinearOperator
{
public:
    /** @brief Create new instance
     *
     * @param[in] lowrank_update_ Use a low-rank update?
     */
    TestOperator1d(const bool lowrank_update_) : LinearOperator(std::make_shared<Lattice1d>(8),
                                                                lowrank_update_ ? 2 : 0),
                                                 lowrank_update(lowrank_update_)
    {
        typedef Eigen::Triplet<double> T;
        std::vector<T> triplet_list;
        const unsigned int nrow = lattice->Nvertex;
        triplet_list.reserve(3 * nrow);
        for (unsigned int i = 0; i < nrow; ++i)
        {
            triplet_list.push_back(T(i, i, +6.0));
            if (i > 0)
            {
                triplet_list.push_back(T(i, (i - 1 + nrow) % nrow, -1.0));
            }
            if (i < nrow - 1)
            {
                triplet_list.push_back(T(i, (i + 1 + nrow) % nrow, -1.0));
            }
        }
        A_sparse.setFromTriplets(triplet_list.begin(), triplet_list.end());
        B = SparseMatrixType(nrow, 2);
        B.insert(3, 0) = 10.0;
        B.insert(4, 1) = 10.0;
        Eigen::VectorXd variance(2);
        variance[0] = 4.2;
        variance[1] = 9.3;
        Sigma_diag.diagonal() = variance;
        Sigma_inv_BT = get_Sigma_inv() * B.transpose();
    }

protected:
    /** @brief use a low-rank update? */
    const bool lowrank_update;
};

/** @brief fixture class for sampler tests */
class SamplerTest : public ::testing::TestWithParam<bool>
{
protected:
    /** @brief initialise tests */
    void SetUp() override
    {
    }

    /** @brief Compare measured covariance to true covariance
     *
     * For a randomly chosen f, apply given sampler repeatedly and measure mean
     * and covariance of samples. Returns a tuple, which contains with following
     * quantities:
     *
     *   first:  difference between Q^{-1} * f and sample_mean in the L-infinity norm
     *   second: difference between exact and measured covariance matrix in the
     *           L-infinity norm
     *
     * @param[in] linear_operator Underlying linear operator
     * @param[in] sampler Sampler to be used
     * @param[in] nsamples number of
     */
    std::pair<double, double> mean_covariance_error(const std::shared_ptr<LinearOperator> linear_operator,
                                                    const std::shared_ptr<Sampler> sampler,
                                                    const unsigned int nsamples)
    {
        unsigned int ndof = linear_operator->get_ndof();
        Eigen::VectorXd f(ndof);
        // Pick an random right hand side f
        unsigned int seed = 1342517;
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> uniform_dist(0, 1);
        Eigen::VectorXd mean_exact(ndof);
        for (unsigned int ell = 0; ell < ndof; ++ell)
        {
            mean_exact[ell] = uniform_dist(rng);
        }
        LinearOperator::DenseMatrixType precision = linear_operator->precision();
        f = precision * mean_exact;
        // Estimators for E(x) and E(x^3)
        Eigen::VectorXd Ex(ndof);
        LinearOperator::DenseMatrixType Exx(ndof, ndof);
        Ex.setZero();
        Exx.setZero();
        unsigned int nsamples_warmup = 1000;
        Eigen::VectorXd x(ndof);
        for (int k = 0; k < nsamples_warmup; ++k)
        {
            sampler->apply(f, x);
        }
        for (int k = 0; k < nsamples; ++k)
        {
            sampler->apply(f, x);
            Ex += 1. / (k + 1) * (x - Ex);
            Exx += 1. / (k + 1) * (x * x.transpose() - Exx);
        }
        // maximum likelihood estimator
        LinearOperator::DenseMatrixType covariance = Exx - Ex * Ex.transpose();
        LinearOperator::DenseMatrixType covariance_exact = linear_operator->covariance();
        double error_mean = (Ex - mean_exact).lpNorm<Eigen::Infinity>();
        double error_covariance = (covariance - covariance_exact).lpNorm<Eigen::Infinity>();
        return std::make_pair(error_mean, error_covariance);
    }
};

INSTANTIATE_TEST_SUITE_P(LowRankCorrection,
                         SamplerTest,
                         testing::Values(false, true));

/* Test sparse Cholesky sampler without / with low rank correction
 *
 * Draw a large number of samples and check that their covariance agrees with
 * the analytical value of the covariance.
 */
TEST_P(SamplerTest, TestSparseCholeskySampler1d)
{
    bool lowrank_correction = GetParam();
    std::shared_ptr<TestOperator1d> linear_operator = std::make_shared<TestOperator1d>(lowrank_correction);
    std::shared_ptr<CLCGenerator> rng = std::make_shared<CLCGenerator>();
    std::shared_ptr<SparseCholeskySampler> sampler = std::make_shared<SparseCholeskySampler>(linear_operator, rng);
    std::pair<double, double> error = mean_covariance_error(linear_operator, sampler, 500000);
    const double tolerance = 2.E-3;
    EXPECT_NEAR(error.first, 0.0, tolerance);
    EXPECT_NEAR(error.second, 0.0, tolerance);
}

/* Test low rank Cholesky sampler without / with low rank correction
 *
 * Draw a large number of samples and check that their covariance agrees with
 * the analytical value of the covariance.
 */
TEST_P(SamplerTest, TestLowRankCholeskySampler1d)
{
    bool lowrank_correction = GetParam();
    std::shared_ptr<TestOperator1d> linear_operator = std::make_shared<TestOperator1d>(lowrank_correction);
    std::shared_ptr<CLCGenerator> rng = std::make_shared<CLCGenerator>();
    std::shared_ptr<LowRankCholeskySampler> sampler = std::make_shared<LowRankCholeskySampler>(linear_operator, rng);
    std::pair<double, double> error = mean_covariance_error(linear_operator, sampler, 500000);
    const double tolerance = 2.E-3;
    EXPECT_NEAR(error.first, 0.0, tolerance);
    EXPECT_NEAR(error.second, 0.0, tolerance);
}

/* Test dense Cholesky sampler without / with low rank correction
 *
 * Draw a large number of samples and check that their covariance agrees with
 * the analytical value of the covariance.
 */
TEST_P(SamplerTest, TestDenseCholeskySampler1d)
{
    bool lowrank_correction = GetParam();
    std::shared_ptr<TestOperator1d> linear_operator = std::make_shared<TestOperator1d>(lowrank_correction);
    std::shared_ptr<CLCGenerator> rng = std::make_shared<CLCGenerator>();
    std::shared_ptr<DenseCholeskySampler> sampler = std::make_shared<DenseCholeskySampler>(linear_operator, rng);
    std::pair<double, double> error = mean_covariance_error(linear_operator, sampler, 500000);
    const double tolerance = 2.E-3;
    EXPECT_NEAR(error.first, 0.0, tolerance);
    EXPECT_NEAR(error.second, 0.0, tolerance);
}

/* Test SSOR sampler without / with low rank correction
 *
 * Draw a large number of samples and check that their covariance agrees with
 * the analytical value of the covariance.
 */
TEST_P(SamplerTest, TestSSORSampler1d)
{
    bool lowrank_correction = GetParam();
    std::shared_ptr<TestOperator1d> linear_operator = std::make_shared<TestOperator1d>(lowrank_correction);
    std::shared_ptr<CLCGenerator> rng = std::make_shared<CLCGenerator>();
    const double omega = 0.8;
    std::shared_ptr<SSORSampler> sampler = std::make_shared<SSORSampler>(linear_operator,
                                                                         rng,
                                                                         omega,
                                                                         1);
    std::pair<double, double> error = mean_covariance_error(linear_operator, sampler, 500000);
    const double tolerance = 2.E-3;
    EXPECT_NEAR(error.first, 0.0, tolerance);
    EXPECT_NEAR(error.second, 0.0, tolerance);
}

/* Test Multigrid MC sampler without / with low rank correction
 *
 * Draw a large number of samples and check that their covariance agrees with
 * the analytical value of the covariance.
 */
TEST_P(SamplerTest, TestMultigridMCSampler1d)
{
    bool lowrank_correction = GetParam();
    MultigridParameters multigrid_params;
    multigrid_params.variant = "exact";
    multigrid_params.nlevel = 3;
    multigrid_params.smoother = "SSOR";
    multigrid_params.coarse_solver = "Cholesky";
    multigrid_params.npresmooth = 1;
    multigrid_params.npostsmooth = 1;
    multigrid_params.ncoarsesmooth = 1;
    multigrid_params.omega = 1.0;
    multigrid_params.cycle = 1;
    multigrid_params.coarse_scaling = 1.0;
    multigrid_params.verbose = 0;
    CholeskyParameters cholesky_params;
    cholesky_params.factorisation = SparseFactorisation;

    std::shared_ptr<TestOperator1d> linear_operator = std::make_shared<TestOperator1d>(lowrank_correction);
    std::shared_ptr<CLCGenerator> rng = std::make_shared<CLCGenerator>();
    std::shared_ptr<MultigridMCSampler> sampler = std::make_shared<MultigridMCSampler>(linear_operator,
                                                                                       rng,
                                                                                       multigrid_params,
                                                                                       cholesky_params);
    std::pair<double, double> error = mean_covariance_error(linear_operator, sampler, 1000000);
    const double tolerance = 2.E-3;
    EXPECT_NEAR(error.first, 0.0, tolerance);
    EXPECT_NEAR(error.second, 0.0, tolerance);
}

/* Test Multigrid MC sampler without / with low rank correction in 2d
 *
 * Draw a large number of samples and check that their covariance agrees with
 * the analytical value of the covariance.
 */
TEST_F(SamplerTest, TestMultigridMCSampler2d)
{
    unsigned int seed = 1212417;
    std::mt19937_64 rng_sigma(seed);
    std::uniform_real_distribution<double> dist_uniform(0.0, 1.0);
    int nx = thorough_testing ? 16 : 8;
    int ny = thorough_testing ? 16 : 8;
    std::shared_ptr<Lattice2d> lattice = std::make_shared<Lattice2d>(nx, ny);
    unsigned int ndof = lattice->Nvertex;
    PeriodicCorrelationLengthModelParameters correlationlengthmodel_params;
    correlationlengthmodel_params.Lambda_min = 1.2;
    correlationlengthmodel_params.Lambda_max = 2.3;
    std::shared_ptr<CorrelationLengthModel> correlationlengthmodel = std::make_shared<PeriodicCorrelationLengthModel>(correlationlengthmodel_params);

    unsigned int n_meas = 4;
    std::vector<Eigen::VectorXd> measurement_locations(n_meas);
    Eigen::VectorXd Sigma_diag(n_meas, n_meas);
    measurement_locations[0] = Eigen::Vector2d({0.25, 0.25});
    measurement_locations[1] = Eigen::Vector2d({0.25, 0.75});
    measurement_locations[2] = Eigen::Vector2d({0.75, 0.25});
    measurement_locations[3] = Eigen::Vector2d({0.75, 0.75});
    for (int k = 0; k < n_meas; ++k)
    {
        Sigma_diag(k) = (1.0 + 2.0 * dist_uniform(rng_sigma));
    }
    std::shared_ptr<CLCGenerator> rng = std::make_shared<CLCGenerator>();
    std::shared_ptr<ShiftedLaplaceFEMOperator> prior_operator = std::make_shared<ShiftedLaplaceFEMOperator>(lattice,
                                                                                                            correlationlengthmodel);
    MeasurementParameters measurement_params;
    measurement_params.n = n_meas;
    measurement_params.measurement_locations = measurement_locations;
    measurement_params.variance = Sigma_diag;
    measurement_params.variance_scaling = 1.E-4;
    measurement_params.radius = 0.05;
    measurement_params.measure_global = false;
    measurement_params.variance_global = 0.0;
    measurement_params.mean_global = 0.0;
    std::shared_ptr<MeasuredOperator> linear_operator = std::make_shared<MeasuredOperator>(prior_operator,
                                                                                           measurement_params);

    MultigridParameters multigrid_params;
    multigrid_params.variant = "exact";
    multigrid_params.nlevel = 3;
    multigrid_params.smoother = "SSOR";
    multigrid_params.coarse_solver = "Cholesky";
    multigrid_params.npresmooth = 1;
    multigrid_params.npostsmooth = 1;
    multigrid_params.ncoarsesmooth = 1;
    multigrid_params.omega = 1.0;
    multigrid_params.cycle = 1;
    multigrid_params.coarse_scaling = 1.0;
    multigrid_params.verbose = 0;
    CholeskyParameters cholesky_params;
    cholesky_params.factorisation = SparseFactorisation;

    std::shared_ptr<Sampler> sampler = std::make_shared<MultigridMCSampler>(linear_operator,
                                                                            rng,
                                                                            multigrid_params,
                                                                            cholesky_params);
    const unsigned int nsamples = thorough_testing ? 2000000 : 10000;
    std::pair<double, double> error = mean_covariance_error(linear_operator, sampler, nsamples);
    const double tolerance = thorough_testing ? 2.2E-3 : 2.E-2;
    EXPECT_NEAR(error.first, 0.0, tolerance);
    EXPECT_NEAR(error.second, 0.0, tolerance);
}

#endif // TEST_SAMPLER_HH