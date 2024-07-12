#include <iostream>
#include <chrono>
#include <ctime>
#include <random>
#include <chrono>
#include <cmath>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/QR>

#include "config.h"
#include "lattice/lattice2d.hh"
#include "lattice/lattice3d.hh"
#include "smoother/ssor_smoother.hh"
#include "linear_operator/linear_operator.hh"
#include "linear_operator/shiftedlaplace_fem_operator.hh"
#include "linear_operator/shiftedlaplace_fd_operator.hh"
#include "linear_operator/squared_shiftedlaplace_fd_operator.hh"
#include "linear_operator/measured_operator.hh"
#include "intergrid/intergrid_operator_linear.hh"
#include "sampler/cholesky_sampler.hh"
#include "sampler/ssor_sampler.hh"
#include "sampler/multigridmc_sampler.hh"
#include "solver/loop_solver.hh"
#include "preconditioner/multigrid_preconditioner.hh"
#include "auxilliary/parameters.hh"
#include "measurements/sampling_time.hh"
#include "measurements/posterior_statistics.hh"
#include "measurements/convergence.hh"
#include "measurements/mean_squared_error.hh"

/* *********************************************************************** *
 *                                M A I N
 * *********************************************************************** */
int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " CONFIGURATIONFILE" << std::endl;
        exit(-1);
    }
    auto t_start = std::chrono::system_clock::now();
    std::time_t start_time = std::chrono::system_clock::to_time_t(t_start);
    std::cout << std::endl;
    std::cout << "+--------------------------------+" << std::endl;
    std::cout << "! Multigrid Monte Carlo sampling !" << std::endl;
    std::cout << "+--------------------------------+" << std::endl;
    std::cout << std::endl;
    std::cout << "Starting run at " << std::ctime(&start_time) << std::endl;
    std::string filename(argv[1]);
    std::cout << "Reading parameters from file \'" << filename << "\'" << std::endl;
    GeneralParameters general_params;
    LatticeParameters lattice_params;
    CholeskyParameters cholesky_params;
    SmootherParameters smoother_params;
    MultigridParameters multigrid_params;
    SamplingParameters sampling_params;
    PriorParameters prior_params;
    ConstantCorrelationLengthModelParameters constantcorrelationlengthmodel_params;
    PeriodicCorrelationLengthModelParameters periodiccorrelationlengthmodel_params;
    MeasurementParameters measurement_params;
    general_params.read_from_file(filename);
    lattice_params.read_from_file(filename);
    cholesky_params.read_from_file(filename);
    smoother_params.read_from_file(filename);
    multigrid_params.read_from_file(filename);
    sampling_params.read_from_file(filename);
    prior_params.read_from_file(filename);
    constantcorrelationlengthmodel_params.read_from_file(filename);
    periodiccorrelationlengthmodel_params.read_from_file(filename);
    measurement_params.read_from_file(filename);

    if (measurement_params.dim != general_params.dim)
    {
        std::cout << "ERROR: dimension of measurement locations differs from problem dimension" << std::endl;
        exit(-1);
    }
#if (defined EIGEN_USE_BLAS && defined EIGEN_USE_LAPACKE)
    std::cout << "Compiled with BLAS/LAPACK support for Eigen." << std::endl
              << std::endl;
#else  // EIGEN_USE_BLAS && EIGEN_USE_LAPACKE
    std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    std::cout << "WARNING: Compiled without BLAS/LAPACK support for Eigen." << std::endl;
    std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    std::cout << std::endl;
#endif // EIGEN_USE_BLAS && EIGEN_USE_LAPACKE

#ifndef NCHOLMOD
    std::cout << "Using sparse Cholesky factorisation from CholMod." << std::endl;
#else  // NCHOLMOD
    std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    std::cout << "WARNING: Falling back on Eigen's SimplicalLLT Cholesky factorisation." << std::endl;
    std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    std::cout << std::endl;
#endif // NCHOLMOD

    // Construct lattice and linear operator
    std::shared_ptr<Lattice> lattice;
    if (general_params.dim == 2)
    {
        lattice = std::make_shared<Lattice2d>(lattice_params.nx,
                                              lattice_params.ny);
    }
    else if (general_params.dim == 3)
    {
        lattice = std::make_shared<Lattice3d>(lattice_params.nx,
                                              lattice_params.ny,
                                              lattice_params.nz);
    }
    else
    {
        std::cout << "ERROR: Invalid dimension : " << general_params.dim << std::endl;
        exit(-1);
    }
    std::shared_ptr<CorrelationLengthModel> correlationlengthmodel;
    if (prior_params.correlationlength_model == "constant")
    {
        correlationlengthmodel = std::make_shared<ConstantCorrelationLengthModel>(constantcorrelationlengthmodel_params);
    }
    else if (prior_params.correlationlength_model == "periodic")
    {
        correlationlengthmodel = std::make_shared<PeriodicCorrelationLengthModel>(periodiccorrelationlengthmodel_params);
    }
    else
    {
        std::cout << "Error: invalid correlationlengthmodel \'" << prior_params.correlationlength_model << "\'" << std::endl;
        exit(-1);
    }
    std::shared_ptr<LinearOperator> prior_operator;
    if (prior_params.pde_model == "shiftedlaplace_fem")
    {
        prior_operator = std::make_shared<ShiftedLaplaceFEMOperator>(lattice, correlationlengthmodel, 1);
    }
    else if (prior_params.pde_model == "shiftedlaplace_fd")
    {
        prior_operator = std::make_shared<ShiftedLaplaceFDOperator>(lattice, correlationlengthmodel, 1);
    }
    else if (prior_params.pde_model == "squared_shiftedlaplace_fd")
    {
        prior_operator = std::make_shared<SquaredShiftedLaplaceFDOperator>(lattice, correlationlengthmodel, 1);
    }
    else
    {
        std::cout << "Error: invalid prior \'" << prior_params.pde_model << "\'" << std::endl;
        exit(-1);
    }
    std::shared_ptr<MeasuredOperator> posterior_operator = std::make_shared<MeasuredOperator>(prior_operator,
                                                                                              measurement_params);
    std::shared_ptr<LinearOperator> linear_operator;
    if (general_params.operator_name == "prior")
    {
        linear_operator = prior_operator;
    }
    else if (general_params.operator_name == "posterior")
    {
        linear_operator = posterior_operator;
    }
    else
    {
        std::cout << "ERROR: invalid operator : " << general_params.operator_name << std::endl;
        exit(-1);
    }
    // Construct samplers
    std::shared_ptr<CLCGenerator> rng = std::make_shared<CLCGenerator>();
    std::shared_ptr<Sampler> multigridmc_sampler = std::make_shared<MultigridMCSampler>(linear_operator,
                                                                                        rng,
                                                                                        multigrid_params,
                                                                                        cholesky_params);
    std::shared_ptr<Sampler> ssor_sampler = std::make_shared<SSORSampler>(linear_operator,
                                                                          rng,
                                                                          smoother_params.omega,
                                                                          smoother_params.nsmooth);
    std::shared_ptr<Sampler> cholesky_sampler;
    if (general_params.do_cholesky)
    {
        auto t_start = std::chrono::high_resolution_clock::now();
        if (cholesky_params.factorisation == SparseFactorisation)
        {
            cholesky_sampler = std::make_shared<SparseCholeskySampler>(linear_operator, rng, true);
        }
        else if (cholesky_params.factorisation == LowRankFactorisation)
        {
            cholesky_sampler = std::make_shared<LowRankCholeskySampler>(linear_operator, rng, true);
        }
        else if (cholesky_params.factorisation == DenseFactorisation)
        {
            cholesky_sampler = std::make_shared<DenseCholeskySampler>(linear_operator, rng);
        }
        auto t_finish = std::chrono::high_resolution_clock::now();
        double t_elapsed = std::chrono::duration_cast<std::chrono::seconds>(t_finish - t_start).count();
        std::cout << std::endl;
        std::cout << "time for Cholesky factorisation = " << t_elapsed << " s" << std::endl;
    }

    std::cout << std::endl;
    // Run sampling experiments
    /* ********************** Cholesky ********************** */
    if (general_params.do_cholesky)
    {
        std::cout << "**** Cholesky ****" << std::endl;
        if (general_params.measure_time_per_sample)
        {
            measure_sampling_time(cholesky_sampler,
                                  sampling_params,
                                  measurement_params,
                                  "Cholesky",
                                  "timeseries_cholesky.txt");
            std::cout << std::endl;
        }
        if (general_params.measure_convergence)
        {
            measure_convergence(cholesky_sampler,
                                sampling_params,
                                measurement_params,
                                "convergence_cholesky.txt");
            std::cout << std::endl;
        }
        if (general_params.measure_mse)
        {
            measure_mean_squared_error(cholesky_sampler,
                                       sampling_params,
                                       measurement_params,
                                       "meansquarederror_cholesky.txt");
            std::cout << std::endl;
        }
    }
    /* ********************** SSOR ********************** */
    if (general_params.do_ssor)
    {
        std::cout << "**** SSOR ****" << std::endl;
        if (general_params.measure_time_per_sample)
        {
            measure_sampling_time(ssor_sampler,
                                  sampling_params,
                                  measurement_params,
                                  "SSOR",
                                  "timeseries_ssor.txt");
            std::cout << std::endl;
        }
        if (general_params.measure_convergence)
        {
            measure_convergence(ssor_sampler,
                                sampling_params,
                                measurement_params,
                                "convergence_ssor.txt");

            std::cout << std::endl;
        }
        if (general_params.measure_mse)
        {
            measure_mean_squared_error(ssor_sampler,
                                       sampling_params,
                                       measurement_params,
                                       "meansquarederror_ssor.txt");
            std::cout << std::endl;
        }
    }
    /* ********************** Multigrid ********************** */
    if (general_params.do_multigridmc)
    {
        std::cout << "**** Multigrid MC ****" << std::endl;
        if (general_params.measure_time_per_sample)
        {
            measure_sampling_time(multigridmc_sampler,
                                  sampling_params,
                                  measurement_params,
                                  "MGMC",
                                  "timeseries_multigridmc.txt");
            std::cout << std::endl;
        }
        if (general_params.measure_convergence)
        {
            measure_convergence(multigridmc_sampler,
                                sampling_params,
                                measurement_params,
                                "convergence_multigridmc.txt");
            std::cout << std::endl;
        }
        if (general_params.save_posterior_statistics)
        {
            posterior_statistics(multigridmc_sampler,
                                 sampling_params,
                                 measurement_params);
            std::cout << std::endl;
        }
        if (general_params.measure_mse)
        {
            measure_mean_squared_error(multigridmc_sampler,
                                       sampling_params,
                                       measurement_params,
                                       "meansquarederror_multigridmc.txt");
            std::cout << std::endl;
        }
        std::dynamic_pointer_cast<MultigridMCSampler>(multigridmc_sampler)->show_acceptance_probabilities();
        std::cout << std::endl;
    }
    // print out total timing information
    auto t_finish = std::chrono::system_clock::now();
    std::time_t finish_time = std::chrono::system_clock::to_time_t(t_finish);
    std::chrono::duration<double> t_diff = t_finish - t_start;
    unsigned int elapsed_seconds = t_diff.count();
    int seconds = elapsed_seconds % 60;
    elapsed_seconds /= 60;
    int minutes = elapsed_seconds % 60;
    elapsed_seconds /= 60;
    int hours = elapsed_seconds % 60;
    std::cout << "Completed run at " << std::ctime(&finish_time);
    printf("Total elapsed time = %3d h %2d m %2d s\n", hours, minutes, seconds);
    std::cout << std::endl;
}
