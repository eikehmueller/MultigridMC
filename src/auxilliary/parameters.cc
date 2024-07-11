#include "parameters.hh"

/** @file parameters.cc
 *
 * @brief implementation of parameters.hh
 */

#include <vector>
#include <iostream>
#include <string>
#include <typeinfo>
#include <Eigen/Dense>
#include <libconfig.h++>

/** @file parameters.cc
 *
 * @brief implementation of parameters.hh
 * */

/* read parameters from disk */
int Parameters::read_from_file(const std::string filename)
{
    std::string classname = typeid(*this).name();
    libconfig::Config cfg;
    // Read the file. If there is an error, report it and exit.
    try
    {
        cfg.readFile(filename.c_str());
    }
    catch (const libconfig::FileIOException &fioex)
    {
        std::cerr << "Error in class \'" << classname << "\': "
                  << "cannot open configuration file \'" << filename << "\'." << std::endl;
        exit(-1);
    }
    const libconfig::Setting &root = cfg.getRoot();
    try
    {
        parse_config(root);
    }

    catch (const libconfig::SettingException &ex)
    {
        std::cerr << "Error in class \'" << classname << "\': "
                  << "cannot read configuration from file \'" << filename << "\'." << std::endl;
        exit(-1);
    }
    return (EXIT_SUCCESS);
}

/* parse general configuration */
void GeneralParameters::parse_config(const libconfig::Setting &root)
{
    const libconfig::Setting &general = root["general"];
    dim = general["dim"];
    do_cholesky = general.lookup("do_cholesky");
    do_ssor = general.lookup("do_ssor");
    do_multigridmc = general.lookup("do_multigridmc");
    measure_time_per_sample = general.lookup("measure_time_per_sample");
    save_posterior_statistics = general.lookup("save_posterior_statistics");
    measure_convergence = general.lookup("measure_convergence");
    measure_mse = general.lookup("measure_mse");
    operator_name = general.lookup("operator").c_str();
    if (not((operator_name == "prior") or (operator_name == "posterior")))
    {
        std::cout << "ERROR: operator has to be \'prior\' or \'posterior\'" << std::endl;
        exit(-1);
    }
    std::cout << "  dimension = " << dim << std::endl;
    std::cout << "  operator = " << operator_name << std::endl;
}

/* parse lattice configuration  */
void LatticeParameters::parse_config(const libconfig::Setting &root)
{
    const libconfig::Setting &lattice = root["lattice"];
    nx = lattice.lookup("nx");
    ny = lattice.lookup("ny");
    nz = lattice.lookup("nz");
    std::cout << "  lattice size = " << nx << " x " << ny << " x " << nz << std::endl;
}

/* parse cholesky configuration */
void CholeskyParameters::parse_config(const libconfig::Setting &root)
{
    const libconfig::Setting &cholesky = root["cholesky"];
    std::string fac_str = cholesky.lookup("factorisation").c_str();
    if (fac_str == "sparse")
    {
        factorisation = SparseFactorisation;
    }
    else if (fac_str == "lowrank")
    {
        factorisation = LowRankFactorisation;
    }
    else if (fac_str == "dense")
    {
        factorisation = DenseFactorisation;
    }
    else
    {
        std::cout << "ERROR: Unknown Cholesky factorisation: \'" << fac_str << "\'" << std::endl;
        exit(-1);
    }
    std::cout << "  Cholesky factorisation = " << fac_str << std::endl;
}

/* parse smoother configuration */
void SmootherParameters::parse_config(const libconfig::Setting &root)
{
    const libconfig::Setting &smoother = root["smoother"];
    omega = smoother.lookup("omega");
    nsmooth = smoother.lookup("nsmooth");
    std::cout << "  smoother/Gibbs sampler " << std::endl;
    std::cout << "    number of smoothing steps = " << nsmooth << std::endl;
    std::cout << "    overrelaxation factor = " << omega << std::endl;
}

/* parse linear solver configuration */
void IterativeSolverParameters::parse_config(const libconfig::Setting &root)
{
    const libconfig::Setting &iterative_solver = root["iterative_solver"];
    rtol = iterative_solver.lookup("rtol");
    atol = iterative_solver.lookup("atol");
    maxiter = iterative_solver.lookup("maxiter");
    verbose = iterative_solver.lookup("verbose");
    std::cout << "  iterative solver " << std::endl;
    std::cout << "    relative tolerance = " << rtol << std::endl;
    std::cout << "    absolute tolerance = " << atol << std::endl;
    std::cout << "    maxiter = " << maxiter << std::endl;
}

/* parse multigrid configuration */
void MultigridParameters::parse_config(const libconfig::Setting &root)
{
    const libconfig::Setting &multigrid = root["multigrid"];
    nlevel = multigrid.lookup("nlevel");
    smoother = multigrid.lookup("smoother").c_str();
    if (not((smoother == "SOR") or (smoother == "SSOR")))
    {
        std::cout << "ERROR: invalid multigrid smoother : \'" << smoother << "\'" << std::endl;
        std::cout << "       must be SOR or SSOR" << std::endl;
        exit(-1);
    }
    coarse_solver = multigrid.lookup("coarse_solver").c_str();
    if (not((coarse_solver == "SSOR") or (coarse_solver == "Cholesky")))
    {
        std::cout << "ERROR: invalid multigrid coarse solver : \'" << coarse_solver << "\'" << std::endl;
        std::cout << "       must be SSOR or Cholesky" << std::endl;
        exit(-1);
    }

    variant = multigrid.lookup("variant").c_str();
    if (not((variant == "exact") or (variant == "fas")))
    {
        std::cout << "ERROR: invalid multigrid variant : \'" << variant << "\'" << std::endl;
        std::cout << "       must be exact or fas" << std::endl;
        exit(-1);
    }

    npresmooth = multigrid.lookup("npresmooth");
    npostsmooth = multigrid.lookup("npostsmooth");
    ncoarsesmooth = multigrid.lookup("ncoarsesmooth");
    cycle = multigrid.lookup("cycle");
    coarse_scaling = multigrid.lookup("coarse_scaling");
    std::string cycle_label = "";
    if (cycle == 1)
    {
        cycle_label = "( V-cycle )";
    }
    else if (cycle == 2)
    {
        cycle_label = "( W-cycle )";
    }
    omega = multigrid.lookup("omega");
    verbose = multigrid.lookup("verbose");
    std::cout << "  multigrid" << std::endl;
    std::cout << "    variant = " << variant << std::endl;
    std::cout << "    levels = " << nlevel << std::endl;
    std::cout << "    npresmooth = " << npresmooth << std::endl;
    std::cout << "    npostsmooth = " << npostsmooth << std::endl;
    std::cout << "    ncoarsesmooth = " << ncoarsesmooth << std::endl;
    std::cout << "    smoother = " << smoother << std::endl;
    std::cout << "    coarse solver = " << coarse_solver << std::endl;
    std::cout << "    overrelaxation factor = " << omega << std::endl;
    std::cout << "    cycle = " << cycle << " " << cycle_label << std::endl;
    std::cout << "    coarse_scaling = " << coarse_scaling << std::endl;
}

/* parse sampling configuration */
void SamplingParameters::parse_config(const libconfig::Setting &root)
{
    const libconfig::Setting &sampling = root["sampling"];
    const libconfig::Setting &timeseries = sampling.lookup("timeseries");
    const libconfig::Setting &convergence = sampling.lookup("convergence");
    const libconfig::Setting &mse = sampling.lookup("mse");
    nsamples = timeseries.lookup("nsamples");
    nwarmup = timeseries.lookup("nwarmup");
    nstepsconvergence = convergence.lookup("nsteps");
    nsamplesconvergence = convergence.lookup("nsamples");
    nthreadsconvergence = convergence.lookup("nthreads");
    nstepsmse = mse.lookup("nsteps");
    nsamplesmse = mse.lookup("nsamples");
    std::cout << "  timeseries: number of samples         = " << nsamples << std::endl;
    std::cout << "  timeseries: number of warmup samples  = " << nwarmup << std::endl;
    std::cout << "  convergence test: OpenMP threads      = " << nthreadsconvergence << std::endl;
    std::cout << "  convergence test: number of steps     = " << nstepsconvergence << std::endl;
    std::cout << "  convergence test: number of samples   = " << nsamplesconvergence << std::endl;
    std::cout << "  MSE test: number of steps             = " << nstepsmse << std::endl;
    std::cout << "  MSE test: number of samples           = " << nsamplesmse << std::endl;
}

/* prior configuration */
void PriorParameters::parse_config(const libconfig::Setting &root)
{

    const libconfig::Setting &prior = root["prior"];
    pde_model = prior.lookup("pdemodel").c_str();
    if (not((pde_model == "shiftedlaplace_fem") or
            (pde_model == "shiftedlaplace_fd") or
            (pde_model == "squared_shiftedlaplace_fd")))
    {
        std::cout << "ERROR: Unknown prior: \'" << pde_model << "\'" << std::endl;
        exit(-1);
    }

    correlationlength_model = prior.lookup("correlationlengthmodel").c_str();
    if (not((correlationlength_model == "constant") or (correlationlength_model == "periodic")))
    {
        std::cout << "ERROR: Invalied correlation length model \'" << correlationlength_model << "\'." << std::endl;
        exit(-1);
    }
    std::cout << "  prior" << std::endl;
    std::cout << "    PDEmodel = " << pde_model << std::endl;
    std::cout << "    correlationlengthmodel = " << correlationlength_model << std::endl;
}

/* constant correlation length model configuration */
void ConstantCorrelationLengthModelParameters::parse_config(const libconfig::Setting &root)
{
    const libconfig::Setting &model = root["constantcorrelationlengthmodel"];
    Lambda = model.lookup("Lambda");
    std::cout << "  constantcorrelationlengthmodel" << std::endl;
    std::cout << "    Lambda = " << Lambda << std::endl;
}

/* periodic correlation length model configuration */
void PeriodicCorrelationLengthModelParameters::parse_config(const libconfig::Setting &root)
{
    const libconfig::Setting &model = root["periodiccorrelationlengthmodel"];
    Lambda_min = model.lookup("Lambda_min");
    Lambda_max = model.lookup("Lambda_max");
    std::cout << "  periodiccorrelationlengthmodel" << std::endl;
    std::cout << "    Lambda_min = " << Lambda_min << std::endl;
    std::cout << "    Lambda_max = " << Lambda_max << std::endl;
    if (not(Lambda_max >= Lambda_min))
    {
        std::cout << "ERROR: upper bound on correlation length has to exceed lower bound." << std::endl;
        exit(-1);
    }
    if (not(Lambda_min > 0))
    {
        std::cout << "ERROR: lower bound on correlation length has to be positive." << std::endl;
        exit(-1);
    }
}

/* parse measurement configuration */
void MeasurementParameters::parse_config(const libconfig::Setting &root)
{
    const libconfig::Setting &measurements = root["measurements"];
    // radius
    radius = measurements.lookup("radius");
    // scaling factor for variance
    variance_scaling = measurements.lookup("variance_scaling");
    // global measurements
    measure_global = measurements.lookup("measure_global");
    variance_global = measurements.lookup("variance_global");
    mean_global = measurements.lookup("mean_global");
    std::string filename = measurements.lookup("filename").c_str();
    // Sample location
    const libconfig::Setting &s_point = measurements.lookup("sample_location");
    Eigen::VectorXd v(s_point.getLength());
    for (int d = 0; d < v.size(); ++d)
    {
        v[d] = double(s_point[d]);
    }
    sample_location = v;

    // Read measurements from separate configuration file
    libconfig::Config measurement_cfg;
    try
    {
        measurement_cfg.readFile(filename.c_str());
    }
    catch (const libconfig::FileIOException &fioex)
    {
        std::cerr << "ERROR opening configuration file with measurements: \'" << filename << "\'." << std::endl;
        exit(-1);
    }
    const libconfig::Setting &measurement_data = measurement_cfg.getRoot();
    try
    {
        // dimension
        dim = measurement_data.lookup("dim");
        // number of measurements
        n = measurement_data.lookup("n");
        // Measurement locations
        const libconfig::Setting &m_points = measurement_data.lookup("measurement_locations");
        measurement_locations.clear();
        for (int j = 0; j < n; ++j)
        {
            Eigen::VectorXd v(dim);
            for (int d = 0; d < dim; ++d)
            {
                v[d] = double(m_points[dim * j + d]);
            }
            measurement_locations.push_back(v);
        }
        // Measured averages
        const libconfig::Setting &s_mean = measurement_data.lookup("mean");
        mean = Eigen::VectorXd(n);
        for (int j = 0; j < n; ++j)
        {
            mean(j) = double(s_mean[j]);
        }
        // Covariance matrix
        const libconfig::Setting &Sigma = measurement_data.lookup("variance");
        variance = Eigen::VectorXd(n);
        for (int j = 0; j < n; ++j)
        {
            variance(j) = Sigma[j];
        }
    }
    catch (const libconfig::SettingException &ex)
    {
        std::cerr << "ERROR reading configuration file with measurements: \'" << filename << "\'." << std::endl;
        exit(-1);
    }

    // print out summary
    std::cout << "  measurements " << std::endl;
    std::cout << "    file with measurements = \'" << filename << "\'" << std::endl;
    std::cout << "    dimension of measurement locations = " << dim << std::endl;
    std::cout << "    number of measurement points = " << n << std::endl;
    std::cout << "    radius of individual measurements = " << radius << std::endl;
    std::cout << "    variance scaling = " << variance_scaling << std::endl;
    std::cout << "    measure global average across domain? ";
    if (measure_global)
    {
        std::cout << "yes" << std::endl;
        std::cout << "    mean of global average = " << mean_global << std::endl;
        std::cout << "    variance of global average = " << variance_global << std::endl;
    }
    else
    {
        std::cout << "no" << std::endl;
    }
    std::cout << std::endl;
}
