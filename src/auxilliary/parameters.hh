#ifndef PARAMETERS_HH
#define PARAMETERS_HH PARAMETERS_HH
#include <vector>
#include <iostream>
#include <string>
#include <typeinfo>
#include <Eigen/Dense>
#include <libconfig.h++>

/** @file parameters.hh
 *
 * @brief read parameters from configuration files
 * */

/** @brief Base class for parameters */
class Parameters
{
public:
    /** @brief read parameters from disk
     *
     * @param[in] filename name of configuration file to read parameters from
     */
    int read_from_file(const std::string filename);

    /** @brief constructor */
    Parameters() = default;

    /** @brief destructor */
    virtual ~Parameters() = default;

protected:
    /** @brief parse configuration
     *
     * This virtual method has to be implemented by each derived class
     * to actually parse the parameters in the libconfig configuration object.
     *
     * @param[in] root root of configuration object
     */
    virtual void parse_config(const libconfig::Setting &root) = 0;
};

/** @brief structure for general parameters*/
class GeneralParameters : public Parameters
{
public:
    /** @brief parse configuration
     *
     * @param[in] root root of configuration object
     */
    virtual void parse_config(const libconfig::Setting &root);

    /** @brief spatial dimension */
    int dim;
    /** @brief Run the Cholesky sampler? */
    bool do_cholesky;
    /** @brief Run the SSOR sampler? */
    bool do_ssor;
    /** @brief Run the MultigridMC sampler? */
    bool do_multigridmc;
    /** @brief measure time per sample? */
    bool measure_time_per_sample;
    /** @brief save posterior statistics to disk? */
    bool save_posterior_statistics;
    /** @brief measure convergence for the SSOR and MGMC samplers? */
    bool measure_convergence;
    /** @brief measure mean square error? */
    bool measure_mse;
    /** @brief operator to use for sampling (prior or posterior)*/
    std::string operator_name;
};

/** @brief structure for lattice parameters */
class LatticeParameters : public Parameters
{
public:
    /** @brief parse configuration
     *
     * @param[in] root root of configuration object
     */
    virtual void parse_config(const libconfig::Setting &root);

    /** @brief extent in x-direction */
    unsigned int nx;
    /** @brief extent in y-direction */
    unsigned int ny;
    /** @brief extent in z-direction */
    unsigned int nz;
};

/** @brief type of Cholesky factorisation */
enum cholesky_t
{
    SparseFactorisation = 0, // Sparse factorisation
    DenseFactorisation = 1,  // Dense factorisation
    LowRankFactorisation = 2 // Sparse, low rank factorisation
};

/** @struct Cholesky factorisation parameters */
class CholeskyParameters : public Parameters
{
public:
    /** @brief parse configuration
     *
     * @param[in] root root of configuration object
     */
    virtual void parse_config(const libconfig::Setting &root);

    /** @brief Cholesky factorisation to use */
    cholesky_t factorisation;
};

/** @brief structure for smoother parameters */
class SmootherParameters : public Parameters
{
public:
    /** @brief parse configuration
     *
     * @param[in] root root of configuration object
     */
    virtual void parse_config(const libconfig::Setting &root);

    /** @brief overrelaxation factor */
    double omega;
    /** @brief number of smoothing steps */
    unsigned int nsmooth;
};

/** @brief iterative linear solver parameters
 */
class IterativeSolverParameters : public Parameters
{
public:
    /** @brief parse configuration
     *
     * @param[in] root root of configuration object
     */
    virtual void parse_config(const libconfig::Setting &root);

    /** @brief relative tolerance for solving */
    double rtol;
    /** @brief absolute tolerance for solving */
    double atol;
    /** @brief maximum number of iterations */
    unsigned int maxiter;
    /** @brief verbosity level */
    int verbose;
};

/** @struct multigrid parameters */
class MultigridParameters : public Parameters
{
public:
    /** @brief parse configuration
     *
     * @param[in] root root of configuration object
     */
    virtual void parse_config(const libconfig::Setting &root);

    /** @brief Number of levels */
    unsigned int nlevel;
    /** @brief smoother */
    std::string smoother;
    /** @brief coarse solver */
    std::string coarse_solver;
    /** @brief Number of presmoothing steps */
    unsigned int npresmooth;
    /** @brief number of postsmoothing steps */
    unsigned int npostsmooth;
    /** @brief number of coarse smoothing steps (if using SSOR)*/
    unsigned int ncoarsesmooth;
    /** @brief overrelaxation factor */
    double omega;
    /** @brief cycle type (1 = V-cycle, 2 = W-cycle)*/
    unsigned int cycle;
    /** @brief factor with which to scale to coarse grid correction */
    double coarse_scaling;
    /** @brief multigrid variant, can be "exact" or "fas" */
    std::string variant;
    /** @brief verbosity level */
    int verbose;
};

/** @brief structure for sampling parameters */
class SamplingParameters : public Parameters
{
public:
    /** @brief parse configuration
     *
     * @param[in] root root of configuration object
     */
    virtual void parse_config(const libconfig::Setting &root);

    /** @brief number of samples */
    unsigned int nsamples;
    /** @brief number of warmup samples */
    unsigned int nwarmup;
    /** @brief number of OpenMP threads using in convergence test */
    unsigned int nthreadsconvergence;
    /** @brief number of samples used in convergence test */
    unsigned int nsamplesconvergence;
    /** @brief number of steps used in convergence test */
    unsigned int nstepsconvergence;
    /** @brief number of samples used in mean-squared-error test */
    unsigned int nsamplesmse;
    /** @brief number of steps used in mean-squared-error test */
    unsigned int nstepsmse;
};

/** @brief structure for prior parameters
 */
class PriorParameters : public Parameters
{
public:
    /** @brief parse configuration
     *
     * @param[in] root root of configuration object
     */
    virtual void parse_config(const libconfig::Setting &root);

    /** @brief PDE model */
    std::string pde_model;
    /** @brief name of the correlationlength model to be used */
    std::string correlationlength_model;
};

/** @brief structure for const correlation length parameters
 */
class ConstantCorrelationLengthModelParameters : public Parameters
{
public:
    /** @brief parse configuration
     *
     * @param[in] root root of configuration object
     */
    virtual void parse_config(const libconfig::Setting &root);

    /** @brief constant correlation length Lambda */
    double Lambda;
};

/** @brief structure for periodic correlation length parameters
 */
class PeriodicCorrelationLengthModelParameters : public Parameters
{
public:
    /** @brief parse configuration
     *
     * @param[in] root root of configuration object
     */
    virtual void parse_config(const libconfig::Setting &root);

    /** @brief lower bound on correlation length Lambda */
    double Lambda_min;
    /** @brief upper bound on correlation length Lambda */
    double Lambda_max;
};

/** @brief Structure for measurement parameters */
class MeasurementParameters : public Parameters
{
public:
    /** @brief parse configuration
     *
     * @param[in] root root of configuration object
     */
    virtual void parse_config(const libconfig::Setting &root);

    /** @brief dimension of measurements */
    int dim;
    /** @brief number of measurements */
    unsigned int n;
    /** @brief measurement locations */
    std::vector<Eigen::VectorXd> measurement_locations;
    /** @brief radius of measurement function */
    double radius;
    /** @brief measured averages */
    Eigen::VectorXd mean;
    /** @brief diagonal of covariance matrix of measurements */
    Eigen::VectorXd variance;
    /** @brief scaling factor to be applied to all variances */
    double variance_scaling;
    /** @brief sample location */
    Eigen::VectorXd sample_location;
    /** @brief measure global average of field? */
    bool measure_global;
    /** @brief variance of global average */
    double variance_global;
    /** @brief mean of global average */
    double mean_global;
};

#endif // PARAMETERS_HH
