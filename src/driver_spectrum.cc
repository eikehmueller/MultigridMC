#include <iostream>
#include <algorithm>
#include <vector>
#include <fstream>
#include <Eigen/Eigenvalues>

#include "config.h"
#include "lattice/lattice2d.hh"
#include "linear_operator/linear_operator.hh"
#include "linear_operator/shiftedlaplace_fem_operator.hh"
#include "linear_operator/measured_operator.hh"
#include "auxilliary/parameters.hh"

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
    std::string filename(argv[1]);
    std::cout << "Reading parameters from file \'" << filename << "\'" << std::endl;
    LatticeParameters lattice_params;
    PriorParameters prior_params;
    ConstantCorrelationLengthModelParameters constantcorrelationlengthmodel_params;
    PeriodicCorrelationLengthModelParameters periodiccorrelationlengthmodel_params;
    MeasurementParameters measurement_params;
    lattice_params.read_from_file(filename);
    prior_params.read_from_file(filename);
    constantcorrelationlengthmodel_params.read_from_file(filename);
    periodiccorrelationlengthmodel_params.read_from_file(filename);
    measurement_params.read_from_file(filename);

    // Construct lattice and linear operator
    std::shared_ptr<Lattice2d> lattice = std::make_shared<Lattice2d>(lattice_params.nx,
                                                                     lattice_params.ny);
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
    std::shared_ptr<ShiftedLaplaceFEMOperator> prior_operator = std::make_shared<ShiftedLaplaceFEMOperator>(lattice,
                                                                                                            correlationlengthmodel);

    std::shared_ptr<MeasuredOperator> posterior_operator = std::make_shared<MeasuredOperator>(prior_operator,
                                                                                              measurement_params);
    LinearOperator::DenseMatrixType covariance = posterior_operator->covariance();
    typedef Eigen::EigenSolver<LinearOperator::DenseMatrixType> EigenSolver;
    EigenSolver eigen_solver(covariance, false);
    EigenSolver::EigenvalueType eigen_values = eigen_solver.eigenvalues();
    unsigned int n = eigen_values.rows();
    std::vector<double> v(n);
    for (int j = 0; j < n; ++j)
    {
        v[j] = eigen_values[j].real();
    }
    std::sort(v.begin(), v.end());
    std::ofstream outfile;
    outfile.open("spectrum.csv");
    for (int j = 0; j < n; ++j)
    {
        outfile << v[j];
        if (j < n - 1)
        {
            outfile << ", ";
        }
        else
        {
            outfile << std::endl;
        }
    }
    outfile.close();
}
