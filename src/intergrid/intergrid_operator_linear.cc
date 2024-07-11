#include "intergrid_operator_linear.hh"

/** @file intergrid_operator_linear.cc
 * @brief Implementation of intergrid_operator_linear.hh
 */

/** @brief Create a new instance */
IntergridOperatorLinear::IntergridOperatorLinear(const std::shared_ptr<Lattice> lattice_) : Base(lattice_,
                                                                                                 int(pow(3, lattice_->dim())),
                                                                                                 1)
{
    int dim = lattice->dim();
    Eigen::VectorXi s(dim);
    // 1d stencil and shift vector
    const double stencil1d[3] = {0.5, 1.0, 0.5};
    const int shift1d[3] = {-1, 0, +1};
    std::vector<Eigen::VectorXi> shift;
    // matrix entries and shifts
    for (int j = 0; j < stencil_size; ++j)
    {
        matrix[j] = 1.0;
        int mu = j;
        for (int d = 0; d < dim; ++d)
        {
            matrix[j] *= stencil1d[mu % 3];
            s[d] = shift1d[mu % 3];
            mu /= 3;
        }
        shift.push_back(s);
    }
    compute_colidx(shift, stencil_size, colidx);

    // matrix entries and shifts
    interpolation_matrix[0] = 1.0;
    std::vector<Eigen::VectorXi> interpolation_shift;
    for (int d = 0; d < dim; ++d)
        s[d] = 0;
    interpolation_shift.push_back(s);
    compute_colidx(interpolation_shift, 1, interpolation_colidx);
}