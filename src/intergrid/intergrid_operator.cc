#include "intergrid_operator.hh"

/** @file intergrid_operator.cc
 * @brief Implementation of intergrid_operator.hh
 */

/* Compute column indices on entire lattice */
void IntergridOperator::compute_colidx(const std::vector<Eigen::VectorXi> shift,
                                       const int stencil_size_,
                                       unsigned int *colidx_ptr)
{
    std::shared_ptr<Lattice> coarse_lattice = lattice->get_coarse_lattice();
    for (unsigned int ell_coarse = 0; ell_coarse < coarse_lattice->Nvertex; ++ell_coarse)
    {
        Eigen::VectorXi idx_coarse = coarse_lattice->vertexidx_linear2euclidean(ell_coarse);
        unsigned int ell = lattice->vertexidx_euclidean2linear(2 * idx_coarse);
        for (int j = 0; j < stencil_size_; ++j)
        {
            colidx_ptr[ell_coarse * stencil_size_ + j] = lattice->shift_vertexidx(ell, shift[j]);
        }
    }
}