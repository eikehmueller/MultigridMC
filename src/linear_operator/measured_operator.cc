#include "measured_operator.hh"

/** @file measured_operator.cc
 *
 * @brief Implementation of measured_operator.hh
 */

/* Create a new instance */
MeasuredOperator::MeasuredOperator(const std::shared_ptr<LinearOperator> base_operator_,
                                   const MeasurementParameters params_) : LinearOperator(base_operator_->get_lattice(),
                                                                                         params_.measurement_locations.size() + params_.measure_global),
                                                                          params(params_),
                                                                          base_operator(base_operator_)
{
    A_sparse = base_operator->get_sparse();
    unsigned int nrow = base_operator->get_lattice()->Nvertex;
    unsigned int n_measurements = params.measurement_locations.size();
    Sigma_diag = Eigen::DiagonalMatrix<double, Eigen::Dynamic>(n_measurements + params.measure_global);
    Sigma_diag.diagonal()(Eigen::seqN(0, n_measurements)) = params.variance_scaling * params.variance;
    typedef Eigen::Triplet<double> T;
    std::vector<T> triplet_list;
    for (int k = 0; k < n_measurements; ++k)
    {
        Eigen::VectorXd x_0 = params.measurement_locations[k];
        Eigen::SparseVector<double> r_meas = measurement_vector(x_0, params.radius);
        for (Eigen::SparseVector<double>::InnerIterator it(r_meas); it; ++it)
        {
            triplet_list.push_back(T(it.row(), k, it.value()));
        }
    }
    if (params.measure_global)
    {
        // shape of lattice
        Eigen::VectorXi shape = lattice->shape();
        // cell volume
        double cell_volume = 1.0;
        for (int d = 0; d < lattice->dim(); ++d)
        {
            cell_volume /= double(shape[d]);
        }
        for (int ell = 0; ell < nrow; ++ell)
        {
            triplet_list.push_back(T(ell, n_measurements, cell_volume));
        }
        Sigma_diag.diagonal()(n_measurements) = params.variance_global;
    }
    B.setFromTriplets(triplet_list.begin(), triplet_list.end());
    Sigma_inv_BT = get_Sigma_inv() * B.transpose();
}

/* radius of sphere */
double MeasuredOperator::V_sphere(const double radius, const unsigned int dim) const
{
    if (dim == 0)
    {
        return 1.0;
    }
    else if (dim == 1)
    {
        return 2. * radius;
    }
    else
    {
        return 2. * M_PI / double(dim) * radius * radius * V_sphere(radius, dim - 2);
    }
}

/* Create measurement vector in dual space */
Eigen::SparseVector<double> MeasuredOperator::measurement_vector(const Eigen::VectorXd x0, const double radius) const
{
    Eigen::SparseVector<double> r_meas(lattice->Nvertex);
    // dimension
    int dim = lattice->dim();
    if (radius < 1.E-12)
    {
        // find vertex which is closest to measurement point x0
        double d_min = double(dim);
        unsigned int ell_min = 0;
        // loop over all vertices
        for (unsigned int ell = 0; ell < lattice->Nvertex; ++ell)
        {
            Eigen::VectorXd x = lattice->vertex_coordinates(ell);
            double dist = (x - x0).norm();
            if (dist < d_min)
            {
                d_min = dist;
                ell_min = ell;
            }
        }
        r_meas.coeffRef(ell_min) = 1.0;
    }
    else
    {
        // shape of lattice
        Eigen::VectorXi shape = lattice->shape();
        // grid spacings in all directions
        Eigen::VectorXd h(dim);
        // cell volume
        double cell_volume = lattice->cell_volume();
        double normalisation = 1. / V_sphere(radius, dim);
        for (int d = 0; d < dim; ++d)
        {
            h[d] = 1. / double(shape[d]);
        }
        GaussLegendreQuadrature quadrature(dim, 1);
        std::vector<double> quad_weights = quadrature.get_weights();
        std::vector<Eigen::VectorXd> quad_points = quadrature.get_points();
        std::vector<int> basis_idx_1d{0, 1}; // indices used to identify the basis functions
        // Vector of all possible basis indices in d dimensions
        std::vector<std::vector<int>> basis_idx = cartesian_product(basis_idx_1d, dim);

        for (int cell_idx = 0; cell_idx < lattice->Ncell; ++cell_idx)
        { // loop over all cells of the lattice
            Eigen::VectorXi cell_coord = lattice->cellidx_linear2euclidean(cell_idx);
            bool overlap = false;
            Eigen::VectorXd x_corner_min(dim);
            x_corner_min.setConstant(2.0);
            Eigen::VectorXd x_corner_max(dim);
            x_corner_max.setConstant(-1.0);
            for (auto it = basis_idx.begin(); it != basis_idx.end(); ++it)
            { // loop over all corners of the cell and check whether one of them overlaps with
              // a ball of radius R around the point x_0
                Eigen::Map<Eigen::VectorXi> omega(it->data(), dim);
                Eigen::VectorXd x_corner = h.cwiseProduct((cell_coord + omega).cast<double>());
                x_corner_min = x_corner_min.cwiseMin(x_corner);
                x_corner_max = x_corner_max.cwiseMax(x_corner);
                overlap = overlap or ((x_corner - x0).norm() < radius);
            }
            // Check whether x0 lies in a particular cell
            bool centre_in_cell = true;
            for (int d = 0; d < dim; ++d)
            {
                centre_in_cell = centre_in_cell and
                                 (x_corner_min[d] <= x0[d]) and
                                 (x0[d] <= x_corner_max[d]);
            }
            overlap = overlap or centre_in_cell;
            if (not overlap) // move on to next cell if there is no overlap
                continue;
            for (auto it = basis_idx.begin(); it != basis_idx.end(); ++it)
            { // now loop over the basis functions associated with the corners
              // of the cell
                Eigen::Map<Eigen::VectorXi> alpha(it->data(), dim);
                unsigned int ell;
                if (lattice->corner_is_internal_vertex(cell_idx, alpha, ell))
                { // found an interior vertex
                    double local_entry = 0.0;
                    for (int j = 0; j < quad_points.size(); ++j)
                    { // Loop over all quadrature points
                        Eigen::VectorXd xhat = quad_points[j];
                        // Convert integer-valued coordinates to coordinates in domain
                        Eigen::VectorXd x = h.cwiseProduct(xhat + cell_coord.cast<double>());
                        // evaluate basis function
                        double xi = (x - x0).norm() / radius;
                        if (xi < 1.0)
                        {
                            double phihat = f_meas(xi);
                            for (int j = 0; j < dim; ++j)
                            {
                                phihat *= (alpha[j] == 0) ? (1.0 - xhat[j]) : xhat[j];
                            }
                            local_entry += phihat * quad_weights[j] * cell_volume * normalisation;
                        }
                    }
                    r_meas.coeffRef(ell) += local_entry;
                }
            }
        }
    }
    return r_meas;
}