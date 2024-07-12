#include "multigridmc_sampler.hh"
/** @file multigridmc_sampler.cc
 *
 * @brief Implementation of multigridmc_sampler.hh
 */

/** Create a new instance */
MultigridMCSampler::MultigridMCSampler(std::shared_ptr<LinearOperator> linear_operator_,
                                       std::shared_ptr<RandomGenerator> rng_,
                                       const MultigridParameters params_,
                                       const CholeskyParameters cholesky_params_) : Sampler(linear_operator_, rng_),
                                                                                    params(params_),
                                                                                    cholesky_params(cholesky_params_),
                                                                                    rhs_is_fixed(false)
{
    // Extract underlying fine lattice
    std::shared_ptr<Lattice> lattice = linear_operator->get_lattice();
    // Linear operator on a given level
    std::shared_ptr<LinearOperator> lin_op = linear_operator;
#pragma omp master
    if (params.verbose > 0)
    {
        std::cout << "Setting up Multigrid MC sampler " << std::endl;
    }

    std::shared_ptr<SamplerFactory> presampler_factory;
    std::shared_ptr<SamplerFactory> postsampler_factory;
    if (params.smoother == "SOR")
    {
        presampler_factory = std::make_shared<SORSamplerFactory>(rng,
                                                                 params.omega,
                                                                 params.npresmooth,
                                                                 forward);
        postsampler_factory = std::make_shared<SORSamplerFactory>(rng,
                                                                  params.omega,
                                                                  params.npostsmooth,
                                                                  backward);
    }
    else if (params.smoother == "SSOR")
    {
        presampler_factory = std::make_shared<SSORSamplerFactory>(rng,
                                                                  params.omega,
                                                                  params.npresmooth);
        postsampler_factory = std::make_shared<SSORSamplerFactory>(rng,
                                                                   params.omega,
                                                                   params.npostsmooth);
    }
    else
    {
        std::cout << "ERROR: invalid sampler \'" << params.smoother << "\'" << std::endl;
        exit(-1);
    }
    std::shared_ptr<IntergridOperatorFactory> intergrid_operator_factory = std::make_shared<IntergridOperatorLinearFactory>();
    std::shared_ptr<SamplerFactory> coarse_sampler_factory;
    if (params.coarse_solver == "Cholesky")
    {
        if (cholesky_params.factorisation == SparseFactorisation)
        {
            coarse_sampler_factory = std::make_shared<SparseCholeskySamplerFactory>(rng);
        }
        else if (cholesky_params.factorisation == LowRankFactorisation)
        {
            coarse_sampler_factory = std::make_shared<LowRankCholeskySamplerFactory>(rng);
        }
        else if (cholesky_params.factorisation == DenseFactorisation)
        {
            coarse_sampler_factory = std::make_shared<DenseCholeskySamplerFactory>(rng);
        }
    }
    else if (params.coarse_solver == "SSOR")
    {
        coarse_sampler_factory = std::make_shared<SSORSamplerFactory>(rng,
                                                                      params.omega,
                                                                      params.ncoarsesmooth);
    }
    else
    {
        std::cout << "ERROR: multigrid coarse sampler \'" << params.coarse_solver << "\'" << std::endl;
        exit(-1);
    }

    for (int level = 0; level < params.nlevel; ++level)
    {
#pragma omp master
        if (params.verbose > 0)
        {
            std::cout << "  level " << level << " lattice : " << lattice->get_info() << std::endl;
        }
        x_ell.push_back(Eigen::VectorXd(lattice->Nvertex));
        f_ell.push_back(Eigen::VectorXd(lattice->Nvertex));
        r_ell.push_back(Eigen::VectorXd(lattice->Nvertex));
        mu_ell.push_back(Eigen::VectorXd(lattice->Nvertex));
        linear_operators.push_back(lin_op);
        std::shared_ptr<Sampler> presampler = presampler_factory->get(lin_op);
        presamplers.push_back(presampler);
        std::shared_ptr<Sampler> postsampler = postsampler_factory->get(lin_op);
        postsamplers.push_back(postsampler);
        if (level < params.nlevel - 1)
        {
            std::shared_ptr<IntergridOperator> intergrid_operator = intergrid_operator_factory->get(lattice);
            intergrid_operators.push_back(intergrid_operator);
            lin_op = std::make_shared<LinearOperator>(lin_op->coarsen(intergrid_operator));
            //  Move to next-coarser lattice
            lattice = lattice->get_coarse_lattice();
        }
    }
    coarse_sampler = coarse_sampler_factory->get(lin_op);
}

/** Recursive sampling on a given level */
void MultigridMCSampler::sample(const unsigned int level) const
{
    if (level == params.nlevel - 1)
    {
        // Coarse level solve
        coarse_sampler->apply(f_ell[level], x_ell[level]);
    }
    else
    {
        int cycle_ = (level > 0) ? params.cycle : 1;
        for (int j = 0; j < cycle_; ++j)
        {
            // Presampler
            presamplers[level]->apply(f_ell[level], x_ell[level]);
            // Recursive call
            if (params.variant == "exact")
            {
                // Compute residual
                linear_operators[level]->apply(x_ell[level], r_ell[level]);
                r_ell[level] = f_ell[level] - r_ell[level];
                intergrid_operators[level]->restrict(r_ell[level], f_ell[level + 1]);
                x_ell[level + 1].setZero();
                sample(level + 1);
                // Prolongate and add
                intergrid_operators[level]->prolongate_add(params.coarse_scaling, x_ell[level + 1], x_ell[level]);
            }
            else if (params.variant == "fas")
            {
                unsigned int n = x_ell[level].size();
                unsigned int n_coarse = x_ell[level + 1].size();
                // TODO: construct vectors once in constructor
                Eigen::VectorXd x_old(n);
                Eigen::VectorXd x_tilde(n);
                Eigen::VectorXd x_old_coarse(n_coarse);
                // Copy old solution
                x_old = x_ell[level];
                intergrid_operators[level]->restrict(f_ell[level], f_ell[level + 1]);
                sample(level + 1);
                // Construct vector
                //   tilde(x)_ell = x_ell^{old}
                //                + I_{2h}^h ( x_{ell+1} - tilde(I)_{h}^{2h} x_ell^{old} )
                x_tilde = x_old;
                intergrid_operators[level]->interpolate(x_old, x_old_coarse);
                intergrid_operators[level]->prolongate_add(1.0, x_ell[level + 1], x_tilde);
                intergrid_operators[level]->prolongate_add(-1.0, x_old_coarse, x_tilde);
                // Acceptance ratio alpha
                double log_p = 0;
                // fine level
                log_p += log_probability(level, x_tilde) - log_probability(level, x_old);
                // coarse level
                log_p += log_probability(level + 1, x_old_coarse) - log_probability(level + 1, x_ell[level + 1]);
                double alpha = exp(log_p);
                double accept = (alpha >= 1.0);
                if (not accept)
                    accept = (rng->draw_uniform_real() < alpha);
                if (accept)
                    x_ell[level] = x_tilde;
            }
            else
            {
                std::cout << "Invalid multigrid variant " << params.variant << std::endl;
                exit(-1);
            }
            // Postsmooth
            postsamplers[level]->apply(f_ell[level], x_ell[level]);
        }
    }
}

/** Solve the linear system Ax = b with one iteration of the multigrid V-cycle */
void MultigridMCSampler::apply(const Eigen::VectorXd &f, Eigen::VectorXd &x) const
{
    x_ell[0] = x;
    if ((params.variant == "fas") and (not rhs_is_fixed))
    {
        set_rhs(f);
    }
    else
    {
        f_ell[0] = f;
    }
    sample(0);
    x = x_ell[0];
}

/** Fix the RHS */
void MultigridMCSampler::fix_rhs(const Eigen::VectorXd &f)
{
    rhs_is_fixed = true;
    set_rhs(f);
}

/** Set the RHS for all levels of the hierarchy and compute the corresponding means */
void MultigridMCSampler::set_rhs(const Eigen::VectorXd &f) const
{
    f_ell[0] = f;
    for (int level = 0; level < params.nlevel; ++level)
    {
        CholeskySolver solver(linear_operators[level]);
        solver.apply(f_ell[level], mu_ell[level]);
        if (level < params.nlevel - 1)
            intergrid_operators[level]->restrict(f_ell[level], f_ell[level + 1]);
    }
}

/** Compute the logarithm of the probability density, ignoring normalisation */
double MultigridMCSampler::log_probability(const unsigned int level,
                                           const Eigen::VectorXd &x) const
{
    // TODO: construct mu on each level once in the constructor

    unsigned int n = x.size();
    Eigen::VectorXd x_mu(n);
    Eigen::VectorXd A_x_mu(n);
    x_mu = x - mu_ell[level];
    linear_operators[level]->apply(x_mu, A_x_mu);
    return -0.5 * A_x_mu.dot(x_mu);
}
