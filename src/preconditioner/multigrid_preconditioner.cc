#include "multigrid_preconditioner.hh"
/** @file multigrid_preconditioner.cc
 *
 * @brief Implementation of multigrid_preconditioner.hh
 */

/** Create a new instance */
MultigridPreconditioner::MultigridPreconditioner(std::shared_ptr<LinearOperator> linear_operator_,
                                                 const MultigridParameters params_) : Preconditioner(linear_operator_),
                                                                                      params(params_)
{
    // Extract underlying fine lattice
    std::shared_ptr<Lattice> lattice = linear_operator->get_lattice();
    // Linear operator on a given level
    std::shared_ptr<LinearOperator> lin_op = linear_operator;
    std::shared_ptr<SmootherFactory> presmoother_factory;
    std::shared_ptr<SmootherFactory> postsmoother_factory;
    if (params.smoother == "SOR")
    {
        presmoother_factory = std::make_shared<SORSmootherFactory>(params.omega,
                                                                   params.npresmooth,
                                                                   forward);
        postsmoother_factory = std::make_shared<SORSmootherFactory>(params.omega,
                                                                    params.npostsmooth,
                                                                    backward);
    }
    else if (params.smoother == "SSOR")
    {
        presmoother_factory = std::make_shared<SSORSmootherFactory>(params.omega,
                                                                    params.npresmooth);
        postsmoother_factory = std::make_shared<SSORSmootherFactory>(params.omega,
                                                                     params.npostsmooth);
    }
    else
    {
        std::cout << "ERROR: invalid smoother \'" << params.smoother << "\'" << std::endl;
        exit(-1);
    }
    std::shared_ptr<IntergridOperatorFactory> intergrid_operator_factory = std::make_shared<IntergridOperatorLinearFactory>();

    std::shared_ptr<LinearSolverFactory> coarse_solver_factory = std::make_shared<CholeskySolverFactory>();
    if (params.coarse_solver != "Cholesky")
    {
        std::cout << "WARNING: ignoring coarse solver setting \'" << params.coarse_solver << "\', using Choleksy." << std::endl;
    }

    for (int level = 0; level < params.nlevel; ++level)
    {
        if (params.verbose > 0)
        {
            std::cout << "  level " << level << " lattice : " << lattice->get_info() << std::endl;
        }
        x_ell.push_back(Eigen::VectorXd(lattice->Nvertex));
        b_ell.push_back(Eigen::VectorXd(lattice->Nvertex));
        r_ell.push_back(Eigen::VectorXd(lattice->Nvertex));
        linear_operators.push_back(lin_op);
        std::shared_ptr<Smoother> presmoother = presmoother_factory->get(lin_op);
        presmoothers.push_back(presmoother);
        std::shared_ptr<Smoother> postsmoother = postsmoother_factory->get(lin_op);
        postsmoothers.push_back(postsmoother);
        if (level < params.nlevel - 1)
        {
            std::shared_ptr<IntergridOperator> intergrid_operator = intergrid_operator_factory->get(lattice);
            intergrid_operators.push_back(intergrid_operator);
            lin_op = std::make_shared<LinearOperator>(lin_op->coarsen(intergrid_operator));
            //  Move to next-coarser lattice
            lattice = lattice->get_coarse_lattice();
        }
    }
    coarse_solver = coarse_solver_factory->get(lin_op);
}

/** Recursive solve on a givel level */
void MultigridPreconditioner::solve(const unsigned int level)
{
    x_ell[level].setZero();
    if (level == params.nlevel - 1)
    {
        // Coarse level solve
        coarse_solver->apply(b_ell[level], x_ell[level]);
    }
    else
    {
        int cycle_ = (level > 0) ? params.cycle : 1;
        for (int j = 0; j < cycle_; ++j)
        {
            // Presmooth
            presmoothers[level]->apply(b_ell[level], x_ell[level]);
            // Compute residual
            linear_operators[level]->apply(x_ell[level], r_ell[level]);
            r_ell[level] = b_ell[level] - r_ell[level];
            intergrid_operators[level]->restrict(r_ell[level], b_ell[level + 1]);
            // Recursive call
            solve(level + 1);
            // Prolongate and add
            intergrid_operators[level]->prolongate_add(params.coarse_scaling, x_ell[level + 1], x_ell[level]);
            // Postsmooth
            postsmoothers[level]->apply(b_ell[level], x_ell[level]);
        }
    }
}

/** Solve the linear system Ax = b with one iteration of the multigrid V-cycle */
void MultigridPreconditioner::apply(const Eigen::VectorXd &b, Eigen::VectorXd &x)
{
    b_ell[0] = b;
    solve(0);
    x = x_ell[0];
}