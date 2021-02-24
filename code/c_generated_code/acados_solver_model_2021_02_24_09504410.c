/*
 * Copyright 2019 Gianluca Frison, Dimitris Kouzoupis, Robin Verschueren,
 * Andrea Zanelli, Niels van Duijkeren, Jonathan Frey, Tommaso Sartor,
 * Branimir Novoselnik, Rien Quirynen, Rezart Qelibari, Dang Doan,
 * Jonas Koenemann, Yutao Chen, Tobias Schöls, Jonas Schlagenhauf, Moritz Diehl
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */

// standard
#include <stdio.h>
#include <stdlib.h>
// acados
#include "acados/utils/print.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

// example specific
#include "model_2021_02_24_09504410_model/model_2021_02_24_09504410_model.h"




#include "model_2021_02_24_09504410_cost/model_2021_02_24_09504410_cost_y_fun.h"
#include "model_2021_02_24_09504410_cost/model_2021_02_24_09504410_cost_y_e_fun.h"

#include "acados_solver_model_2021_02_24_09504410.h"

#define NX     8
#define NZ     0
#define NU     19
#define NP     0
#define NBX    8
#define NBX0   8
#define NBU    19
#define NSBX   0
#define NSBU   0
#define NSH    0
#define NSG    0
#define NSPHI  0
#define NSHN   0
#define NSGN   0
#define NSPHIN 0
#define NSBXN  0
#define NS     0
#define NSN    0
#define NG     0
#define NBXN   8
#define NGN    0
#define NY     31
#define NYN    12
#define N      7
#define NH     0
#define NPHI   0
#define NHN    0
#define NPHIN  0
#define NR     0


// ** global data **
ocp_nlp_in * nlp_in;
ocp_nlp_out * nlp_out;
ocp_nlp_solver * nlp_solver;
void * nlp_opts;
ocp_nlp_plan * nlp_solver_plan;
ocp_nlp_config * nlp_config;
ocp_nlp_dims * nlp_dims;

// number of expected runtime parameters
const unsigned int nlp_np = NP;


external_function_param_casadi * impl_dae_fun;
external_function_param_casadi * impl_dae_fun_jac_x_xdot_z;
external_function_param_casadi * impl_dae_jac_x_xdot_u_z;


external_function_param_casadi * nl_constr_h_fun;
external_function_param_casadi * nl_constr_h_fun_jac;


external_function_param_casadi nl_constr_h_e_fun_jac;
external_function_param_casadi nl_constr_h_e_fun;
external_function_param_casadi * cost_y_fun;
external_function_param_casadi * cost_y_fun_jac_ut_xt;
external_function_param_casadi * cost_y_hess;
external_function_param_casadi cost_y_e_fun;
external_function_param_casadi cost_y_e_fun_jac_ut_xt;
external_function_param_casadi cost_y_e_hess;


int acados_create()
{
    int status = 0;

    /************************************************
    *  plan & config
    ************************************************/
    nlp_solver_plan = ocp_nlp_plan_create(N);
    nlp_solver_plan->nlp_solver = SQP;
    

    nlp_solver_plan->ocp_qp_solver_plan.qp_solver = PARTIAL_CONDENSING_HPIPM;
    for (int i = 0; i < N; i++)
        nlp_solver_plan->nlp_cost[i] = NONLINEAR_LS;

    nlp_solver_plan->nlp_cost[N] = NONLINEAR_LS;

    for (int i = 0; i < N; i++)
    {
        
        nlp_solver_plan->nlp_dynamics[i] = CONTINUOUS_MODEL;
        nlp_solver_plan->sim_solver_plan[i].sim_solver = IRK;
    }

    for (int i = 0; i < N; i++)
    {
        nlp_solver_plan->nlp_constraints[i] = BGH;
    }
    nlp_solver_plan->nlp_constraints[N] = BGH;
    nlp_config = ocp_nlp_config_create(*nlp_solver_plan);


    /************************************************
    *  dimensions
    ************************************************/
    int nx[N+1];
    int nu[N+1];
    int nbx[N+1];
    int nbu[N+1];
    int nsbx[N+1];
    int nsbu[N+1];
    int nsg[N+1];
    int nsh[N+1];
    int nsphi[N+1];
    int ns[N+1];
    int ng[N+1];
    int nh[N+1];
    int nphi[N+1];
    int nz[N+1];
    int ny[N+1];
    int nr[N+1];
    int nbxe[N+1];

    for (int i = 0; i < N+1; i++)
    {
        // common
        nx[i]     = NX;
        nu[i]     = NU;
        nz[i]     = NZ;
        ns[i]     = NS;
        // cost
        ny[i]     = NY;
        // constraints
        nbx[i]    = NBX;
        nbu[i]    = NBU;
        nsbx[i]   = NSBX;
        nsbu[i]   = NSBU;
        nsg[i] = NSG;
        nsh[i]    = NSH;
        nsphi[i]  = NSPHI;
        ng[i]     = NG;
        nh[i]     = NH;
        nphi[i]   = NPHI;
        nr[i]     = NR;
        nbxe[i]   = 0;
    }

    // for initial state
    nbx[0]  = NBX0;
    nsbx[0] = 0;
    ns[0] = NS - NSBX;
    nbxe[0] = 0;

    // terminal - common
    nu[N]   = 0;
    nz[N]   = 0;
    ns[N]   = NSN;
    // cost
    ny[N]   = NYN;
    // constraint
    nbx[N]   = NBXN;
    nbu[N]   = 0;
    ng[N]    = NGN;
    nh[N]    = NHN;
    nphi[N]  = NPHIN;
    nr[N]    = 0;

    nsbx[N]  = NSBXN;
    nsbu[N]  = 0;
    nsg[N]   = NSGN;
    nsh[N]   = NSHN;
    nsphi[N] = NSPHIN;

    /* create and set ocp_nlp_dims */
    nlp_dims = ocp_nlp_dims_create(nlp_config);

    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nx", nx);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nu", nu);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nz", nz);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "ns", ns);

    for (int i = 0; i <= N; i++)
    {
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbx", &nbx[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbu", &nbu[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsbx", &nsbx[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsbu", &nsbu[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "ng", &ng[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsg", &nsg[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbxe", &nbxe[i]);
    }

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_dims_set_cost(nlp_config, nlp_dims, i, "ny", &ny[i]);
    }
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nh", &nh[N]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nsh", &nsh[N]);
    ocp_nlp_dims_set_cost(nlp_config, nlp_dims, N, "ny", &ny[N]);



    /************************************************
    *  external functions
    ************************************************/


    // implicit dae
    impl_dae_fun = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        impl_dae_fun[i].casadi_fun = &model_2021_02_24_09504410_impl_dae_fun;
        impl_dae_fun[i].casadi_work = &model_2021_02_24_09504410_impl_dae_fun_work;
        impl_dae_fun[i].casadi_sparsity_in = &model_2021_02_24_09504410_impl_dae_fun_sparsity_in;
        impl_dae_fun[i].casadi_sparsity_out = &model_2021_02_24_09504410_impl_dae_fun_sparsity_out;
        impl_dae_fun[i].casadi_n_in = &model_2021_02_24_09504410_impl_dae_fun_n_in;
        impl_dae_fun[i].casadi_n_out = &model_2021_02_24_09504410_impl_dae_fun_n_out;
        external_function_param_casadi_create(&impl_dae_fun[i], 0);
    }

    impl_dae_fun_jac_x_xdot_z = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        impl_dae_fun_jac_x_xdot_z[i].casadi_fun = &model_2021_02_24_09504410_impl_dae_fun_jac_x_xdot_z;
        impl_dae_fun_jac_x_xdot_z[i].casadi_work = &model_2021_02_24_09504410_impl_dae_fun_jac_x_xdot_z_work;
        impl_dae_fun_jac_x_xdot_z[i].casadi_sparsity_in = &model_2021_02_24_09504410_impl_dae_fun_jac_x_xdot_z_sparsity_in;
        impl_dae_fun_jac_x_xdot_z[i].casadi_sparsity_out = &model_2021_02_24_09504410_impl_dae_fun_jac_x_xdot_z_sparsity_out;
        impl_dae_fun_jac_x_xdot_z[i].casadi_n_in = &model_2021_02_24_09504410_impl_dae_fun_jac_x_xdot_z_n_in;
        impl_dae_fun_jac_x_xdot_z[i].casadi_n_out = &model_2021_02_24_09504410_impl_dae_fun_jac_x_xdot_z_n_out;
        external_function_param_casadi_create(&impl_dae_fun_jac_x_xdot_z[i], 0);
    }

    impl_dae_jac_x_xdot_u_z = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        impl_dae_jac_x_xdot_u_z[i].casadi_fun = &model_2021_02_24_09504410_impl_dae_jac_x_xdot_u_z;
        impl_dae_jac_x_xdot_u_z[i].casadi_work = &model_2021_02_24_09504410_impl_dae_jac_x_xdot_u_z_work;
        impl_dae_jac_x_xdot_u_z[i].casadi_sparsity_in = &model_2021_02_24_09504410_impl_dae_jac_x_xdot_u_z_sparsity_in;
        impl_dae_jac_x_xdot_u_z[i].casadi_sparsity_out = &model_2021_02_24_09504410_impl_dae_jac_x_xdot_u_z_sparsity_out;
        impl_dae_jac_x_xdot_u_z[i].casadi_n_in = &model_2021_02_24_09504410_impl_dae_jac_x_xdot_u_z_n_in;
        impl_dae_jac_x_xdot_u_z[i].casadi_n_out = &model_2021_02_24_09504410_impl_dae_jac_x_xdot_u_z_n_out;
        external_function_param_casadi_create(&impl_dae_jac_x_xdot_u_z[i], 0);
    }


    // nonlinear least squares cost
    cost_y_fun = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++)
    {
        cost_y_fun[i].casadi_fun = &model_2021_02_24_09504410_cost_y_fun;
        cost_y_fun[i].casadi_n_in = &model_2021_02_24_09504410_cost_y_fun_n_in;
        cost_y_fun[i].casadi_n_out = &model_2021_02_24_09504410_cost_y_fun_n_out;
        cost_y_fun[i].casadi_sparsity_in = &model_2021_02_24_09504410_cost_y_fun_sparsity_in;
        cost_y_fun[i].casadi_sparsity_out = &model_2021_02_24_09504410_cost_y_fun_sparsity_out;
        cost_y_fun[i].casadi_work = &model_2021_02_24_09504410_cost_y_fun_work;

        external_function_param_casadi_create(&cost_y_fun[i], 0);
    }

    cost_y_fun_jac_ut_xt = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++)
    {
        cost_y_fun_jac_ut_xt[i].casadi_fun = &model_2021_02_24_09504410_cost_y_fun_jac_ut_xt;
        cost_y_fun_jac_ut_xt[i].casadi_n_in = &model_2021_02_24_09504410_cost_y_fun_jac_ut_xt_n_in;
        cost_y_fun_jac_ut_xt[i].casadi_n_out = &model_2021_02_24_09504410_cost_y_fun_jac_ut_xt_n_out;
        cost_y_fun_jac_ut_xt[i].casadi_sparsity_in = &model_2021_02_24_09504410_cost_y_fun_jac_ut_xt_sparsity_in;
        cost_y_fun_jac_ut_xt[i].casadi_sparsity_out = &model_2021_02_24_09504410_cost_y_fun_jac_ut_xt_sparsity_out;
        cost_y_fun_jac_ut_xt[i].casadi_work = &model_2021_02_24_09504410_cost_y_fun_jac_ut_xt_work;

        external_function_param_casadi_create(&cost_y_fun_jac_ut_xt[i], 0);
    }

    cost_y_hess = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++)
    {
        cost_y_hess[i].casadi_fun = &model_2021_02_24_09504410_cost_y_hess;
        cost_y_hess[i].casadi_n_in = &model_2021_02_24_09504410_cost_y_hess_n_in;
        cost_y_hess[i].casadi_n_out = &model_2021_02_24_09504410_cost_y_hess_n_out;
        cost_y_hess[i].casadi_sparsity_in = &model_2021_02_24_09504410_cost_y_hess_sparsity_in;
        cost_y_hess[i].casadi_sparsity_out = &model_2021_02_24_09504410_cost_y_hess_sparsity_out;
        cost_y_hess[i].casadi_work = &model_2021_02_24_09504410_cost_y_hess_work;

        external_function_param_casadi_create(&cost_y_hess[i], 0);
    }
    // nonlinear least square function
    cost_y_e_fun.casadi_fun = &model_2021_02_24_09504410_cost_y_e_fun;
    cost_y_e_fun.casadi_n_in = &model_2021_02_24_09504410_cost_y_e_fun_n_in;
    cost_y_e_fun.casadi_n_out = &model_2021_02_24_09504410_cost_y_e_fun_n_out;
    cost_y_e_fun.casadi_sparsity_in = &model_2021_02_24_09504410_cost_y_e_fun_sparsity_in;
    cost_y_e_fun.casadi_sparsity_out = &model_2021_02_24_09504410_cost_y_e_fun_sparsity_out;
    cost_y_e_fun.casadi_work = &model_2021_02_24_09504410_cost_y_e_fun_work;
    external_function_param_casadi_create(&cost_y_e_fun, 0);

    cost_y_e_fun_jac_ut_xt.casadi_fun = &model_2021_02_24_09504410_cost_y_e_fun_jac_ut_xt;
    cost_y_e_fun_jac_ut_xt.casadi_n_in = &model_2021_02_24_09504410_cost_y_e_fun_jac_ut_xt_n_in;
    cost_y_e_fun_jac_ut_xt.casadi_n_out = &model_2021_02_24_09504410_cost_y_e_fun_jac_ut_xt_n_out;
    cost_y_e_fun_jac_ut_xt.casadi_sparsity_in = &model_2021_02_24_09504410_cost_y_e_fun_jac_ut_xt_sparsity_in;
    cost_y_e_fun_jac_ut_xt.casadi_sparsity_out = &model_2021_02_24_09504410_cost_y_e_fun_jac_ut_xt_sparsity_out;
    cost_y_e_fun_jac_ut_xt.casadi_work = &model_2021_02_24_09504410_cost_y_e_fun_jac_ut_xt_work;
    external_function_param_casadi_create(&cost_y_e_fun_jac_ut_xt, 0);

    cost_y_e_hess.casadi_fun = &model_2021_02_24_09504410_cost_y_e_hess;
    cost_y_e_hess.casadi_n_in = &model_2021_02_24_09504410_cost_y_e_hess_n_in;
    cost_y_e_hess.casadi_n_out = &model_2021_02_24_09504410_cost_y_e_hess_n_out;
    cost_y_e_hess.casadi_sparsity_in = &model_2021_02_24_09504410_cost_y_e_hess_sparsity_in;
    cost_y_e_hess.casadi_sparsity_out = &model_2021_02_24_09504410_cost_y_e_hess_sparsity_out;
    cost_y_e_hess.casadi_work = &model_2021_02_24_09504410_cost_y_e_hess_work;
    external_function_param_casadi_create(&cost_y_e_hess, 0);

    /************************************************
    *  nlp_in
    ************************************************/
    nlp_in = ocp_nlp_in_create(nlp_config, nlp_dims);

    double time_steps[N];
    time_steps[0] = 0.03;
    time_steps[1] = 0.03;
    time_steps[2] = 0.03;
    time_steps[3] = 0.03;
    time_steps[4] = 0.03;
    time_steps[5] = 0.03;
    time_steps[6] = 0.03;

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_in_set(nlp_config, nlp_dims, nlp_in, i, "Ts", &time_steps[i]);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "scaling", &time_steps[i]);
    }

    /**** Dynamics ****/
    for (int i = 0; i < N; i++)
    {
        ocp_nlp_dynamics_model_set(nlp_config, nlp_dims, nlp_in, i, "impl_dae_fun", &impl_dae_fun[i]);
        ocp_nlp_dynamics_model_set(nlp_config, nlp_dims, nlp_in, i,
                                   "impl_dae_fun_jac_x_xdot_z", &impl_dae_fun_jac_x_xdot_z[i]);
        ocp_nlp_dynamics_model_set(nlp_config, nlp_dims, nlp_in, i,
                                   "impl_dae_jac_x_xdot_u", &impl_dae_jac_x_xdot_u_z[i]);
    
    }


    /**** Cost ****/

    double W[NY*NY];
    
    W[0+(NY) * 0] = 100;
    W[0+(NY) * 1] = 0;
    W[0+(NY) * 2] = 0;
    W[0+(NY) * 3] = 0;
    W[0+(NY) * 4] = 0;
    W[0+(NY) * 5] = 0;
    W[0+(NY) * 6] = 0;
    W[0+(NY) * 7] = 0;
    W[0+(NY) * 8] = 0;
    W[0+(NY) * 9] = 0;
    W[0+(NY) * 10] = 0;
    W[0+(NY) * 11] = 0;
    W[0+(NY) * 12] = 0;
    W[0+(NY) * 13] = 0;
    W[0+(NY) * 14] = 0;
    W[0+(NY) * 15] = 0;
    W[0+(NY) * 16] = 0;
    W[0+(NY) * 17] = 0;
    W[0+(NY) * 18] = 0;
    W[0+(NY) * 19] = 0;
    W[0+(NY) * 20] = 0;
    W[0+(NY) * 21] = 0;
    W[0+(NY) * 22] = 0;
    W[0+(NY) * 23] = 0;
    W[0+(NY) * 24] = 0;
    W[0+(NY) * 25] = 0;
    W[0+(NY) * 26] = 0;
    W[0+(NY) * 27] = 0;
    W[0+(NY) * 28] = 0;
    W[0+(NY) * 29] = 0;
    W[0+(NY) * 30] = 0;
    W[1+(NY) * 0] = 0;
    W[1+(NY) * 1] = 100;
    W[1+(NY) * 2] = 0;
    W[1+(NY) * 3] = 0;
    W[1+(NY) * 4] = 0;
    W[1+(NY) * 5] = 0;
    W[1+(NY) * 6] = 0;
    W[1+(NY) * 7] = 0;
    W[1+(NY) * 8] = 0;
    W[1+(NY) * 9] = 0;
    W[1+(NY) * 10] = 0;
    W[1+(NY) * 11] = 0;
    W[1+(NY) * 12] = 0;
    W[1+(NY) * 13] = 0;
    W[1+(NY) * 14] = 0;
    W[1+(NY) * 15] = 0;
    W[1+(NY) * 16] = 0;
    W[1+(NY) * 17] = 0;
    W[1+(NY) * 18] = 0;
    W[1+(NY) * 19] = 0;
    W[1+(NY) * 20] = 0;
    W[1+(NY) * 21] = 0;
    W[1+(NY) * 22] = 0;
    W[1+(NY) * 23] = 0;
    W[1+(NY) * 24] = 0;
    W[1+(NY) * 25] = 0;
    W[1+(NY) * 26] = 0;
    W[1+(NY) * 27] = 0;
    W[1+(NY) * 28] = 0;
    W[1+(NY) * 29] = 0;
    W[1+(NY) * 30] = 0;
    W[2+(NY) * 0] = 0;
    W[2+(NY) * 1] = 0;
    W[2+(NY) * 2] = 100;
    W[2+(NY) * 3] = 0;
    W[2+(NY) * 4] = 0;
    W[2+(NY) * 5] = 0;
    W[2+(NY) * 6] = 0;
    W[2+(NY) * 7] = 0;
    W[2+(NY) * 8] = 0;
    W[2+(NY) * 9] = 0;
    W[2+(NY) * 10] = 0;
    W[2+(NY) * 11] = 0;
    W[2+(NY) * 12] = 0;
    W[2+(NY) * 13] = 0;
    W[2+(NY) * 14] = 0;
    W[2+(NY) * 15] = 0;
    W[2+(NY) * 16] = 0;
    W[2+(NY) * 17] = 0;
    W[2+(NY) * 18] = 0;
    W[2+(NY) * 19] = 0;
    W[2+(NY) * 20] = 0;
    W[2+(NY) * 21] = 0;
    W[2+(NY) * 22] = 0;
    W[2+(NY) * 23] = 0;
    W[2+(NY) * 24] = 0;
    W[2+(NY) * 25] = 0;
    W[2+(NY) * 26] = 0;
    W[2+(NY) * 27] = 0;
    W[2+(NY) * 28] = 0;
    W[2+(NY) * 29] = 0;
    W[2+(NY) * 30] = 0;
    W[3+(NY) * 0] = 0;
    W[3+(NY) * 1] = 0;
    W[3+(NY) * 2] = 0;
    W[3+(NY) * 3] = 100;
    W[3+(NY) * 4] = 0;
    W[3+(NY) * 5] = 0;
    W[3+(NY) * 6] = 0;
    W[3+(NY) * 7] = 0;
    W[3+(NY) * 8] = 0;
    W[3+(NY) * 9] = 0;
    W[3+(NY) * 10] = 0;
    W[3+(NY) * 11] = 0;
    W[3+(NY) * 12] = 0;
    W[3+(NY) * 13] = 0;
    W[3+(NY) * 14] = 0;
    W[3+(NY) * 15] = 0;
    W[3+(NY) * 16] = 0;
    W[3+(NY) * 17] = 0;
    W[3+(NY) * 18] = 0;
    W[3+(NY) * 19] = 0;
    W[3+(NY) * 20] = 0;
    W[3+(NY) * 21] = 0;
    W[3+(NY) * 22] = 0;
    W[3+(NY) * 23] = 0;
    W[3+(NY) * 24] = 0;
    W[3+(NY) * 25] = 0;
    W[3+(NY) * 26] = 0;
    W[3+(NY) * 27] = 0;
    W[3+(NY) * 28] = 0;
    W[3+(NY) * 29] = 0;
    W[3+(NY) * 30] = 0;
    W[4+(NY) * 0] = 0;
    W[4+(NY) * 1] = 0;
    W[4+(NY) * 2] = 0;
    W[4+(NY) * 3] = 0;
    W[4+(NY) * 4] = 100;
    W[4+(NY) * 5] = 0;
    W[4+(NY) * 6] = 0;
    W[4+(NY) * 7] = 0;
    W[4+(NY) * 8] = 0;
    W[4+(NY) * 9] = 0;
    W[4+(NY) * 10] = 0;
    W[4+(NY) * 11] = 0;
    W[4+(NY) * 12] = 0;
    W[4+(NY) * 13] = 0;
    W[4+(NY) * 14] = 0;
    W[4+(NY) * 15] = 0;
    W[4+(NY) * 16] = 0;
    W[4+(NY) * 17] = 0;
    W[4+(NY) * 18] = 0;
    W[4+(NY) * 19] = 0;
    W[4+(NY) * 20] = 0;
    W[4+(NY) * 21] = 0;
    W[4+(NY) * 22] = 0;
    W[4+(NY) * 23] = 0;
    W[4+(NY) * 24] = 0;
    W[4+(NY) * 25] = 0;
    W[4+(NY) * 26] = 0;
    W[4+(NY) * 27] = 0;
    W[4+(NY) * 28] = 0;
    W[4+(NY) * 29] = 0;
    W[4+(NY) * 30] = 0;
    W[5+(NY) * 0] = 0;
    W[5+(NY) * 1] = 0;
    W[5+(NY) * 2] = 0;
    W[5+(NY) * 3] = 0;
    W[5+(NY) * 4] = 0;
    W[5+(NY) * 5] = 100;
    W[5+(NY) * 6] = 0;
    W[5+(NY) * 7] = 0;
    W[5+(NY) * 8] = 0;
    W[5+(NY) * 9] = 0;
    W[5+(NY) * 10] = 0;
    W[5+(NY) * 11] = 0;
    W[5+(NY) * 12] = 0;
    W[5+(NY) * 13] = 0;
    W[5+(NY) * 14] = 0;
    W[5+(NY) * 15] = 0;
    W[5+(NY) * 16] = 0;
    W[5+(NY) * 17] = 0;
    W[5+(NY) * 18] = 0;
    W[5+(NY) * 19] = 0;
    W[5+(NY) * 20] = 0;
    W[5+(NY) * 21] = 0;
    W[5+(NY) * 22] = 0;
    W[5+(NY) * 23] = 0;
    W[5+(NY) * 24] = 0;
    W[5+(NY) * 25] = 0;
    W[5+(NY) * 26] = 0;
    W[5+(NY) * 27] = 0;
    W[5+(NY) * 28] = 0;
    W[5+(NY) * 29] = 0;
    W[5+(NY) * 30] = 0;
    W[6+(NY) * 0] = 0;
    W[6+(NY) * 1] = 0;
    W[6+(NY) * 2] = 0;
    W[6+(NY) * 3] = 0;
    W[6+(NY) * 4] = 0;
    W[6+(NY) * 5] = 0;
    W[6+(NY) * 6] = 100;
    W[6+(NY) * 7] = 0;
    W[6+(NY) * 8] = 0;
    W[6+(NY) * 9] = 0;
    W[6+(NY) * 10] = 0;
    W[6+(NY) * 11] = 0;
    W[6+(NY) * 12] = 0;
    W[6+(NY) * 13] = 0;
    W[6+(NY) * 14] = 0;
    W[6+(NY) * 15] = 0;
    W[6+(NY) * 16] = 0;
    W[6+(NY) * 17] = 0;
    W[6+(NY) * 18] = 0;
    W[6+(NY) * 19] = 0;
    W[6+(NY) * 20] = 0;
    W[6+(NY) * 21] = 0;
    W[6+(NY) * 22] = 0;
    W[6+(NY) * 23] = 0;
    W[6+(NY) * 24] = 0;
    W[6+(NY) * 25] = 0;
    W[6+(NY) * 26] = 0;
    W[6+(NY) * 27] = 0;
    W[6+(NY) * 28] = 0;
    W[6+(NY) * 29] = 0;
    W[6+(NY) * 30] = 0;
    W[7+(NY) * 0] = 0;
    W[7+(NY) * 1] = 0;
    W[7+(NY) * 2] = 0;
    W[7+(NY) * 3] = 0;
    W[7+(NY) * 4] = 0;
    W[7+(NY) * 5] = 0;
    W[7+(NY) * 6] = 0;
    W[7+(NY) * 7] = 100;
    W[7+(NY) * 8] = 0;
    W[7+(NY) * 9] = 0;
    W[7+(NY) * 10] = 0;
    W[7+(NY) * 11] = 0;
    W[7+(NY) * 12] = 0;
    W[7+(NY) * 13] = 0;
    W[7+(NY) * 14] = 0;
    W[7+(NY) * 15] = 0;
    W[7+(NY) * 16] = 0;
    W[7+(NY) * 17] = 0;
    W[7+(NY) * 18] = 0;
    W[7+(NY) * 19] = 0;
    W[7+(NY) * 20] = 0;
    W[7+(NY) * 21] = 0;
    W[7+(NY) * 22] = 0;
    W[7+(NY) * 23] = 0;
    W[7+(NY) * 24] = 0;
    W[7+(NY) * 25] = 0;
    W[7+(NY) * 26] = 0;
    W[7+(NY) * 27] = 0;
    W[7+(NY) * 28] = 0;
    W[7+(NY) * 29] = 0;
    W[7+(NY) * 30] = 0;
    W[8+(NY) * 0] = 0;
    W[8+(NY) * 1] = 0;
    W[8+(NY) * 2] = 0;
    W[8+(NY) * 3] = 0;
    W[8+(NY) * 4] = 0;
    W[8+(NY) * 5] = 0;
    W[8+(NY) * 6] = 0;
    W[8+(NY) * 7] = 0;
    W[8+(NY) * 8] = 100;
    W[8+(NY) * 9] = 0;
    W[8+(NY) * 10] = 0;
    W[8+(NY) * 11] = 0;
    W[8+(NY) * 12] = 0;
    W[8+(NY) * 13] = 0;
    W[8+(NY) * 14] = 0;
    W[8+(NY) * 15] = 0;
    W[8+(NY) * 16] = 0;
    W[8+(NY) * 17] = 0;
    W[8+(NY) * 18] = 0;
    W[8+(NY) * 19] = 0;
    W[8+(NY) * 20] = 0;
    W[8+(NY) * 21] = 0;
    W[8+(NY) * 22] = 0;
    W[8+(NY) * 23] = 0;
    W[8+(NY) * 24] = 0;
    W[8+(NY) * 25] = 0;
    W[8+(NY) * 26] = 0;
    W[8+(NY) * 27] = 0;
    W[8+(NY) * 28] = 0;
    W[8+(NY) * 29] = 0;
    W[8+(NY) * 30] = 0;
    W[9+(NY) * 0] = 0;
    W[9+(NY) * 1] = 0;
    W[9+(NY) * 2] = 0;
    W[9+(NY) * 3] = 0;
    W[9+(NY) * 4] = 0;
    W[9+(NY) * 5] = 0;
    W[9+(NY) * 6] = 0;
    W[9+(NY) * 7] = 0;
    W[9+(NY) * 8] = 0;
    W[9+(NY) * 9] = 100;
    W[9+(NY) * 10] = 0;
    W[9+(NY) * 11] = 0;
    W[9+(NY) * 12] = 0;
    W[9+(NY) * 13] = 0;
    W[9+(NY) * 14] = 0;
    W[9+(NY) * 15] = 0;
    W[9+(NY) * 16] = 0;
    W[9+(NY) * 17] = 0;
    W[9+(NY) * 18] = 0;
    W[9+(NY) * 19] = 0;
    W[9+(NY) * 20] = 0;
    W[9+(NY) * 21] = 0;
    W[9+(NY) * 22] = 0;
    W[9+(NY) * 23] = 0;
    W[9+(NY) * 24] = 0;
    W[9+(NY) * 25] = 0;
    W[9+(NY) * 26] = 0;
    W[9+(NY) * 27] = 0;
    W[9+(NY) * 28] = 0;
    W[9+(NY) * 29] = 0;
    W[9+(NY) * 30] = 0;
    W[10+(NY) * 0] = 0;
    W[10+(NY) * 1] = 0;
    W[10+(NY) * 2] = 0;
    W[10+(NY) * 3] = 0;
    W[10+(NY) * 4] = 0;
    W[10+(NY) * 5] = 0;
    W[10+(NY) * 6] = 0;
    W[10+(NY) * 7] = 0;
    W[10+(NY) * 8] = 0;
    W[10+(NY) * 9] = 0;
    W[10+(NY) * 10] = 100;
    W[10+(NY) * 11] = 0;
    W[10+(NY) * 12] = 0;
    W[10+(NY) * 13] = 0;
    W[10+(NY) * 14] = 0;
    W[10+(NY) * 15] = 0;
    W[10+(NY) * 16] = 0;
    W[10+(NY) * 17] = 0;
    W[10+(NY) * 18] = 0;
    W[10+(NY) * 19] = 0;
    W[10+(NY) * 20] = 0;
    W[10+(NY) * 21] = 0;
    W[10+(NY) * 22] = 0;
    W[10+(NY) * 23] = 0;
    W[10+(NY) * 24] = 0;
    W[10+(NY) * 25] = 0;
    W[10+(NY) * 26] = 0;
    W[10+(NY) * 27] = 0;
    W[10+(NY) * 28] = 0;
    W[10+(NY) * 29] = 0;
    W[10+(NY) * 30] = 0;
    W[11+(NY) * 0] = 0;
    W[11+(NY) * 1] = 0;
    W[11+(NY) * 2] = 0;
    W[11+(NY) * 3] = 0;
    W[11+(NY) * 4] = 0;
    W[11+(NY) * 5] = 0;
    W[11+(NY) * 6] = 0;
    W[11+(NY) * 7] = 0;
    W[11+(NY) * 8] = 0;
    W[11+(NY) * 9] = 0;
    W[11+(NY) * 10] = 0;
    W[11+(NY) * 11] = 100;
    W[11+(NY) * 12] = 0;
    W[11+(NY) * 13] = 0;
    W[11+(NY) * 14] = 0;
    W[11+(NY) * 15] = 0;
    W[11+(NY) * 16] = 0;
    W[11+(NY) * 17] = 0;
    W[11+(NY) * 18] = 0;
    W[11+(NY) * 19] = 0;
    W[11+(NY) * 20] = 0;
    W[11+(NY) * 21] = 0;
    W[11+(NY) * 22] = 0;
    W[11+(NY) * 23] = 0;
    W[11+(NY) * 24] = 0;
    W[11+(NY) * 25] = 0;
    W[11+(NY) * 26] = 0;
    W[11+(NY) * 27] = 0;
    W[11+(NY) * 28] = 0;
    W[11+(NY) * 29] = 0;
    W[11+(NY) * 30] = 0;
    W[12+(NY) * 0] = 0;
    W[12+(NY) * 1] = 0;
    W[12+(NY) * 2] = 0;
    W[12+(NY) * 3] = 0;
    W[12+(NY) * 4] = 0;
    W[12+(NY) * 5] = 0;
    W[12+(NY) * 6] = 0;
    W[12+(NY) * 7] = 0;
    W[12+(NY) * 8] = 0;
    W[12+(NY) * 9] = 0;
    W[12+(NY) * 10] = 0;
    W[12+(NY) * 11] = 0;
    W[12+(NY) * 12] = 100;
    W[12+(NY) * 13] = 0;
    W[12+(NY) * 14] = 0;
    W[12+(NY) * 15] = 0;
    W[12+(NY) * 16] = 0;
    W[12+(NY) * 17] = 0;
    W[12+(NY) * 18] = 0;
    W[12+(NY) * 19] = 0;
    W[12+(NY) * 20] = 0;
    W[12+(NY) * 21] = 0;
    W[12+(NY) * 22] = 0;
    W[12+(NY) * 23] = 0;
    W[12+(NY) * 24] = 0;
    W[12+(NY) * 25] = 0;
    W[12+(NY) * 26] = 0;
    W[12+(NY) * 27] = 0;
    W[12+(NY) * 28] = 0;
    W[12+(NY) * 29] = 0;
    W[12+(NY) * 30] = 0;
    W[13+(NY) * 0] = 0;
    W[13+(NY) * 1] = 0;
    W[13+(NY) * 2] = 0;
    W[13+(NY) * 3] = 0;
    W[13+(NY) * 4] = 0;
    W[13+(NY) * 5] = 0;
    W[13+(NY) * 6] = 0;
    W[13+(NY) * 7] = 0;
    W[13+(NY) * 8] = 0;
    W[13+(NY) * 9] = 0;
    W[13+(NY) * 10] = 0;
    W[13+(NY) * 11] = 0;
    W[13+(NY) * 12] = 0;
    W[13+(NY) * 13] = 100;
    W[13+(NY) * 14] = 0;
    W[13+(NY) * 15] = 0;
    W[13+(NY) * 16] = 0;
    W[13+(NY) * 17] = 0;
    W[13+(NY) * 18] = 0;
    W[13+(NY) * 19] = 0;
    W[13+(NY) * 20] = 0;
    W[13+(NY) * 21] = 0;
    W[13+(NY) * 22] = 0;
    W[13+(NY) * 23] = 0;
    W[13+(NY) * 24] = 0;
    W[13+(NY) * 25] = 0;
    W[13+(NY) * 26] = 0;
    W[13+(NY) * 27] = 0;
    W[13+(NY) * 28] = 0;
    W[13+(NY) * 29] = 0;
    W[13+(NY) * 30] = 0;
    W[14+(NY) * 0] = 0;
    W[14+(NY) * 1] = 0;
    W[14+(NY) * 2] = 0;
    W[14+(NY) * 3] = 0;
    W[14+(NY) * 4] = 0;
    W[14+(NY) * 5] = 0;
    W[14+(NY) * 6] = 0;
    W[14+(NY) * 7] = 0;
    W[14+(NY) * 8] = 0;
    W[14+(NY) * 9] = 0;
    W[14+(NY) * 10] = 0;
    W[14+(NY) * 11] = 0;
    W[14+(NY) * 12] = 0;
    W[14+(NY) * 13] = 0;
    W[14+(NY) * 14] = 100;
    W[14+(NY) * 15] = 0;
    W[14+(NY) * 16] = 0;
    W[14+(NY) * 17] = 0;
    W[14+(NY) * 18] = 0;
    W[14+(NY) * 19] = 0;
    W[14+(NY) * 20] = 0;
    W[14+(NY) * 21] = 0;
    W[14+(NY) * 22] = 0;
    W[14+(NY) * 23] = 0;
    W[14+(NY) * 24] = 0;
    W[14+(NY) * 25] = 0;
    W[14+(NY) * 26] = 0;
    W[14+(NY) * 27] = 0;
    W[14+(NY) * 28] = 0;
    W[14+(NY) * 29] = 0;
    W[14+(NY) * 30] = 0;
    W[15+(NY) * 0] = 0;
    W[15+(NY) * 1] = 0;
    W[15+(NY) * 2] = 0;
    W[15+(NY) * 3] = 0;
    W[15+(NY) * 4] = 0;
    W[15+(NY) * 5] = 0;
    W[15+(NY) * 6] = 0;
    W[15+(NY) * 7] = 0;
    W[15+(NY) * 8] = 0;
    W[15+(NY) * 9] = 0;
    W[15+(NY) * 10] = 0;
    W[15+(NY) * 11] = 0;
    W[15+(NY) * 12] = 0;
    W[15+(NY) * 13] = 0;
    W[15+(NY) * 14] = 0;
    W[15+(NY) * 15] = 100;
    W[15+(NY) * 16] = 0;
    W[15+(NY) * 17] = 0;
    W[15+(NY) * 18] = 0;
    W[15+(NY) * 19] = 0;
    W[15+(NY) * 20] = 0;
    W[15+(NY) * 21] = 0;
    W[15+(NY) * 22] = 0;
    W[15+(NY) * 23] = 0;
    W[15+(NY) * 24] = 0;
    W[15+(NY) * 25] = 0;
    W[15+(NY) * 26] = 0;
    W[15+(NY) * 27] = 0;
    W[15+(NY) * 28] = 0;
    W[15+(NY) * 29] = 0;
    W[15+(NY) * 30] = 0;
    W[16+(NY) * 0] = 0;
    W[16+(NY) * 1] = 0;
    W[16+(NY) * 2] = 0;
    W[16+(NY) * 3] = 0;
    W[16+(NY) * 4] = 0;
    W[16+(NY) * 5] = 0;
    W[16+(NY) * 6] = 0;
    W[16+(NY) * 7] = 0;
    W[16+(NY) * 8] = 0;
    W[16+(NY) * 9] = 0;
    W[16+(NY) * 10] = 0;
    W[16+(NY) * 11] = 0;
    W[16+(NY) * 12] = 0;
    W[16+(NY) * 13] = 0;
    W[16+(NY) * 14] = 0;
    W[16+(NY) * 15] = 0;
    W[16+(NY) * 16] = 100;
    W[16+(NY) * 17] = 0;
    W[16+(NY) * 18] = 0;
    W[16+(NY) * 19] = 0;
    W[16+(NY) * 20] = 0;
    W[16+(NY) * 21] = 0;
    W[16+(NY) * 22] = 0;
    W[16+(NY) * 23] = 0;
    W[16+(NY) * 24] = 0;
    W[16+(NY) * 25] = 0;
    W[16+(NY) * 26] = 0;
    W[16+(NY) * 27] = 0;
    W[16+(NY) * 28] = 0;
    W[16+(NY) * 29] = 0;
    W[16+(NY) * 30] = 0;
    W[17+(NY) * 0] = 0;
    W[17+(NY) * 1] = 0;
    W[17+(NY) * 2] = 0;
    W[17+(NY) * 3] = 0;
    W[17+(NY) * 4] = 0;
    W[17+(NY) * 5] = 0;
    W[17+(NY) * 6] = 0;
    W[17+(NY) * 7] = 0;
    W[17+(NY) * 8] = 0;
    W[17+(NY) * 9] = 0;
    W[17+(NY) * 10] = 0;
    W[17+(NY) * 11] = 0;
    W[17+(NY) * 12] = 0;
    W[17+(NY) * 13] = 0;
    W[17+(NY) * 14] = 0;
    W[17+(NY) * 15] = 0;
    W[17+(NY) * 16] = 0;
    W[17+(NY) * 17] = 100;
    W[17+(NY) * 18] = 0;
    W[17+(NY) * 19] = 0;
    W[17+(NY) * 20] = 0;
    W[17+(NY) * 21] = 0;
    W[17+(NY) * 22] = 0;
    W[17+(NY) * 23] = 0;
    W[17+(NY) * 24] = 0;
    W[17+(NY) * 25] = 0;
    W[17+(NY) * 26] = 0;
    W[17+(NY) * 27] = 0;
    W[17+(NY) * 28] = 0;
    W[17+(NY) * 29] = 0;
    W[17+(NY) * 30] = 0;
    W[18+(NY) * 0] = 0;
    W[18+(NY) * 1] = 0;
    W[18+(NY) * 2] = 0;
    W[18+(NY) * 3] = 0;
    W[18+(NY) * 4] = 0;
    W[18+(NY) * 5] = 0;
    W[18+(NY) * 6] = 0;
    W[18+(NY) * 7] = 0;
    W[18+(NY) * 8] = 0;
    W[18+(NY) * 9] = 0;
    W[18+(NY) * 10] = 0;
    W[18+(NY) * 11] = 0;
    W[18+(NY) * 12] = 0;
    W[18+(NY) * 13] = 0;
    W[18+(NY) * 14] = 0;
    W[18+(NY) * 15] = 0;
    W[18+(NY) * 16] = 0;
    W[18+(NY) * 17] = 0;
    W[18+(NY) * 18] = 100;
    W[18+(NY) * 19] = 0;
    W[18+(NY) * 20] = 0;
    W[18+(NY) * 21] = 0;
    W[18+(NY) * 22] = 0;
    W[18+(NY) * 23] = 0;
    W[18+(NY) * 24] = 0;
    W[18+(NY) * 25] = 0;
    W[18+(NY) * 26] = 0;
    W[18+(NY) * 27] = 0;
    W[18+(NY) * 28] = 0;
    W[18+(NY) * 29] = 0;
    W[18+(NY) * 30] = 0;
    W[19+(NY) * 0] = 0;
    W[19+(NY) * 1] = 0;
    W[19+(NY) * 2] = 0;
    W[19+(NY) * 3] = 0;
    W[19+(NY) * 4] = 0;
    W[19+(NY) * 5] = 0;
    W[19+(NY) * 6] = 0;
    W[19+(NY) * 7] = 0;
    W[19+(NY) * 8] = 0;
    W[19+(NY) * 9] = 0;
    W[19+(NY) * 10] = 0;
    W[19+(NY) * 11] = 0;
    W[19+(NY) * 12] = 0;
    W[19+(NY) * 13] = 0;
    W[19+(NY) * 14] = 0;
    W[19+(NY) * 15] = 0;
    W[19+(NY) * 16] = 0;
    W[19+(NY) * 17] = 0;
    W[19+(NY) * 18] = 0;
    W[19+(NY) * 19] = 100;
    W[19+(NY) * 20] = 0;
    W[19+(NY) * 21] = 0;
    W[19+(NY) * 22] = 0;
    W[19+(NY) * 23] = 0;
    W[19+(NY) * 24] = 0;
    W[19+(NY) * 25] = 0;
    W[19+(NY) * 26] = 0;
    W[19+(NY) * 27] = 0;
    W[19+(NY) * 28] = 0;
    W[19+(NY) * 29] = 0;
    W[19+(NY) * 30] = 0;
    W[20+(NY) * 0] = 0;
    W[20+(NY) * 1] = 0;
    W[20+(NY) * 2] = 0;
    W[20+(NY) * 3] = 0;
    W[20+(NY) * 4] = 0;
    W[20+(NY) * 5] = 0;
    W[20+(NY) * 6] = 0;
    W[20+(NY) * 7] = 0;
    W[20+(NY) * 8] = 0;
    W[20+(NY) * 9] = 0;
    W[20+(NY) * 10] = 0;
    W[20+(NY) * 11] = 0;
    W[20+(NY) * 12] = 0;
    W[20+(NY) * 13] = 0;
    W[20+(NY) * 14] = 0;
    W[20+(NY) * 15] = 0;
    W[20+(NY) * 16] = 0;
    W[20+(NY) * 17] = 0;
    W[20+(NY) * 18] = 0;
    W[20+(NY) * 19] = 0;
    W[20+(NY) * 20] = 100;
    W[20+(NY) * 21] = 0;
    W[20+(NY) * 22] = 0;
    W[20+(NY) * 23] = 0;
    W[20+(NY) * 24] = 0;
    W[20+(NY) * 25] = 0;
    W[20+(NY) * 26] = 0;
    W[20+(NY) * 27] = 0;
    W[20+(NY) * 28] = 0;
    W[20+(NY) * 29] = 0;
    W[20+(NY) * 30] = 0;
    W[21+(NY) * 0] = 0;
    W[21+(NY) * 1] = 0;
    W[21+(NY) * 2] = 0;
    W[21+(NY) * 3] = 0;
    W[21+(NY) * 4] = 0;
    W[21+(NY) * 5] = 0;
    W[21+(NY) * 6] = 0;
    W[21+(NY) * 7] = 0;
    W[21+(NY) * 8] = 0;
    W[21+(NY) * 9] = 0;
    W[21+(NY) * 10] = 0;
    W[21+(NY) * 11] = 0;
    W[21+(NY) * 12] = 0;
    W[21+(NY) * 13] = 0;
    W[21+(NY) * 14] = 0;
    W[21+(NY) * 15] = 0;
    W[21+(NY) * 16] = 0;
    W[21+(NY) * 17] = 0;
    W[21+(NY) * 18] = 0;
    W[21+(NY) * 19] = 0;
    W[21+(NY) * 20] = 0;
    W[21+(NY) * 21] = 100;
    W[21+(NY) * 22] = 0;
    W[21+(NY) * 23] = 0;
    W[21+(NY) * 24] = 0;
    W[21+(NY) * 25] = 0;
    W[21+(NY) * 26] = 0;
    W[21+(NY) * 27] = 0;
    W[21+(NY) * 28] = 0;
    W[21+(NY) * 29] = 0;
    W[21+(NY) * 30] = 0;
    W[22+(NY) * 0] = 0;
    W[22+(NY) * 1] = 0;
    W[22+(NY) * 2] = 0;
    W[22+(NY) * 3] = 0;
    W[22+(NY) * 4] = 0;
    W[22+(NY) * 5] = 0;
    W[22+(NY) * 6] = 0;
    W[22+(NY) * 7] = 0;
    W[22+(NY) * 8] = 0;
    W[22+(NY) * 9] = 0;
    W[22+(NY) * 10] = 0;
    W[22+(NY) * 11] = 0;
    W[22+(NY) * 12] = 0;
    W[22+(NY) * 13] = 0;
    W[22+(NY) * 14] = 0;
    W[22+(NY) * 15] = 0;
    W[22+(NY) * 16] = 0;
    W[22+(NY) * 17] = 0;
    W[22+(NY) * 18] = 0;
    W[22+(NY) * 19] = 0;
    W[22+(NY) * 20] = 0;
    W[22+(NY) * 21] = 0;
    W[22+(NY) * 22] = 100;
    W[22+(NY) * 23] = 0;
    W[22+(NY) * 24] = 0;
    W[22+(NY) * 25] = 0;
    W[22+(NY) * 26] = 0;
    W[22+(NY) * 27] = 0;
    W[22+(NY) * 28] = 0;
    W[22+(NY) * 29] = 0;
    W[22+(NY) * 30] = 0;
    W[23+(NY) * 0] = 0;
    W[23+(NY) * 1] = 0;
    W[23+(NY) * 2] = 0;
    W[23+(NY) * 3] = 0;
    W[23+(NY) * 4] = 0;
    W[23+(NY) * 5] = 0;
    W[23+(NY) * 6] = 0;
    W[23+(NY) * 7] = 0;
    W[23+(NY) * 8] = 0;
    W[23+(NY) * 9] = 0;
    W[23+(NY) * 10] = 0;
    W[23+(NY) * 11] = 0;
    W[23+(NY) * 12] = 0;
    W[23+(NY) * 13] = 0;
    W[23+(NY) * 14] = 0;
    W[23+(NY) * 15] = 0;
    W[23+(NY) * 16] = 0;
    W[23+(NY) * 17] = 0;
    W[23+(NY) * 18] = 0;
    W[23+(NY) * 19] = 0;
    W[23+(NY) * 20] = 0;
    W[23+(NY) * 21] = 0;
    W[23+(NY) * 22] = 0;
    W[23+(NY) * 23] = 10;
    W[23+(NY) * 24] = 0;
    W[23+(NY) * 25] = 0;
    W[23+(NY) * 26] = 0;
    W[23+(NY) * 27] = 0;
    W[23+(NY) * 28] = 0;
    W[23+(NY) * 29] = 0;
    W[23+(NY) * 30] = 0;
    W[24+(NY) * 0] = 0;
    W[24+(NY) * 1] = 0;
    W[24+(NY) * 2] = 0;
    W[24+(NY) * 3] = 0;
    W[24+(NY) * 4] = 0;
    W[24+(NY) * 5] = 0;
    W[24+(NY) * 6] = 0;
    W[24+(NY) * 7] = 0;
    W[24+(NY) * 8] = 0;
    W[24+(NY) * 9] = 0;
    W[24+(NY) * 10] = 0;
    W[24+(NY) * 11] = 0;
    W[24+(NY) * 12] = 0;
    W[24+(NY) * 13] = 0;
    W[24+(NY) * 14] = 0;
    W[24+(NY) * 15] = 0;
    W[24+(NY) * 16] = 0;
    W[24+(NY) * 17] = 0;
    W[24+(NY) * 18] = 0;
    W[24+(NY) * 19] = 0;
    W[24+(NY) * 20] = 0;
    W[24+(NY) * 21] = 0;
    W[24+(NY) * 22] = 0;
    W[24+(NY) * 23] = 0;
    W[24+(NY) * 24] = 10;
    W[24+(NY) * 25] = 0;
    W[24+(NY) * 26] = 0;
    W[24+(NY) * 27] = 0;
    W[24+(NY) * 28] = 0;
    W[24+(NY) * 29] = 0;
    W[24+(NY) * 30] = 0;
    W[25+(NY) * 0] = 0;
    W[25+(NY) * 1] = 0;
    W[25+(NY) * 2] = 0;
    W[25+(NY) * 3] = 0;
    W[25+(NY) * 4] = 0;
    W[25+(NY) * 5] = 0;
    W[25+(NY) * 6] = 0;
    W[25+(NY) * 7] = 0;
    W[25+(NY) * 8] = 0;
    W[25+(NY) * 9] = 0;
    W[25+(NY) * 10] = 0;
    W[25+(NY) * 11] = 0;
    W[25+(NY) * 12] = 0;
    W[25+(NY) * 13] = 0;
    W[25+(NY) * 14] = 0;
    W[25+(NY) * 15] = 0;
    W[25+(NY) * 16] = 0;
    W[25+(NY) * 17] = 0;
    W[25+(NY) * 18] = 0;
    W[25+(NY) * 19] = 0;
    W[25+(NY) * 20] = 0;
    W[25+(NY) * 21] = 0;
    W[25+(NY) * 22] = 0;
    W[25+(NY) * 23] = 0;
    W[25+(NY) * 24] = 0;
    W[25+(NY) * 25] = 10;
    W[25+(NY) * 26] = 0;
    W[25+(NY) * 27] = 0;
    W[25+(NY) * 28] = 0;
    W[25+(NY) * 29] = 0;
    W[25+(NY) * 30] = 0;
    W[26+(NY) * 0] = 0;
    W[26+(NY) * 1] = 0;
    W[26+(NY) * 2] = 0;
    W[26+(NY) * 3] = 0;
    W[26+(NY) * 4] = 0;
    W[26+(NY) * 5] = 0;
    W[26+(NY) * 6] = 0;
    W[26+(NY) * 7] = 0;
    W[26+(NY) * 8] = 0;
    W[26+(NY) * 9] = 0;
    W[26+(NY) * 10] = 0;
    W[26+(NY) * 11] = 0;
    W[26+(NY) * 12] = 0;
    W[26+(NY) * 13] = 0;
    W[26+(NY) * 14] = 0;
    W[26+(NY) * 15] = 0;
    W[26+(NY) * 16] = 0;
    W[26+(NY) * 17] = 0;
    W[26+(NY) * 18] = 0;
    W[26+(NY) * 19] = 0;
    W[26+(NY) * 20] = 0;
    W[26+(NY) * 21] = 0;
    W[26+(NY) * 22] = 0;
    W[26+(NY) * 23] = 0;
    W[26+(NY) * 24] = 0;
    W[26+(NY) * 25] = 0;
    W[26+(NY) * 26] = 10;
    W[26+(NY) * 27] = 0;
    W[26+(NY) * 28] = 0;
    W[26+(NY) * 29] = 0;
    W[26+(NY) * 30] = 0;
    W[27+(NY) * 0] = 0;
    W[27+(NY) * 1] = 0;
    W[27+(NY) * 2] = 0;
    W[27+(NY) * 3] = 0;
    W[27+(NY) * 4] = 0;
    W[27+(NY) * 5] = 0;
    W[27+(NY) * 6] = 0;
    W[27+(NY) * 7] = 0;
    W[27+(NY) * 8] = 0;
    W[27+(NY) * 9] = 0;
    W[27+(NY) * 10] = 0;
    W[27+(NY) * 11] = 0;
    W[27+(NY) * 12] = 0;
    W[27+(NY) * 13] = 0;
    W[27+(NY) * 14] = 0;
    W[27+(NY) * 15] = 0;
    W[27+(NY) * 16] = 0;
    W[27+(NY) * 17] = 0;
    W[27+(NY) * 18] = 0;
    W[27+(NY) * 19] = 0;
    W[27+(NY) * 20] = 0;
    W[27+(NY) * 21] = 0;
    W[27+(NY) * 22] = 0;
    W[27+(NY) * 23] = 0;
    W[27+(NY) * 24] = 0;
    W[27+(NY) * 25] = 0;
    W[27+(NY) * 26] = 0;
    W[27+(NY) * 27] = 1000;
    W[27+(NY) * 28] = 0;
    W[27+(NY) * 29] = 0;
    W[27+(NY) * 30] = 0;
    W[28+(NY) * 0] = 0;
    W[28+(NY) * 1] = 0;
    W[28+(NY) * 2] = 0;
    W[28+(NY) * 3] = 0;
    W[28+(NY) * 4] = 0;
    W[28+(NY) * 5] = 0;
    W[28+(NY) * 6] = 0;
    W[28+(NY) * 7] = 0;
    W[28+(NY) * 8] = 0;
    W[28+(NY) * 9] = 0;
    W[28+(NY) * 10] = 0;
    W[28+(NY) * 11] = 0;
    W[28+(NY) * 12] = 0;
    W[28+(NY) * 13] = 0;
    W[28+(NY) * 14] = 0;
    W[28+(NY) * 15] = 0;
    W[28+(NY) * 16] = 0;
    W[28+(NY) * 17] = 0;
    W[28+(NY) * 18] = 0;
    W[28+(NY) * 19] = 0;
    W[28+(NY) * 20] = 0;
    W[28+(NY) * 21] = 0;
    W[28+(NY) * 22] = 0;
    W[28+(NY) * 23] = 0;
    W[28+(NY) * 24] = 0;
    W[28+(NY) * 25] = 0;
    W[28+(NY) * 26] = 0;
    W[28+(NY) * 27] = 0;
    W[28+(NY) * 28] = 1000;
    W[28+(NY) * 29] = 0;
    W[28+(NY) * 30] = 0;
    W[29+(NY) * 0] = 0;
    W[29+(NY) * 1] = 0;
    W[29+(NY) * 2] = 0;
    W[29+(NY) * 3] = 0;
    W[29+(NY) * 4] = 0;
    W[29+(NY) * 5] = 0;
    W[29+(NY) * 6] = 0;
    W[29+(NY) * 7] = 0;
    W[29+(NY) * 8] = 0;
    W[29+(NY) * 9] = 0;
    W[29+(NY) * 10] = 0;
    W[29+(NY) * 11] = 0;
    W[29+(NY) * 12] = 0;
    W[29+(NY) * 13] = 0;
    W[29+(NY) * 14] = 0;
    W[29+(NY) * 15] = 0;
    W[29+(NY) * 16] = 0;
    W[29+(NY) * 17] = 0;
    W[29+(NY) * 18] = 0;
    W[29+(NY) * 19] = 0;
    W[29+(NY) * 20] = 0;
    W[29+(NY) * 21] = 0;
    W[29+(NY) * 22] = 0;
    W[29+(NY) * 23] = 0;
    W[29+(NY) * 24] = 0;
    W[29+(NY) * 25] = 0;
    W[29+(NY) * 26] = 0;
    W[29+(NY) * 27] = 0;
    W[29+(NY) * 28] = 0;
    W[29+(NY) * 29] = 1000;
    W[29+(NY) * 30] = 0;
    W[30+(NY) * 0] = 0;
    W[30+(NY) * 1] = 0;
    W[30+(NY) * 2] = 0;
    W[30+(NY) * 3] = 0;
    W[30+(NY) * 4] = 0;
    W[30+(NY) * 5] = 0;
    W[30+(NY) * 6] = 0;
    W[30+(NY) * 7] = 0;
    W[30+(NY) * 8] = 0;
    W[30+(NY) * 9] = 0;
    W[30+(NY) * 10] = 0;
    W[30+(NY) * 11] = 0;
    W[30+(NY) * 12] = 0;
    W[30+(NY) * 13] = 0;
    W[30+(NY) * 14] = 0;
    W[30+(NY) * 15] = 0;
    W[30+(NY) * 16] = 0;
    W[30+(NY) * 17] = 0;
    W[30+(NY) * 18] = 0;
    W[30+(NY) * 19] = 0;
    W[30+(NY) * 20] = 0;
    W[30+(NY) * 21] = 0;
    W[30+(NY) * 22] = 0;
    W[30+(NY) * 23] = 0;
    W[30+(NY) * 24] = 0;
    W[30+(NY) * 25] = 0;
    W[30+(NY) * 26] = 0;
    W[30+(NY) * 27] = 0;
    W[30+(NY) * 28] = 0;
    W[30+(NY) * 29] = 0;
    W[30+(NY) * 30] = 1000;

    double yref[NY];
    
    yref[0] = 0;
    yref[1] = 0;
    yref[2] = 0;
    yref[3] = 0;
    yref[4] = 0;
    yref[5] = 0;
    yref[6] = 0;
    yref[7] = 0;
    yref[8] = 0;
    yref[9] = 0;
    yref[10] = 0;
    yref[11] = 0;
    yref[12] = 0;
    yref[13] = 0;
    yref[14] = 0;
    yref[15] = 0;
    yref[16] = 0;
    yref[17] = 0;
    yref[18] = 0;
    yref[19] = 0;
    yref[20] = 0;
    yref[21] = 0;
    yref[22] = 0;
    yref[23] = 0;
    yref[24] = 0;
    yref[25] = 0;
    yref[26] = 0;
    yref[27] = 0;
    yref[28] = 0;
    yref[29] = 0;
    yref[30] = 0;

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "W", W);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "yref", yref);
    }


    for (int i = 0; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "nls_y_fun", &cost_y_fun[i]);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "nls_y_fun_jac", &cost_y_fun_jac_ut_xt[i]);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "nls_y_hess", &cost_y_hess[i]);
    }




    // terminal cost


    double yref_e[NYN];
    
    yref_e[0] = 0;
    yref_e[1] = 0;
    yref_e[2] = 0;
    yref_e[3] = 0;
    yref_e[4] = 0;
    yref_e[5] = 0;
    yref_e[6] = 0;
    yref_e[7] = 0;
    yref_e[8] = 0;
    yref_e[9] = 0;
    yref_e[10] = 0;
    yref_e[11] = 0;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "yref", yref_e);

    double W_e[NYN*NYN];
    
    W_e[0+(NYN) * 0] = 100;
    W_e[0+(NYN) * 1] = 0;
    W_e[0+(NYN) * 2] = 0;
    W_e[0+(NYN) * 3] = 0;
    W_e[0+(NYN) * 4] = 0;
    W_e[0+(NYN) * 5] = 0;
    W_e[0+(NYN) * 6] = 0;
    W_e[0+(NYN) * 7] = 0;
    W_e[0+(NYN) * 8] = 0;
    W_e[0+(NYN) * 9] = 0;
    W_e[0+(NYN) * 10] = 0;
    W_e[0+(NYN) * 11] = 0;
    W_e[1+(NYN) * 0] = 0;
    W_e[1+(NYN) * 1] = 100;
    W_e[1+(NYN) * 2] = 0;
    W_e[1+(NYN) * 3] = 0;
    W_e[1+(NYN) * 4] = 0;
    W_e[1+(NYN) * 5] = 0;
    W_e[1+(NYN) * 6] = 0;
    W_e[1+(NYN) * 7] = 0;
    W_e[1+(NYN) * 8] = 0;
    W_e[1+(NYN) * 9] = 0;
    W_e[1+(NYN) * 10] = 0;
    W_e[1+(NYN) * 11] = 0;
    W_e[2+(NYN) * 0] = 0;
    W_e[2+(NYN) * 1] = 0;
    W_e[2+(NYN) * 2] = 100;
    W_e[2+(NYN) * 3] = 0;
    W_e[2+(NYN) * 4] = 0;
    W_e[2+(NYN) * 5] = 0;
    W_e[2+(NYN) * 6] = 0;
    W_e[2+(NYN) * 7] = 0;
    W_e[2+(NYN) * 8] = 0;
    W_e[2+(NYN) * 9] = 0;
    W_e[2+(NYN) * 10] = 0;
    W_e[2+(NYN) * 11] = 0;
    W_e[3+(NYN) * 0] = 0;
    W_e[3+(NYN) * 1] = 0;
    W_e[3+(NYN) * 2] = 0;
    W_e[3+(NYN) * 3] = 100;
    W_e[3+(NYN) * 4] = 0;
    W_e[3+(NYN) * 5] = 0;
    W_e[3+(NYN) * 6] = 0;
    W_e[3+(NYN) * 7] = 0;
    W_e[3+(NYN) * 8] = 0;
    W_e[3+(NYN) * 9] = 0;
    W_e[3+(NYN) * 10] = 0;
    W_e[3+(NYN) * 11] = 0;
    W_e[4+(NYN) * 0] = 0;
    W_e[4+(NYN) * 1] = 0;
    W_e[4+(NYN) * 2] = 0;
    W_e[4+(NYN) * 3] = 0;
    W_e[4+(NYN) * 4] = 10;
    W_e[4+(NYN) * 5] = 0;
    W_e[4+(NYN) * 6] = 0;
    W_e[4+(NYN) * 7] = 0;
    W_e[4+(NYN) * 8] = 0;
    W_e[4+(NYN) * 9] = 0;
    W_e[4+(NYN) * 10] = 0;
    W_e[4+(NYN) * 11] = 0;
    W_e[5+(NYN) * 0] = 0;
    W_e[5+(NYN) * 1] = 0;
    W_e[5+(NYN) * 2] = 0;
    W_e[5+(NYN) * 3] = 0;
    W_e[5+(NYN) * 4] = 0;
    W_e[5+(NYN) * 5] = 10;
    W_e[5+(NYN) * 6] = 0;
    W_e[5+(NYN) * 7] = 0;
    W_e[5+(NYN) * 8] = 0;
    W_e[5+(NYN) * 9] = 0;
    W_e[5+(NYN) * 10] = 0;
    W_e[5+(NYN) * 11] = 0;
    W_e[6+(NYN) * 0] = 0;
    W_e[6+(NYN) * 1] = 0;
    W_e[6+(NYN) * 2] = 0;
    W_e[6+(NYN) * 3] = 0;
    W_e[6+(NYN) * 4] = 0;
    W_e[6+(NYN) * 5] = 0;
    W_e[6+(NYN) * 6] = 10;
    W_e[6+(NYN) * 7] = 0;
    W_e[6+(NYN) * 8] = 0;
    W_e[6+(NYN) * 9] = 0;
    W_e[6+(NYN) * 10] = 0;
    W_e[6+(NYN) * 11] = 0;
    W_e[7+(NYN) * 0] = 0;
    W_e[7+(NYN) * 1] = 0;
    W_e[7+(NYN) * 2] = 0;
    W_e[7+(NYN) * 3] = 0;
    W_e[7+(NYN) * 4] = 0;
    W_e[7+(NYN) * 5] = 0;
    W_e[7+(NYN) * 6] = 0;
    W_e[7+(NYN) * 7] = 10;
    W_e[7+(NYN) * 8] = 0;
    W_e[7+(NYN) * 9] = 0;
    W_e[7+(NYN) * 10] = 0;
    W_e[7+(NYN) * 11] = 0;
    W_e[8+(NYN) * 0] = 0;
    W_e[8+(NYN) * 1] = 0;
    W_e[8+(NYN) * 2] = 0;
    W_e[8+(NYN) * 3] = 0;
    W_e[8+(NYN) * 4] = 0;
    W_e[8+(NYN) * 5] = 0;
    W_e[8+(NYN) * 6] = 0;
    W_e[8+(NYN) * 7] = 0;
    W_e[8+(NYN) * 8] = 1000;
    W_e[8+(NYN) * 9] = 0;
    W_e[8+(NYN) * 10] = 0;
    W_e[8+(NYN) * 11] = 0;
    W_e[9+(NYN) * 0] = 0;
    W_e[9+(NYN) * 1] = 0;
    W_e[9+(NYN) * 2] = 0;
    W_e[9+(NYN) * 3] = 0;
    W_e[9+(NYN) * 4] = 0;
    W_e[9+(NYN) * 5] = 0;
    W_e[9+(NYN) * 6] = 0;
    W_e[9+(NYN) * 7] = 0;
    W_e[9+(NYN) * 8] = 0;
    W_e[9+(NYN) * 9] = 1000;
    W_e[9+(NYN) * 10] = 0;
    W_e[9+(NYN) * 11] = 0;
    W_e[10+(NYN) * 0] = 0;
    W_e[10+(NYN) * 1] = 0;
    W_e[10+(NYN) * 2] = 0;
    W_e[10+(NYN) * 3] = 0;
    W_e[10+(NYN) * 4] = 0;
    W_e[10+(NYN) * 5] = 0;
    W_e[10+(NYN) * 6] = 0;
    W_e[10+(NYN) * 7] = 0;
    W_e[10+(NYN) * 8] = 0;
    W_e[10+(NYN) * 9] = 0;
    W_e[10+(NYN) * 10] = 1000;
    W_e[10+(NYN) * 11] = 0;
    W_e[11+(NYN) * 0] = 0;
    W_e[11+(NYN) * 1] = 0;
    W_e[11+(NYN) * 2] = 0;
    W_e[11+(NYN) * 3] = 0;
    W_e[11+(NYN) * 4] = 0;
    W_e[11+(NYN) * 5] = 0;
    W_e[11+(NYN) * 6] = 0;
    W_e[11+(NYN) * 7] = 0;
    W_e[11+(NYN) * 8] = 0;
    W_e[11+(NYN) * 9] = 0;
    W_e[11+(NYN) * 10] = 0;
    W_e[11+(NYN) * 11] = 1000;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "W", W_e);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "nls_y_fun", &cost_y_e_fun);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "nls_y_fun_jac", &cost_y_e_fun_jac_ut_xt);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "nls_y_hess", &cost_y_e_hess);



    /**** Constraints ****/

    // bounds for initial stage

    // x0
    int idxbx0[8];
    
    idxbx0[0] = 0;
    idxbx0[1] = 1;
    idxbx0[2] = 2;
    idxbx0[3] = 3;
    idxbx0[4] = 4;
    idxbx0[5] = 5;
    idxbx0[6] = 6;
    idxbx0[7] = 7;

    double lbx0[8];
    double ubx0[8];
    
    lbx0[0] = -0.18581072895085443;
    ubx0[0] = 0.014189271049145594;
    lbx0[1] = -0.4397201660005633;
    ubx0[1] = -0.23972016600056328;
    lbx0[2] = -0.21410356815053644;
    ubx0[2] = -0.014103568150536427;
    lbx0[3] = 0.049708482412232635;
    ubx0[3] = 0.24970848241223265;
    lbx0[4] = -31.41592653589793;
    ubx0[4] = 31.41592653589793;
    lbx0[5] = -31.41592653589793;
    ubx0[5] = 31.41592653589793;
    lbx0[6] = -31.41592653589793;
    ubx0[6] = 31.41592653589793;
    lbx0[7] = -31.41592653589793;
    ubx0[7] = 31.41592653589793;

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "idxbx", idxbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "ubx", ubx0);



    /* constraints that are the same for initial and intermediate */



    // u
    int idxbu[NBU];
    
    idxbu[0] = 0;
    idxbu[1] = 1;
    idxbu[2] = 2;
    idxbu[3] = 3;
    idxbu[4] = 4;
    idxbu[5] = 5;
    idxbu[6] = 6;
    idxbu[7] = 7;
    idxbu[8] = 8;
    idxbu[9] = 9;
    idxbu[10] = 10;
    idxbu[11] = 11;
    idxbu[12] = 12;
    idxbu[13] = 13;
    idxbu[14] = 14;
    idxbu[15] = 15;
    idxbu[16] = 16;
    idxbu[17] = 17;
    idxbu[18] = 18;
    double lbu[NBU];
    double ubu[NBU];
    
    lbu[0] = 0;
    ubu[0] = 1;
    lbu[1] = 0;
    ubu[1] = 1;
    lbu[2] = 0;
    ubu[2] = 1;
    lbu[3] = 0;
    ubu[3] = 1;
    lbu[4] = 0;
    ubu[4] = 1;
    lbu[5] = 0;
    ubu[5] = 1;
    lbu[6] = 0;
    ubu[6] = 1;
    lbu[7] = 0;
    ubu[7] = 1;
    lbu[8] = 0;
    ubu[8] = 1;
    lbu[9] = 0;
    ubu[9] = 1;
    lbu[10] = 0;
    ubu[10] = 1;
    lbu[11] = 0;
    ubu[11] = 1;
    lbu[12] = 0;
    ubu[12] = 1;
    lbu[13] = 0;
    ubu[13] = 1;
    lbu[14] = 0;
    ubu[14] = 1;
    lbu[15] = 0;
    ubu[15] = 1;
    lbu[16] = 0;
    ubu[16] = 1;
    lbu[17] = 0;
    ubu[17] = 1;
    lbu[18] = 0;
    ubu[18] = 1;

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "idxbu", idxbu);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lbu", lbu);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "ubu", ubu);
    }











    // x
    int idxbx[NBX];
    
    idxbx[0] = 0;
    idxbx[1] = 1;
    idxbx[2] = 2;
    idxbx[3] = 3;
    idxbx[4] = 4;
    idxbx[5] = 5;
    idxbx[6] = 6;
    idxbx[7] = 7;
    double lbx[NBX];
    double ubx[NBX];
    
    lbx[0] = -3.141592653589793;
    ubx[0] = 3.141592653589793;
    lbx[1] = -3.141592653589793;
    ubx[1] = 3.141592653589793;
    lbx[2] = -3.141592653589793;
    ubx[2] = 3.141592653589793;
    lbx[3] = -6.283185307179586;
    ubx[3] = 6.283185307179586;
    lbx[4] = -31.41592653589793;
    ubx[4] = 31.41592653589793;
    lbx[5] = -31.41592653589793;
    ubx[5] = 31.41592653589793;
    lbx[6] = -31.41592653589793;
    ubx[6] = 31.41592653589793;
    lbx[7] = -31.41592653589793;
    ubx[7] = 31.41592653589793;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "idxbx", idxbx);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lbx", lbx);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "ubx", ubx);
    }








    /* terminal constraints */

    // set up bounds for last stage
    // x
    int idxbx_e[NBXN];
    
    idxbx_e[0] = 0;
    idxbx_e[1] = 1;
    idxbx_e[2] = 2;
    idxbx_e[3] = 3;
    idxbx_e[4] = 4;
    idxbx_e[5] = 5;
    idxbx_e[6] = 6;
    idxbx_e[7] = 7;
    double lbx_e[NBXN];
    double ubx_e[NBXN];
    
    lbx_e[0] = -3.141592653589793;
    ubx_e[0] = 3.141592653589793;
    lbx_e[1] = -3.141592653589793;
    ubx_e[1] = 3.141592653589793;
    lbx_e[2] = -3.141592653589793;
    ubx_e[2] = 3.141592653589793;
    lbx_e[3] = -6.283185307179586;
    ubx_e[3] = 6.283185307179586;
    lbx_e[4] = -31.41592653589793;
    ubx_e[4] = 31.41592653589793;
    lbx_e[5] = -31.41592653589793;
    ubx_e[5] = 31.41592653589793;
    lbx_e[6] = -31.41592653589793;
    ubx_e[6] = 31.41592653589793;
    lbx_e[7] = -31.41592653589793;
    ubx_e[7] = 31.41592653589793;
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "idxbx", idxbx_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "lbx", lbx_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "ubx", ubx_e);
















    /************************************************
    *  opts
    ************************************************/

    nlp_opts = ocp_nlp_solver_opts_create(nlp_config, nlp_dims);


    int num_steps_val = 1;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_num_steps", &num_steps_val);

    int ns_val = 4;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_num_stages", &ns_val);

    int newton_iter_val = 5;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_newton_iter", &newton_iter_val);

    bool tmp_bool = false;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_jac_reuse", &tmp_bool);

    double nlp_solver_step_length = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "step_length", &nlp_solver_step_length);

    double levenberg_marquardt = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "levenberg_marquardt", &levenberg_marquardt);

    /* options QP solver */
    int qp_solver_cond_N;
    // NOTE: there is no condensing happening here!
    qp_solver_cond_N = N;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_cond_N", &qp_solver_cond_N);


    int qp_solver_iter_max = 50;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_iter_max", &qp_solver_iter_max);
    // set SQP specific options
    double nlp_solver_tol_stat = 0.00000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "tol_stat", &nlp_solver_tol_stat);

    double nlp_solver_tol_eq = 0.0000000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "tol_eq", &nlp_solver_tol_eq);

    double nlp_solver_tol_ineq = 0.000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "tol_ineq", &nlp_solver_tol_ineq);

    double nlp_solver_tol_comp = 0.0000000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "tol_comp", &nlp_solver_tol_comp);

    int nlp_solver_max_iter = 30;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "max_iter", &nlp_solver_max_iter);

    int initialize_t_slacks = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "initialize_t_slacks", &initialize_t_slacks);

    int print_level = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "print_level", &print_level);


    int ext_cost_num_hess = 0;


    /* out */
    nlp_out = ocp_nlp_out_create(nlp_config, nlp_dims);

    // initialize primal solution
    double x0[8];

    // initialize with x0
    
    x0[0] = -0.18581072895085443;
    x0[1] = -0.4397201660005633;
    x0[2] = -0.21410356815053644;
    x0[3] = 0.049708482412232635;
    x0[4] = -31.41592653589793;
    x0[5] = -31.41592653589793;
    x0[6] = -31.41592653589793;
    x0[7] = -31.41592653589793;


    double u0[NU];
    
    u0[0] = 0.0;
    u0[1] = 0.0;
    u0[2] = 0.0;
    u0[3] = 0.0;
    u0[4] = 0.0;
    u0[5] = 0.0;
    u0[6] = 0.0;
    u0[7] = 0.0;
    u0[8] = 0.0;
    u0[9] = 0.0;
    u0[10] = 0.0;
    u0[11] = 0.0;
    u0[12] = 0.0;
    u0[13] = 0.0;
    u0[14] = 0.0;
    u0[15] = 0.0;
    u0[16] = 0.0;
    u0[17] = 0.0;
    u0[18] = 0.0;

    for (int i = 0; i < N; i++)
    {
        // x0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "x", x0);
        // u0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "u", u0);
    }
    ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, N, "x", x0);
    
    nlp_solver = ocp_nlp_solver_create(nlp_config, nlp_dims, nlp_opts);




    status = ocp_nlp_precompute(nlp_solver, nlp_in, nlp_out);

    if (status != ACADOS_SUCCESS)
    {
        printf("\nocp_precompute failed!\n\n");
        exit(1);
    }

    return status;
}


int acados_update_params(int stage, double *p, int np)
{
    int solver_status = 0;

    int casadi_np = 0;
    if (casadi_np != np) {
        printf("acados_update_params: trying to set %i parameters for external functions."
            " External function has %i parameters. Exiting.\n", np, casadi_np);
        exit(1);
    }

    return solver_status;
}



int acados_solve()
{
    // solve NLP 
    int solver_status = ocp_nlp_solve(nlp_solver, nlp_in, nlp_out);

    return solver_status;
}


int acados_free()
{
    // free memory
    ocp_nlp_solver_opts_destroy(nlp_opts);
    ocp_nlp_in_destroy(nlp_in);
    ocp_nlp_out_destroy(nlp_out);
    ocp_nlp_solver_destroy(nlp_solver);
    ocp_nlp_dims_destroy(nlp_dims);
    ocp_nlp_config_destroy(nlp_config);
    ocp_nlp_plan_destroy(nlp_solver_plan);

    /* free external function */
    // dynamics
    for (int i = 0; i < 7; i++)
    {
        external_function_param_casadi_free(&impl_dae_fun[i]);
        external_function_param_casadi_free(&impl_dae_fun_jac_x_xdot_z[i]);
        external_function_param_casadi_free(&impl_dae_jac_x_xdot_u_z[i]);
    }
    free(impl_dae_fun);
    free(impl_dae_fun_jac_x_xdot_z);
    free(impl_dae_jac_x_xdot_u_z);

    // cost
    for (int i = 0; i < 7; i++)
    {
        external_function_param_casadi_free(&cost_y_fun[i]);
        external_function_param_casadi_free(&cost_y_fun_jac_ut_xt[i]);
        external_function_param_casadi_free(&cost_y_hess[i]);
    }
    free(cost_y_fun);
    free(cost_y_fun_jac_ut_xt);
    free(cost_y_hess);
    external_function_param_casadi_free(&cost_y_e_fun);
    external_function_param_casadi_free(&cost_y_e_fun_jac_ut_xt);
    external_function_param_casadi_free(&cost_y_e_hess);

    // constraints

    return 0;
}

ocp_nlp_in * acados_get_nlp_in() { return  nlp_in; }
ocp_nlp_out * acados_get_nlp_out() { return  nlp_out; }
ocp_nlp_solver * acados_get_nlp_solver() { return  nlp_solver; }
ocp_nlp_config * acados_get_nlp_config() { return  nlp_config; }
void * acados_get_nlp_opts() { return  nlp_opts; }
ocp_nlp_dims * acados_get_nlp_dims() { return  nlp_dims; }
ocp_nlp_plan * acados_get_nlp_plan() { return  nlp_solver_plan; }


void acados_print_stats()
{
    int sqp_iter, stat_m, stat_n, tmp_int;
    ocp_nlp_get(nlp_config, nlp_solver, "sqp_iter", &sqp_iter);
    ocp_nlp_get(nlp_config, nlp_solver, "stat_n", &stat_n);
    ocp_nlp_get(nlp_config, nlp_solver, "stat_m", &stat_m);

    
    double stat[300];
    ocp_nlp_get(nlp_config, nlp_solver, "statistics", stat);

    int nrow = sqp_iter+1 < stat_m ? sqp_iter+1 : stat_m;

    printf("iter\tres_stat\tres_eq\t\tres_ineq\tres_comp\tqp_stat\tqp_iter\n");
    for (int i = 0; i < nrow; i++)
    {
        for (int j = 0; j < stat_n + 1; j++)
        {
            if (j == 0 || j > 4)
            {
                tmp_int = (int) stat[i + j * nrow];
                printf("%d\t", tmp_int);
            }
            else
            {
                printf("%e\t", stat[i + j * nrow]);
            }
        }
        printf("\n");
    }
}