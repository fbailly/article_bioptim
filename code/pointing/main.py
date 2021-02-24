"""
This is a basic example on how to use biorbd model driven by muscle to perform an optimal reaching task.
The arms must reach a marker placed upward in front while minimizing the muscles activity

Please note that using show_meshes=True in the animator may be long due to the creation of a huge CasADi graph of the
mesh points.
"""

import biorbd
import numpy as np
from time import time
from bioptim import (
    OptimalControlProgram,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsList,
    DynamicsFcn,
    Bounds,
    BoundsList,
    QAndQDotBounds,
    InitialGuessList,
    OdeSolver,
    Solver,
    Shooting,
)


def compute_error_single_shooting(sol, duration):
    sol_merged = sol.merge_phases()

    if sol_merged.phase_time[-1] < duration:
        raise ValueError(
            f'Single shooting integration duration must be smaller than ocp duration :{sol_merged.phase_time[-1]} s')

    trans_idx = []
    rot_idx = []
    for i in range(sol.ocp.nlp[0].model.nbQ()):
        if sol.ocp.nlp[0].model.nameDof()[i].to_string()[-4:-1] == 'Rot':
            rot_idx += [i]
        else:
            trans_idx += [i]
    rot_idx = np.array(rot_idx)
    trans_idx = np.array(trans_idx)

    sol_int = sol.integrate(shooting_type=Shooting.SINGLE_CONTINUOUS, merge_phases=True, keepdims=True)
    sn_1s = int(sol_int.ns[0] / sol_int.phase_time[-1] * duration)  # shooting node at {duration} second
    if len(rot_idx) > 0:
        single_shoot_error_r = np.sqrt(
            np.mean((sol_int.states['q'][rot_idx, sn_1s] - sol_merged.states['q'][rot_idx, sn_1s]) ** 2)) \
                                    * 180 / np.pi
    else:
        single_shoot_error_r = 'N.A.'
    if len(trans_idx) > 0:
        single_shoot_error_t = np.sqrt(np.mean(
            (sol_int.states['q'][trans_idx, 5 * sn_1s] - sol_merged.states['q'][trans_idx, sn_1s]) ** 2)) / 1000
    else:
       single_shoot_error_t = 'N.A.'
    return single_shoot_error_t, single_shoot_error_r

def prepare_ocp(
    biorbd_model: biorbd.Model,
    final_time: float,
    n_shooting: int,
    use_sx: bool,
    weights: np.array(4),
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod
    final_time: float
        The time at the final node
    n_shooting: int
        The number of shooting points
    weight: float
        The weight applied to the SUPERIMPOSE_MARKERS final objective function. The bigger this number is, the greater
        the model will try to reach the marker. This is in relation with the other objective functions
    ode_solver: OdeSolver
        The ode solver to use

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """


    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=weights[0])
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, weight=weights[1])
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=weights[2])
    objective_functions.add(
        ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS, first_marker_idx=0, second_marker_idx=1, weight=weights[3])

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))

    # Force initial position
    if use_sx:
        x_bounds[0][:, 0] = [1.24, 1.55, 0, 0]
    else:
        x_bounds[0][:, 0] = [1.0, 1.3, 0, 0]
    # Initial guess
    x_init = InitialGuessList()
    init_state = [1.57] * biorbd_model.nbQ() + [0] * biorbd_model.nbQdot()
    x_init.add(init_state)

    # Define control path constraint
    muscle_min, muscle_max, muscle_init = 0, 1, 0.5
    tau_min, tau_max, tau_init = -10, 10, 0
    u_bounds = BoundsList()
    u_bounds.add(
        [tau_min] * biorbd_model.nbGeneralizedTorque() + [muscle_min] * biorbd_model.nbMuscleTotal(),
        [tau_max] * biorbd_model.nbGeneralizedTorque() + [muscle_max] * biorbd_model.nbMuscleTotal(),
    )
    # u_bounds[0][:, 0] = [0] * biorbd_model.nbGeneralizedTorque() + [0] * biorbd_model.nbMuscleTotal()
    u_init = InitialGuessList()
    u_init.add([tau_init] * biorbd_model.nbGeneralizedTorque() + [muscle_init] * biorbd_model.nbMuscleTotal())
    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        n_threads=8,
        use_sx=use_sx,
    )

def generate_table(out):
    model_path = "/".join(__file__.split("/")[:-1]) + "/arm26.bioMod"
    biorbd_model_ip = biorbd.Model(model_path)

    # IPOPT
    use_IPOPT = True
    weights = np.array([100, 1, 1, 100000])
    ocp = prepare_ocp(biorbd_model=biorbd_model_ip, final_time=2, n_shooting=50,
                      use_sx=not use_IPOPT, weights=weights)
    opts = {"linear_solver": "ma57", "hessian_approximation": "exact"}
    solver = Solver.IPOPT

    # --- Solve the program --- #
    tic = time()
    sol = ocp.solve(solver=solver, solver_options=opts,)
    toc = time() - tic
    sol_merged = sol.merge_phases()

    out.nx = sol_merged.states["all"].shape[0]
    out.nu = sol_merged.controls["all"].shape[0]
    out.ns = sol_merged.ns[0]
    out.solver.append(out.Solver("Ipopt"))
    out.solver[0].n_iteration = sol.iterations
    out.solver[0].cost = sol.cost
    out.solver[0].convergence_time = toc
    out.solver[0].compute_error_single_shooting(sol, 1)

    # ACADOS
    use_IPOPT = False
    biorbd_model_ac = biorbd.Model(model_path)
    ocp = prepare_ocp(biorbd_model=biorbd_model_ac, final_time=2, n_shooting=50,
                      use_sx=not use_IPOPT, weights=weights)
    opts = {"sim_method_num_steps": 5, "tol": 1e-8, "integrator_type": "ERK", "hessian_approx": "GAUSS_NEWTON"}
    solver = Solver.ACADOS

    # --- Solve the program --- #
    sol = ocp.solve(solver=solver, solver_options=opts,)

    out.solver.append(out.Solver("Acados"))
    out.solver[1].n_iteration = sol.iterations
    out.solver[1].cost = sol.cost
    out.solver[1].convergence_time = sol.time_to_optimize
    out.solver[1].compute_error_single_shooting(sol, 1)


if __name__ == "__main__":
    """
    Prepare and solve and animate a reaching task ocp
    """
    use_IPOPT = True
    weights = np.array([100, 1, 1, 100000])
    biorbd_model = biorbd.Model("arm26.bioMod")
    ocp = prepare_ocp(biorbd_model=biorbd_model, final_time=2, n_shooting=50,
                      use_sx=not use_IPOPT, weights=weights)

    # --- Solve the program --- #
    if use_IPOPT:
        opts = {"linear_solver": "ma57", "hessian_approximation": "exact"}
        solver = Solver.IPOPT
    else:
        opts = {"sim_method_num_steps": 5, "tol": 1e-8, "integrator_type": "ERK", "hessian_approx": "GAUSS_NEWTON"}
        solver = Solver.ACADOS
    sol = ocp.solve(solver=solver, solver_options=opts, show_online_optim=False)

    # --- Show results --- #
    sol.print()
    single_shooting_duration = 1
    ss_err_t, ss_err_r = compute_error_single_shooting(sol, 1)
    print("*********************************************")
    print(f"Problem solved with {solver.value}")
    print(f"Solving time : {sol.time_to_optimize}s")
    print(f"Single shooting error at {single_shooting_duration}s in translation (mm)= {ss_err_t}")
    print(f"Single shooting error at {single_shooting_duration}s in rotation (°)= {ss_err_r}")
    # result.graphs()
    sol.animate(show_meshes=True, background_color=(1, 1, 1),
                   show_local_ref_frame=False, show_global_center_of_mass=False,
                   show_segments_center_of_mass=False,)
