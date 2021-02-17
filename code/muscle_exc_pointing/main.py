"""
This is a basic example on how to use biorbd model driven by muscle to perform an optimal reaching task.
The arms must reach a marker placed upward in front while minimizing the muscles activity

Please note that using show_meshes=True in the animator may be long due to the creation of a huge CasADi graph of the
mesh points.
"""

import biorbd
import numpy as np
from bioptim import (
    OptimalControlProgram,
    ObjectiveList,
    ObjectiveFcn,
    ConstraintList,
    ConstraintFcn,
    Node,
    DynamicsList,
    Data,
    DynamicsFcn,
    Bounds,
    BoundsList,
    QAndQDotBounds,
    InitialGuessList,
    ShowResult,
    OdeSolver,
    Simulate,
    Solver,
)

def compute_error_single_shooting(ocp, sol, duration):
    if ocp.nlp[0].tf < duration:
        raise ValueError(f'Single shooting integration duration must be smaller than ocp duration :{ocp.nlp[0].tf} s')
    data = Data.get_data(ocp, sol)
    data_ss = Data.get_data(ocp, Simulate.from_solve(ocp, sol, True)['x'])
    sn_1s = int(ocp.nlp[0].ns/ocp.nlp[0].tf*duration) # shooting node at {duration} second
    err = []
    for key in data[0].keys():
        key_shape = data[0][key].shape
        err += [np.sqrt(np.mean((data[0][key][:, sn_1s]-data_ss[0][key][:, sn_1s])**2))]
    return np.mean(err)

def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    use_sx: bool,
    use_exc: bool,
    weights: np.array(4),
    ode_solver: OdeSolver = OdeSolver.RK4,
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

    biorbd_model = biorbd.Model(biorbd_model_path)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=weights[0])
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, weight=weights[1])
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=weights[2])
    objective_functions.add(
        ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS, first_marker_idx=0, second_marker_idx=1, weight=weights[3])

    # Dynamics
    dynamics = DynamicsList()
    if use_exc:
        dynamics.add(DynamicsFcn.MUSCLE_EXCITATIONS_AND_TORQUE_DRIVEN)
    else:
        dynamics.add(DynamicsFcn.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))

    # Add muscle to the bounds
    if use_exc:
        activation_min, activation_max, activation_init = 0, 1, 0
        x_bounds[0].concatenate(
            Bounds([activation_min] * biorbd_model.nbMuscles(), [activation_max] * biorbd_model.nbMuscles())
        )

    # Force initial position
    if use_sx:
        if use_exc:
            x_bounds[0][:, 0] = [1.24, 1.55, 0, 0] + [0]*biorbd_model.nbMuscles()
        else:
            x_bounds[0][:, 0] = [1.24, 1.55, 0, 0]
    else:
        if use_exc:
            x_bounds[0][:, 0] = [1.0, 1.3, 0, 0] + [0] * biorbd_model.nbMuscles()
        else:
            x_bounds[0][:, 0] = [1.0, 1.3, 0, 0]
    # Initial guess
    x_init = InitialGuessList()
    if use_exc:
        init_state = [1.57] * biorbd_model.nbQ() + [0] * biorbd_model.nbQdot() + [0.5] * biorbd_model.nbMuscles()
    else:
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
        ode_solver=ode_solver,
        n_threads=8,
        use_sx=use_sx,
    )


if __name__ == "__main__":
    """
    Prepare and solve and animate a reaching task ocp
    """
    use_IPOPT = True
    use_exc = False
    weights = np.array([100, 1, 1, 100000])
    ocp = prepare_ocp(biorbd_model_path="arm26.bioMod", final_time=2, n_shooting=50,
                      use_exc=use_exc, use_sx=not use_IPOPT, weights=weights)

    # --- Solve the program --- #
    if use_IPOPT:
        opts = {"linear_solver": "ma57", "hessian_approximation": "exact"}
        solver = Solver.IPOPT
    else:
        opts = {"sim_method_num_steps": 5, "tol": 1e-8, "integrator_type": "ERK", "hessian_approx": "GAUSS_NEWTON"}
        solver = Solver.ACADOS
    sol = ocp.solve(solver=solver, solver_options=opts, show_online_optim=False)

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.objective_functions()
    data_opt = Data.get_data(ocp,sol['x'])
    single_shooting_duration = 1
    ss_err = compute_error_single_shooting(ocp, sol, 1)
    print("*********************************************")
    print(f"Problem solved with {solver.value}")
    print(f"Solving time : {sol['time_tot']}s")
    print(f"Single shooting error at {single_shooting_duration}s= {ss_err}")
    # result.graphs()
    result.animate(show_meshes=True, background_color=(1, 1, 1),
                   show_local_ref_frame=False, show_global_center_of_mass=False,
                   show_segments_center_of_mass=False,)
