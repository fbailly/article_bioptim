"""
This is a basic example on how to use moving horizon estimation for muscle force estimation using a 4 degree of freedom
(Dof) Arm model actuated by 19 hill-type muscles. controls are muscle activations.
Model joint angles are tracked to match with reference ones, muscle activations are minimized.
"""

import biorbd
from time import time
import numpy as np
import pickle
import scipy.io as sio
import bioviz
from math import ceil
from casadi import MX, Function
from bioptim import (
    OptimalControlProgram,
    Solution,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsList,
    DynamicsFcn,
    BoundsList,
    QAndQDotBounds,
    InitialGuess,
    Solver,
    InterpolationType,
)


def muscle_forces(q: MX, qdot: MX, a: MX, controls: MX, model: biorbd.Model, use_activation=True):
    """
    Compute muscle force
    Parameters
    ----------
    q: MX
        Symbolic value of joint angle
    qdot: MX
        Symbolic value of joint velocity
    controls: int
        Symbolic value of activations
    model: biorbd.Model
        biorbd model build with the bioMod
    use_activation: bool
        True if activation drive False if excitation driven
    Returns
    -------
    List of muscle forces
    """
    muscle_states = biorbd.VecBiorbdMuscleState(model.nbMuscles())
    for k in range(model.nbMuscles()):
        if use_activation:
            muscle_states[k].setActivation(controls[k])
        else:
            muscle_states[k].setActivation(a[k])
            muscle_states[k].setExcitation(controls[k])
    muscle_forces = model.muscleForces(muscle_states, q, qdot).to_mx()
    return muscle_forces


def force_func(biorbd_model: biorbd.Model, use_activation=True):
    """
    Define Casadi function to use muscle_forces
    Parameters
    ----------
    model: biorbd.Model
        biorbd model build with the bioMod
    use_activation: bool
        True if activation drive False if excitation driven
    """
    qMX = MX.sym("qMX", biorbd_model.nbQ(), 1)
    dqMX = MX.sym("dqMX", biorbd_model.nbQ(), 1)
    aMX = MX.sym("aMX", biorbd_model.nbMuscles(), 1)
    uMX = MX.sym("uMX", biorbd_model.nbMuscles(), 1)
    return Function(
        "MuscleForce",
        [qMX, dqMX, aMX, uMX],
        [muscle_forces(qMX, dqMX, aMX, uMX, biorbd_model, use_activation=use_activation)],
        ["qMX", "dqMX", "aMX", "uMX"],
        ["Force"],
    ).expand()


def generate_noise(biorbd_model, q: np.array, q_noise_lvl: float):
    """
    Generate random Centered Gaussian noise apply on joint angles
    Parameters
    ----------
    model: biorbd.Model
        biorbd model build with the bioMod
    q: np.array
        Array of reference joint angles
    q_noise_lvl: float
        Standard deviation value in percent
    Returns
    ---------
    Array of noisy joint angles
    """
    n_q = biorbd_model.nbQ()
    q_noise = np.ndarray((n_q, q.shape[1]))
    for i in range(n_q):
        noise = np.random.normal(0, abs(q_noise_lvl * q[i, :] / 100))
        q_noise[i, :] = q[i, :] + noise
    return q_noise


def warm_start_mhe(sol):
    """
    Ensures the problems continuity
    Parameters
    ----------
    ocp: ocp
        the optimal control program
    sol: sol
        the solutions of the previous problem
    Returns
    ---------
    Initial states and controls for next problem (x0, u0)
    States and controls to save as solution (x_out, u_out)
    """
    x = sol.states['all']
    u = sol.controls['all']

    x0 = np.hstack((x[:, 1:], np.tile(x[:, [-1]], 1)))  # discard oldest estimate of the window, duplicates youngest
    u0 = u[:, :-1]
    x_out = x[:, 0]
    u_out = u[:, 0]
    return x0, u0, x_out, u_out


def define_objective(q: np.array, iter: int, rt_ratio: int, Ns_mhe: int, biorbd_model: biorbd.Model, use_noise=True):
    """
    Define the objective function for the ocp
    Parameters
    ----------
    q: np.array
        State to track
    iter: int
        Current iteration
    rt_ratio: int
        Value of the reference data ratio to send to the estimator
    Ns_mhe: int
        Size of the windows
    biorbd_model: biorbd.Model
        biorbd model build with the bioMod
    use_noise: bool
        True if noisy reference data
    Returns
    ---------
    The objective function
    """
    objectives = ObjectiveList()
    if use_noise is not True:
        weight = {"track_state": 1000000, "min_act": 1000, "min_dq": 10, "min_q": 10}
    else:
        weight = {"track_state": 1000, "min_act": 100, "min_dq": 100, "min_q": 10}
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=weight["min_act"])
    objectives.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        weight=weight["min_dq"],
        index=np.array(range(biorbd_model.nbQ(), biorbd_model.nbQ() * 2)),
    )
    objectives.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        weight=weight["min_q"],
        index=np.array(range(biorbd_model.nbQ())),
    )
    q = q[:, ::rt_ratio]
    objectives.add(
        ObjectiveFcn.Lagrange.TRACK_STATE,
        weight=weight["track_state"],
        target=q[:, iter : (Ns_mhe + 1 + iter)],
        index=range(biorbd_model.nbQ()),
    )
    return objectives


def prepare_ocp(biorbd_model: biorbd.Model, final_time: float, number_shooting_points: int):
    """
    Prepare to build a blank ocp witch will be update several times
    Parameters
    ----------
    biorbd_model: biorbd.Model
        biorbd model build with the bioMod
    final_time: float
        The time at the final node
    n_shooting: int
        The number of shooting points
    Returns
    -------
    The blank OptimalControlProgram
    """
    # Add objective functions
    objective_functions = ObjectiveList()

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.MUSCLE_ACTIVATIONS_DRIVEN)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([0] * biorbd_model.nbMuscles(), [1] * biorbd_model.nbMuscles())

    x_init = InitialGuess([0] * biorbd_model.nbQ() * 2)
    u_init = InitialGuess([0] * biorbd_model.nbMuscles())

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        number_shooting_points,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        use_sx=True,
    )


def prepare_short_ocp(biorbd_model: biorbd.Model, final_time: float, number_shooting_points: int):
    """
    Prepare to build a blank short ocp to use single shooting bioptim function
    Parameters
    ----------
    biorbd_model: biorbd.Model
        biorbd model build with the bioMod
    final_time: float
        The time at the final node
    n_shooting: int
        The number of shooting points
    Returns
    -------
    The blank OptimalControlProgram
    """
    # Add objective functions
    objective_functions = ObjectiveList()

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.MUSCLE_ACTIVATIONS_DRIVEN)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([0] * biorbd_model.nbMuscles(), [1] * biorbd_model.nbMuscles())

    x_init = InitialGuess([0] * biorbd_model.nbQ() * 2)
    u_init = InitialGuess([0] * biorbd_model.nbMuscles())

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        number_shooting_points,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        use_sx=True,
    )


def generate_table(out):
    model_path = "/".join(__file__.split("/")[:-1]) + "/arm_wt_rot_scap.bioMod"
    biorbd_model = biorbd.Model(model_path)

    # --- Prepare and solve MHE --- #
    T = 8
    Ns = 800
    Ns_mhe = 7
    rt_ratio = 3
    T_mhe = T / (Ns / rt_ratio) * Ns_mhe

    # --- Prepare reference data --- #
    with open(f"MHE_example/Data/sim_ac_8000ms_800sn_REACH2_co_level_0_step5_ERK.bob", "rb") as file:
        data = pickle.load(file)
    states = data["data"][0]
    controls = data["data"][1]
    q_ref, dq_ref, u_ref = states["q"], states["qdot"], controls["muscles"]

    ocp = prepare_ocp(biorbd_model=biorbd_model, final_time=T_mhe, number_shooting_points=Ns_mhe)
    Q_noise = 5
    x_ref = np.concatenate((generate_noise(biorbd_model, q_ref, Q_noise), dq_ref))
    X_est = np.zeros((biorbd_model.nbQ() * 2, x_ref[:, ::rt_ratio].shape[1] - Ns_mhe))
    U_est = np.zeros((biorbd_model.nbMuscles(), u_ref[:, ::rt_ratio].shape[1] - Ns_mhe))

    # Update bounds
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))
    x_bounds[0].min[: biorbd_model.nbQ(), 0] = x_ref[: biorbd_model.nbQ(), 0] - 0.1
    x_bounds[0].max[: biorbd_model.nbQ(), 0] = x_ref[: biorbd_model.nbQ(), 0] + 0.1
    ocp.update_bounds(x_bounds)

    # Update initial guess
    x_init = InitialGuess(x_ref[:, : Ns_mhe + 1], interpolation=InterpolationType.EACH_FRAME)
    u_init = InitialGuess([0.2] * biorbd_model.nbMuscles(), interpolation=InterpolationType.CONSTANT)
    ocp.update_initial_guess(x_init, u_init)

    # Update objectives functions
    objectives = define_objective(q_ref, 0, rt_ratio, Ns_mhe, biorbd_model)
    ocp.update_objectives(objectives)

    # Initialize the solver options
    sol = ocp.solve(
        solver=Solver.ACADOS,
        show_online_optim=False,
        solver_options={
            "nlp_solver_tol_comp": 1e-10,
            "nlp_solver_tol_eq": 1e-10,
            "nlp_solver_tol_stat": 1e-8,
            "integrator_type": "IRK",
            "nlp_solver_type": "SQP",
            "sim_method_num_steps": 1,
            "print_level": 0,
            "nlp_solver_max_iter": 30,
        },
    )

    # Set solutions and set initial guess for next optimisation
    x0, u0, X_est[:, 0], U_est[:, 0] = warm_start_mhe(sol)
    cost = []
    n_iter = []
    tic = time()  # Save initial time
    for iter in range(1, X_est.shape[1]):
        # set initial state
        ocp.nlp[0].x_bounds.min[:, 0] = x0[:, 0]
        ocp.nlp[0].x_bounds.max[:, 0] = x0[:, 0]

        # Update initial guess
        x_init = InitialGuess(x0, interpolation=InterpolationType.EACH_FRAME)
        u_init = InitialGuess(u0, interpolation=InterpolationType.EACH_FRAME)
        ocp.update_initial_guess(x_init, u_init)

        # Update objectives functions
        objectives = define_objective(q_ref, iter, rt_ratio, Ns_mhe, biorbd_model)
        ocp.update_objectives(objectives)

        # Solve problem
        sol = ocp.solve(
            solver=Solver.ACADOS,
            show_online_optim=False,
            solver_options={"nlp_solver_tol_comp": 1e-6,"nlp_solver_tol_eq": 1e-6,"nlp_solver_tol_stat": 1e-5},
        )
        # Set solutions and set initial guess for next optimisation
        x0, u0, x_out, u_out = warm_start_mhe(sol)
        X_est[:, iter] = x_out
        if iter < U_est.shape[1]:
            U_est[:, iter] = u_out
    toc = time() - tic
    N = X_est.shape[1] - 1
    Tf = (Ns - Ns % rt_ratio) / (Ns / T)
    final_time = Tf - (Ns_mhe * (Tf / (N + Ns_mhe)))
    short_ocp = prepare_short_ocp(biorbd_model, final_time=final_time, number_shooting_points=N)
    X_init_guess = InitialGuess(X_est, interpolation=InterpolationType.EACH_FRAME)
    U_init_guess = InitialGuess(U_est, interpolation=InterpolationType.EACH_FRAME)
    sol = Solution(short_ocp, [X_init_guess, U_init_guess])

    out.solver.append(out.Solver("Acados"))
    out.nx = X_est.shape[0]
    out.nu = U_est.shape[0]
    out.ns = N
    out.solver[0].n_iteration = iter
    out.solver[0].cost = sol.cost
    out.solver[0].convergence_time = toc
    out.solver[0].compute_error_single_shooting(sol, 1)


if __name__ == "__main__":
    """
    Prepare and solve the MHE example
    """
    use_noise = True  # True to add noise on reference joint angles
    model = "arm_wt_rot_scap.bioMod"
    T = 8
    Ns = 800
    with open(f"Data/sim_ac_8000ms_800sn_REACH2_co_level_0_step5_ERK.bob", "rb") as file:
        data = pickle.load(file)
    states = data["data"][0]
    controls = data["data"][1]
    q_ref = states["q"]
    dq_ref = states["qdot"]
    a_ref = states["muscles"]
    u_ref = controls["muscles"]

    biorbd_model = biorbd.Model(model)
    Ns_mhe = 7
    rt_ratio = 3
    T_mhe = T / (Ns / rt_ratio) * Ns_mhe
    x_wt_noise = np.concatenate((q_ref, dq_ref))

    force_ref_tmp = np.ndarray((biorbd_model.nbMuscles(), u_ref.shape[1]))
    get_force = force_func(biorbd_model, use_activation=False)
    for i in range(biorbd_model.nbMuscles()):
        for k in range(u_ref.shape[1]):
            force_ref_tmp[i, k] = get_force(q_ref[:, k], dq_ref[:, k], a_ref[:, k], u_ref[:, k])[i, :]
    force_ref = force_ref_tmp[:, ::rt_ratio]

    Q_noise = 5
    if use_noise:
        q_ref = generate_noise(biorbd_model, q_ref, Q_noise)
    x_ref = np.concatenate((q_ref, dq_ref))

    X_est = np.zeros((biorbd_model.nbQ() * 2, x_ref[:, ::rt_ratio].shape[1] - Ns_mhe))
    U_est = np.zeros((biorbd_model.nbMuscles(), u_ref[:, ::rt_ratio].shape[1] - Ns_mhe))

    # Initial and final state
    get_force = force_func(biorbd_model)
    # --- Solve the program using ACADOS --- #
    ocp = prepare_ocp(biorbd_model=biorbd_model, final_time=T_mhe, number_shooting_points=Ns_mhe)

    # Update bounds
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))
    x_bounds[0].min[: biorbd_model.nbQ(), 0] = x_ref[: biorbd_model.nbQ(), 0] - 0.1
    x_bounds[0].max[: biorbd_model.nbQ(), 0] = x_ref[: biorbd_model.nbQ(), 0] + 0.1
    ocp.update_bounds(x_bounds)

    # Update initial guess
    x_init = InitialGuess(x_ref[:, : Ns_mhe + 1], interpolation=InterpolationType.EACH_FRAME)
    u_init = InitialGuess([0.2] * biorbd_model.nbMuscles(), interpolation=InterpolationType.CONSTANT)
    ocp.update_initial_guess(x_init, u_init)

    # Update objectives functions
    objectives = define_objective(q_ref, 0, rt_ratio, Ns_mhe, biorbd_model)
    ocp.update_objectives(objectives)

    # Initialize the solver options
    sol = ocp.solve(
        solver=Solver.ACADOS,
        show_online_optim=False,
        solver_options={
            "nlp_solver_tol_comp": 1e-10,
            "nlp_solver_tol_eq": 1e-10,
            "nlp_solver_tol_stat": 1e-8,
            "integrator_type": "IRK",
            "nlp_solver_type": "SQP",
            "sim_method_num_steps": 1,
            "print_level": 0,
            "nlp_solver_max_iter": 30,
        },
    )

    # Set solutions and set initial guess for next optimisation
    x0, u0, X_est[:, 0], U_est[:, 0] = warm_start_mhe(sol)

    tic = time()  # Save initial time
    for iter in range(1, X_est.shape[1]):
        # set initial state
        ocp.nlp[0].x_bounds.min[:, 0] = x0[:, 0]
        ocp.nlp[0].x_bounds.max[:, 0] = x0[:, 0]

        # Update initial guess
        x_init = InitialGuess(x0, interpolation=InterpolationType.EACH_FRAME)
        u_init = InitialGuess(u0, interpolation=InterpolationType.EACH_FRAME)
        ocp.update_initial_guess(x_init, u_init)

        # Update objectives functions
        objectives = define_objective(q_ref, iter, rt_ratio, Ns_mhe, biorbd_model)
        ocp.update_objectives(objectives)

        # Solve problem
        sol = ocp.solve(
            solver=Solver.ACADOS,
            show_online_optim=False,
            solver_options={
                "nlp_solver_tol_comp": 1e-6,
                "nlp_solver_tol_eq": 1e-6,
                "nlp_solver_tol_stat": 1e-5,
            },
        )
        # Set solutions and set initial guess for next optimisation
        x0, u0, x_out, u_out = warm_start_mhe(sol)
        X_est[:, iter] = x_out
        if iter < U_est.shape[1]:
            U_est[:, iter] = u_out

    a_est = U_est
    q_ref = q_ref[:, ::rt_ratio]
    force_est = np.ndarray((biorbd_model.nbMuscles(), U_est.shape[1]))
    for i in range(biorbd_model.nbMuscles()):
        for k in range(U_est.shape[1]):
            force_est[i, k] = get_force(
                X_est[: biorbd_model.nbQ(), k],
                X_est[biorbd_model.nbQ() : biorbd_model.nbQ() * 2, k],
                a_est[:, k],
                U_est[:, k],
            )[i, :]

    toc = time() - tic
    final_offset = 5  # Number of last nodes to ignore when calculate RMSE
    init_offset = 5  # Number of initial nodes to ignore when calculate RMSE
    offset = Ns_mhe

    # --- RMSE --- #
    RMSE_Q = (
        np.sqrt(
            np.square(
                X_est[: biorbd_model.nbQ(), init_offset:-final_offset] - q_ref[:, init_offset : -final_offset - Ns_mhe]
            ).mean(axis=1)
        ).mean()
        * 180
        / np.pi
    )
    STD_Q = (
        np.sqrt(
            np.square(
                X_est[: biorbd_model.nbQ(), init_offset:-final_offset] - q_ref[:, init_offset : -final_offset - Ns_mhe]
            ).mean(axis=1)
        ).std()
        * 180
        / np.pi
    )
    RMSE_F = np.sqrt(
        np.square(force_est[:, init_offset:-final_offset] - force_ref[:, init_offset : -final_offset - Ns_mhe]).mean(
            axis=1
        )
    ).mean()
    STD_F = np.sqrt(
        np.square(force_est[:, init_offset:-final_offset] - force_ref[:, init_offset : -final_offset - Ns_mhe]).mean(
            axis=1
        )
    ).std()
    print(f"Q RMSE: {RMSE_Q} +/- {STD_Q}; F RMSE: {RMSE_F} +/- {STD_F}")
    x_ref = np.concatenate((x_ref, a_ref))

    dic = {
        "X_est": X_est,
        "U_est": U_est,
        "x_ref": x_ref,
        "x_init": x_wt_noise,
        "u_ref": u_ref,
        "time_per_mhe": toc / ceil(Ns / rt_ratio - Ns_mhe),
        "time_tot": toc,
        "Q_noise": Q_noise,
        "N_mhe": Ns_mhe,
        "N_tot": Ns,
        "rt_ratio": rt_ratio,
        "f_est": force_est,
        "f_ref": force_ref,
    }
    sio.savemat(f"Data/MHE_results.mat", dic)
    duration = 1
    T = 8
    Ns = 800

    print("*********************************************")
    print(f"Problem solved with Acados")
    print(f"Solving time : {dic['time_tot']}s")
    print(f"Solving frequency : {1/dic['time_per_mhe']}s")

    # ------ Animate ------ #
    b = bioviz.Viz(model)
    b.load_movement(X_est[: biorbd_model.nbQ(), :])
    b.exec()
