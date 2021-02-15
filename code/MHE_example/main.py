import biorbd
from time import time
import numpy as np
import pickle
import scipy.io as sio
from math import ceil
from casadi import MX, Function
from bioptim import (
    OptimalControlProgram,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsList,
    DynamicsFcn,
    BoundsList,
    QAndQDotBounds,
    InitialGuess,
    Solver,
    Data,
    InterpolationType
)


# Return muscle force
def muscles_forces(q, qdot, a, controls, model, use_activation=False):
    muscles_states = biorbd.VecBiorbdMuscleState(model.nbMuscles())
    for k in range(model.nbMuscles()):
        if use_activation:
            muscles_states[k].setActivation(controls[k])
        else:
            muscles_states[k].setActivation(a[k])
            muscles_states[k].setExcitation(controls[k])
    muscles_force = model.muscleForces(muscles_states, q, qdot).to_mx()
    muscles_tau = model.muscularJointTorque(muscles_states, q, qdot).to_mx()
    return muscles_force, muscles_tau


# Return biorbd muscles force function
def force_func(biorbd_model, use_activation=False):
    qMX = MX.sym("qMX", biorbd_model.nbQ(), 1)
    dqMX = MX.sym("dqMX", biorbd_model.nbQ(), 1)
    aMX = MX.sym("aMX", biorbd_model.nbMuscles(), 1)
    uMX = MX.sym("uMX", biorbd_model.nbMuscles(), 1)
    return Function(
        "MuscleForce",
        [qMX, dqMX, aMX, uMX],
        [muscles_forces(qMX, dqMX, aMX, uMX, biorbd_model, use_activation=use_activation)[0],
         muscles_forces(qMX, dqMX, aMX, uMX, biorbd_model, use_activation=use_activation)[1]],
        ["qMX", "dqMX", "aMX", "uMX"],
        ["Force", "Torque"],
    ).expand()


def generate_noise(biorbd_model, q, q_noise_lvl):
    # Noise on marker position with gaussian normal distribution
    n_q = biorbd_model.nbQ()
    q_noise = np.ndarray((n_q, q.shape[1]))
    for i in range(n_q):
        noise = np.random.normal(0, abs(q_noise_lvl * q[i, :] / 100))
        q_noise[i, :] = q[i, :] + noise
    return q_noise


def warm_start_mhe(ocp, sol):
    # Define problem variable
    data = Data.get_data(ocp, sol)
    q = data[0]["q"]
    dq = data[0]["qdot"]
    u = data[1]["muscles"]
    x = np.vstack([q, dq])

    # Prepare data to return
    x0 = np.hstack((x[:, 1:], np.tile(x[:, [-1]], 1)))  # discard oldest estimate of the window, duplicates youngest
    u0 = u[:, :-1]
    x_out = x[:, 0]
    u_out = u[:, 0]
    return x0, u0, x_out, u_out


def define_objective(iter, rt_ratio, Ns_mhe, biorbd_model):
    objectives = ObjectiveList()
    w_state = 100000
    w_control = 10000
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=w_control)
    objectives.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        weight=1000,
        index=np.array(range(biorbd_model.nbQ(), biorbd_model.nbQ() * 2)),
    )
    objectives.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        weight=100,
        index=np.array(range(biorbd_model.nbQ())),
    )
    objectives.add(
        ObjectiveFcn.Lagrange.TRACK_STATE,
        weight=w_state,
        target=q_ref[:, iter * rt_ratio : (Ns_mhe + 1 + iter) * rt_ratio : rt_ratio],
        index=range(biorbd_model.nbQ()),
    )
    return objectives


def prepare_ocp(biorbd_model, final_time, number_shooting_points, use_SX=True):

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
        use_sx=use_SX,
    )


if __name__ == "__main__":
    use_noise = True
    model = 'arm_wt_rot_scap.bioMod'
    start_delay = 25
    T = 8
    Ns = 800
    Ns = Ns - start_delay
    T = T * 800 / Ns
    with open(f"sim_ac_8000ms_800sn_REACH2_co_level_0.bob", "rb") as file:
        data = pickle.load(file)
    states = data["data"][0]
    controls = data["data"][1]
    q_ref = states["q"][:, :][:, start_delay:]
    dq_ref = states["q_dot"][:, start_delay:]
    a_ref = states["muscles"][:, start_delay:]
    u_ref = controls["muscles"][:, start_delay:]

    biorbd_model = biorbd.Model(model)
    Ns_mhe = 7
    rt_ratio = 4
    T_mhe = T / (Ns / rt_ratio) * Ns_mhe
    x_wt_noise = np.concatenate((q_ref, dq_ref))
    Q_noise = 5  # Percentage of Q for gaussian standard deviation
    if use_noise:
        q_ref = generate_noise(biorbd_model, q_ref, Q_noise)
    x_ref = np.concatenate((q_ref, dq_ref))

    X_est = np.zeros((biorbd_model.nbQ() * 2, ceil((Ns + 1) / rt_ratio) - Ns_mhe))
    U_est = np.zeros((biorbd_model.nbMuscles(), ceil(Ns / rt_ratio) - Ns_mhe))

    force_ref_tmp = np.ndarray((biorbd_model.nbMuscles(), Ns))
    get_force = force_func(biorbd_model, use_activation=False)
    for i in range(biorbd_model.nbMuscles()):
        for k in range(Ns):
            force_ref_tmp[i, k] = get_force(q_ref[:, k], dq_ref[:, k], a_ref[:, k], u_ref[:, k])[0][i, :]
    force_ref = force_ref_tmp[:, 0:Ns:rt_ratio]

    # Initial and final state
    get_force = force_func(biorbd_model)
    # --- Solve the program using ACADOS --- #
    ocp = prepare_ocp(biorbd_model=biorbd_model, final_time=T_mhe, number_shooting_points=Ns_mhe, use_SX=True)

    # Update bounds
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))
    x_bounds[0].min[: biorbd_model.nbQ(), 0] = x_ref[: biorbd_model.nbQ(), 0] - 0.1
    x_bounds[0].max[: biorbd_model.nbQ(), 0] = x_ref[: biorbd_model.nbQ(), 0] + 0.1
    ocp.update_bounds(x_bounds)

    # Update initial guess
    x_init = InitialGuess(x_ref[:, :Ns_mhe+1], interpolation=InterpolationType.EACH_FRAME)
    u_init = InitialGuess([0.2] * biorbd_model.nbMuscles(), interpolation=InterpolationType.CONSTANT)
    ocp.update_initial_guess(x_init, u_init)

    # Update objectives functions
    objectives = define_objective(0, rt_ratio, Ns_mhe, biorbd_model)
    ocp.update_objectives(objectives)

    # Initialize the solver options
    sol = ocp.solve(
        solver=Solver.ACADOS,
        show_online_optim=False,
        solver_options={
            "nlp_solver_tol_comp": 1e-4,
            "nlp_solver_tol_eq": 1e-4,
            "nlp_solver_tol_stat": 1e-4,
            "integrator_type": "IRK",
            "nlp_solver_type": "SQP",
            "sim_method_num_steps": 1,
            "print_level": 0,
            "nlp_solver_max_iter": 15,
        },
    )

    # Set solutions and set initial guess for next optimisation
    x0, u0, X_est[:, 0], U_est[:, 0] = warm_start_mhe(ocp, sol)

    tic = time()  # Save initial time
    for iter in range(1, int((Ns + 1) / rt_ratio - Ns_mhe)):
        # set initial state
        ocp.nlp[0].x_bounds.min[:, 0] = x0[:, 0]
        ocp.nlp[0].x_bounds.max[:, 0] = x0[:, 0]

        # Update initial guess
        x_init = InitialGuess(x0, interpolation=InterpolationType.EACH_FRAME)
        u_init = InitialGuess(u0, interpolation=InterpolationType.EACH_FRAME)
        ocp.update_initial_guess(x_init, u_init)

        # Update objectives functions
        objectives = define_objective(iter, rt_ratio, Ns_mhe, biorbd_model)
        ocp.update_objectives(objectives)
        # Solve problem
        sol = ocp.solve(
            solver=Solver.ACADOS,
            show_online_optim=False,
            solver_options={
                "nlp_solver_tol_comp": 1e-4,
                "nlp_solver_tol_eq": 1e-4,
                "nlp_solver_tol_stat": 1e-4,
            },
        )
        # Set solutions and set initial guess for next optimisation
        x0, u0, x_out, u_out = warm_start_mhe(ocp, sol)
        X_est[:, iter] = x_out
        if iter < int(Ns / rt_ratio) - Ns_mhe:
            U_est[:, iter] = u_out

    a_est = U_est
    u_ref = u_ref[:, 0:Ns:rt_ratio]
    q_ref = q_ref[:, 0: Ns + 1: rt_ratio]
    dq_ref = dq_ref[:, 0: Ns + 1: rt_ratio]
    force_est = np.ndarray((biorbd_model.nbMuscles(), int(ceil(Ns / rt_ratio) - Ns_mhe)))
    for i in range(biorbd_model.nbMuscles()):
        for k in range(int(ceil(Ns / rt_ratio) - Ns_mhe)):
            force_est[i, k] = get_force(
                X_est[:biorbd_model.nbQ(), k], X_est[biorbd_model.nbQ():biorbd_model.nbQ()*2, k], a_est[:, k],
                U_est[:, k]
                )[0][i, :]

    toc = time() - tic
    print(time()-tic)
    print(toc/ceil((Ns + 1) / rt_ratio - Ns_mhe))
    final_offset = 30  # Number of last nodes to ignore when calculate RMSE
    init_offset = 5
    offset = Ns_mhe
    RMSE_Q = np.sqrt(
        np.square(X_est[: biorbd_model.nbQ(), init_offset:-10]- q_ref[:, init_offset:-10-Ns_mhe]).mean(axis=1)
    ).mean()*180/np.pi
    STD_Q = np.sqrt(
        np.square(
            X_est[: biorbd_model.nbQ(), init_offset:-10]- q_ref[:, init_offset:-10-Ns_mhe]).mean(axis=1)
    ).std() * 180 / np.pi

    dic = {
        "X_est": X_est,
        "U_est": U_est,
        "x_ref": x_ref,
        "x_init": x_wt_noise,
        "u_sol": u_ref,
        "time_per_mhe": toc / ceil(Ns / rt_ratio - Ns_mhe),
        "time_tot": toc,
        "Q_noise": Q_noise,
        "N_mhe": Ns_mhe,
        "N_tot": Ns,
        "rt_ratio": rt_ratio,
        "f_est": force_est,
        "f_ref": force_ref,
    }
    sio.savemat(f"MHE_results.mat", dic)
