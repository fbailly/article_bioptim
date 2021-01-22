from time import time

import numpy as np
import biorbd
from casadi import if_else, lt, vertcat
from bioptim import (
    BidirectionalMapping,
    BoundsList,
    ConstraintFcn,
    ConstraintList,
    Data,
    DynamicsFcn,
    DynamicsList,
    InitialGuess,
    InitialGuessList,
    InterpolationType,
    Mapping,
    Node,
    ObjectiveFcn,
    ObjectiveList,
    OptimalControlProgram,
    PlotType,
    QAndQDotBounds,
    ShowResult,
)


def com_dot_z(ocp, nlp, t, x, u, p):
    q = nlp.mapping["q"].expand.map(x[0][: nlp.shape["q"]])
    q_dot = nlp.mapping["q"].expand.map(x[0][nlp.shape["q"]:])
    com_dot_func = biorbd.to_casadi_func("Compute_CoM_dot", nlp.model.CoMdot, nlp.q, nlp.q_dot)

    com_dot = com_dot_func(q, q_dot)
    return com_dot[2]


def toe_on_floor(ocp, nlp, t, x, u, p):
    # floor = -0.77865438
    nb_q = nlp.shape["q"]
    q_reduced = nlp.X[0][:nb_q]
    q = nlp.mapping["q"].expand.map(q_reduced)
    marker_func = biorbd.to_casadi_func("toe_on_floor", nlp.model.marker, nlp.q, 2)
    toe_marker_z = marker_func(q)[2]
    return toe_marker_z + 0.779


def heel_on_floor(ocp, nlp, t, x, u, p):
    # floor = -0.77865829
    nb_q = nlp.shape["q"]
    q_reduced = nlp.X[0][:nb_q]
    q = nlp.mapping["q"].expand.map(q_reduced)
    marker_func = biorbd.to_casadi_func("heel_on_floor", nlp.model.marker, nlp.q, 3)
    tal_marker_z = marker_func(q)[2]
    return tal_marker_z + 0.779


def tau_actuator_constraints(ocp, nlp, t, x, u, p, minimal_tau=None):
    nq = nlp.mapping["q"].reduce.len
    q = [nlp.mapping["q"].expand.map(mx[:nq]) for mx in x]
    q_dot = [nlp.mapping["q_dot"].expand.map(mx[nq:]) for mx in x]

    min_bound = []
    max_bound = []

    func = biorbd.to_casadi_func("torqueMax", nlp.model.torqueMax, nlp.q, nlp.q_dot)
    for i in range(len(u)):
        bound = func(q[i], q_dot[i])
        if minimal_tau:
            min_bound.append(nlp.mapping["tau"].reduce.map(if_else(lt(bound[:, 1], minimal_tau), minimal_tau, bound[:, 1])))
            max_bound.append(nlp.mapping["tau"].reduce.map(if_else(lt(bound[:, 0], minimal_tau), minimal_tau, bound[:, 0])))
        else:
            min_bound.append(nlp.mapping["tau"].reduce.map(bound[:, 1]))
            max_bound.append(nlp.mapping["tau"].reduce.map(bound[:, 0]))

    obj = vertcat(*u)
    min_bound = vertcat(*min_bound)
    max_bound = vertcat(*max_bound)

    return (
        vertcat(np.zeros(min_bound.shape), np.ones(max_bound.shape) * -np.inf),
        vertcat(obj + min_bound, obj - max_bound),
        vertcat(np.ones(min_bound.shape) * np.inf, np.zeros(max_bound.shape)),
    )


def plot_com(x, nlp):
    q = nlp.mapping["q"].expand.map(x[:7, :])
    com_func = biorbd.to_casadi_func("Compute_CoM", nlp.model.CoM, nlp.q)
    return np.array(com_func(q))[2]


def plot_com_dot(x, nlp):
    q = nlp.mapping["q"].expand.map(x[:7, :])
    q_dot = nlp.mapping["q"].expand.map(x[7:, :])
    com_dot_func = biorbd.to_casadi_func("Compute_CoM", nlp.model.CoMdot, nlp.q, nlp.q_dot)
    return np.array(com_dot_func(q, q_dot))[2]


def plot_torque_bounds(x, min_or_max, nlp, minimal_tau=None):
    q = nlp.mapping["q"].expand.map(x[:7, :])
    q_dot = nlp.mapping["q"].expand.map(x[7:, :])
    func = biorbd.to_casadi_func("TorqueMax", nlp.model.torqueMax, nlp.q, nlp.q_dot)

    res = []
    for dof in [6, 7, 8, 9]:
        bound = []

        for i in range(len(x[0])):
            tmp = func(q[:, i], q_dot[:, i])
            if minimal_tau and tmp[dof, min_or_max] < minimal_tau:
                bound.append(minimal_tau)
            else:
                bound.append(tmp[dof, min_or_max])
        res.append(np.array(bound))

    return np.array(res)


def add_custom_plots(ocp, nb_phases, x_bounds, nq, minimal_tau=None):
    for i in range(nb_phases):
        nlp = ocp.nlp[i]
        # Plot Torque Bounds
        if not minimal_tau:
            ocp.add_plot("tau", lambda x, u, p: plot_torque_bounds(x, 0, nlp), phase=i, plot_type=PlotType.STEP, color="g")
            ocp.add_plot("tau", lambda x, u, p: -plot_torque_bounds(x, 1, nlp), phase=i, plot_type=PlotType.STEP, color="g")
        else:
            ocp.add_plot("tau", lambda x, u, p: plot_torque_bounds(x, 0, nlp), phase=i, plot_type=PlotType.STEP, color="g", linestyle="-.")
            ocp.add_plot("tau", lambda x, u, p: -plot_torque_bounds(x, 1, nlp), phase=i, plot_type=PlotType.STEP, color="g", linestyle="-.")
            ocp.add_plot("tau", lambda x, u, p: plot_torque_bounds(x, 0, nlp, minimal_tau=minimal_tau), phase=i, plot_type=PlotType.STEP, color="g")
            ocp.add_plot("tau", lambda x, u, p: -plot_torque_bounds(x, 1, nlp, minimal_tau=minimal_tau), phase=i, plot_type=PlotType.STEP, color="g")
        # Plot CoM pos and speed
        ocp.add_plot("CoM", lambda x, u, p: plot_com(x, nlp), phase=i, plot_type=PlotType.PLOT)
        ocp.add_plot("CoM_dot", lambda x, u, p: plot_com_dot(x, nlp), phase=i, plot_type=PlotType.PLOT)
        # Plot q and nb_q_dot ranges
        ocp.add_plot(
            "q",
            lambda x, u, p: np.repeat(x_bounds[i].min[:nq, 1][:, np.newaxis], x.shape[1], axis=1),
            phase=i,
            plot_type=PlotType.PLOT,
        )
        ocp.add_plot(
            "q",
            lambda x, u, p: np.repeat(x_bounds[i].max[:nq, 1][:, np.newaxis], x.shape[1], axis=1),
            phase=i,
            plot_type=PlotType.PLOT,
        )
        ocp.add_plot(
            "q_dot",
            lambda x, u, p: np.repeat(x_bounds[i].min[nq:, 1][:, np.newaxis], x.shape[1], axis=1),
            phase=i,
            plot_type=PlotType.PLOT,
        )
        ocp.add_plot(
            "q_dot",
            lambda x, u, p: np.repeat(x_bounds[i].max[nq:, 1][:, np.newaxis], x.shape[1], axis=1),
            phase=i,
            plot_type=PlotType.PLOT,
        )
    return ocp




def prepare_ocp(model_path, phase_time, ns, time_min, time_max):
    # --- Options --- #
    # Model path
    biorbd_model = [biorbd.Model(elt) for elt in model_path]

    nb_phases = len(biorbd_model)

    q_mapping = BidirectionalMapping(
        Mapping([0, 1, 2, -1, 3, -1, 3, 4, 5, 6, 4, 5, 6], [5]), Mapping([0, 1, 2, 4, 7, 8, 9])
    )
    q_mapping = q_mapping, q_mapping
    tau_mapping = BidirectionalMapping(
        Mapping([-1, -1, -1, -1, 0, -1, 0, 1, 2, 3, 1, 2, 3], [5]), Mapping([4, 7, 8, 9])
    )
    tau_mapping = tau_mapping, tau_mapping
    nq = len(q_mapping[0].reduce.map_idx)

    # Objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_PREDICTED_COM_HEIGHT, weight=-100, phase=1)

    for i in range(nb_phases):
        objective_functions.add(
            ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=0.0001, phase=i, min_bound=time_min[i], max_bound=time_max[i]
        )

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN_WITH_CONTACT)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN_WITH_CONTACT)

    # Constraints
    constraints = ConstraintList()

    # Positivity constraints of the normal component of the reaction forces
    contact_axes = (1, 2, 4, 5)
    for i in contact_axes:
        constraints.add(ConstraintFcn.CONTACT_FORCE, phase=0, node=Node.ALL, contact_force_idx=i, max_bound=np.inf)
    contact_axes = (1, 3)
    for i in contact_axes:
        constraints.add(ConstraintFcn.CONTACT_FORCE, phase=1, node=Node.ALL, contact_force_idx=i, max_bound=np.inf)

    # Non-slipping constraints
    # N.B.: Application on only one of the two feet is sufficient, as the slippage cannot occurs on only one foot.
    constraints.add(
        ConstraintFcn.NON_SLIPPING,
        phase=0,
        node=Node.ALL,
        normal_component_idx=(1, 2),
        tangential_component_idx=0,
        static_friction_coefficient=0.5,
    )
    constraints.add(
        ConstraintFcn.NON_SLIPPING,
        phase=1,
        node=Node.ALL,
        normal_component_idx=1,
        tangential_component_idx=0,
        static_friction_coefficient=0.5,
    )

    # Custom constraints for positivity of CoM_dot on z axis just before the take-off
    constraints.add(com_dot_z, phase=1, node=Node.END, min_bound=0, max_bound=np.inf)

    # Constraint arm positivity
    constraints.add(ConstraintFcn.TRACK_STATE, phase=1, node=Node.END, index=3, min_bound=1.0, max_bound=np.inf)

    # Constraint foot positivity
    constraints.add(heel_on_floor, phase=1, node=Node.ALL, min_bound=-0.0001, max_bound=np.inf)

    # Torque constraint
    for i in range(nb_phases):
        constraints.add(tau_actuator_constraints, phase=i, node=Node.ALL, minimal_tau=20)


    # Path constraint
    nb_q = q_mapping[0].reduce.len
    nb_q_dot = nb_q
    pose_at_first_node = [0, 0, -0.5336, 1.4, 0.8, -0.9, 0.47]

    # Initialize x_bounds (Interpolation type is CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)
    x_bounds = BoundsList()
    for i in range(nb_phases):
        x_bounds.add(bounds=QAndQDotBounds(biorbd_model[i], all_generalized_mapping=q_mapping[i]))
    x_bounds[0][:, 0] = pose_at_first_node + [0] * nb_q_dot

    # Initial guess for states (Interpolation type is CONSTANT)
    x_init = InitialGuessList()
    for i in range(nb_phases):
        x_init.add(pose_at_first_node + [0] * nb_q_dot)

    # Control path constraint
    u_bounds = BoundsList()
    for i in range(nb_phases):
        u_bounds.add([-500] * tau_mapping[i].reduce.len, [500] * tau_mapping[i].reduce.len)

    # Initial guess for controls
    u_init = InitialGuessList()
    for i in range(nb_phases):
        u_init.add([0] * tau_mapping[i].reduce.len)

    # ------------- #

    ocp = OptimalControlProgram(
        biorbd_model,
        dynamics,
        ns,
        phase_time,
        x_init=x_init,
        x_bounds=x_bounds,
        u_init=u_init,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        q_mapping=q_mapping,
        q_dot_mapping=q_mapping,
        tau_mapping=tau_mapping,
        nb_threads=2,
        use_SX=False,
    )
    return add_custom_plots(ocp, nb_phases, x_bounds, nq, minimal_tau=20)


def warm_start_nmpc(sol, ocp):
    state, ctrl, param = Data.get_data(ocp, sol, concatenate=False, get_parameters=True)
    u_init = InitialGuessList()
    x_init = InitialGuessList()
    for i in range(ocp.nb_phases):
        u_init.add(np.concatenate([ctrl[d][i][:, :-1] for d in ctrl]), interpolation=InterpolationType.EACH_FRAME)
        x_init.add(np.concatenate([state[d][i] for d in state]), interpolation=InterpolationType.EACH_FRAME)

    time = InitialGuess(param["time"], name="time")
    ocp.update_initial_guess(x_init=x_init, u_init=u_init, param_init=time)


if __name__ == "__main__":
    model_path = (
        "jumper2contacts.bioMod",
        "jumper1contacts.bioMod",
    )
    time_min = [0.2, 0.05]
    time_max = [1, 1]
    phase_time = [0.6, 0.2]
    number_shooting_points = [30, 15]

    tic = time()

    ocp = prepare_ocp(
        model_path=model_path,
        phase_time=phase_time,
        ns=number_shooting_points,
        time_min=time_min,
        time_max=time_max,
    )

    sol = ocp.solve(
        show_online_optim=False,
        solver_options={"hessian_approximation": "limited-memory", "max_iter": 200}
    )

    warm_start_nmpc(sol, ocp)
    ocp.solver.set_lagrange_multiplier(sol)

    sol = ocp.solve(
        show_online_optim=False,
        solver_options={"hessian_approximation": "exact",
                        "max_iter": 1000,
                        "warm_start_init_point": "yes",
                        }
    )

    toc = time() - tic
    print(f"Time to solve : {toc}sec")

    result = ShowResult(ocp, sol)
    result.animate(nb_frames=241)
