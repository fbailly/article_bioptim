import numpy as np
import biorbd
import casadi as cas
from bioptim import (
    OptimalControlProgram,
    DynamicsFcn,
    Bounds,
    ConstraintFcn,
    ObjectiveFcn,
    BiMapping,
    ConstraintList,
    InitialGuessList,
    InterpolationType,
    ObjectiveList,
    Node,
    DynamicsList,
    BoundsList,
    OdeSolver,
    PenaltyNodes,
)


def prepare_ocp(biorbd_model_path: str, final_time: float, n_shooting: int) -> OptimalControlProgram:
    """
    Prepare the Euler version of the ocp
    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod
    final_time: float
        The initial guess for the time at the final node
    n_shooting: int
        The number of shooting points
    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    # --- Options --- #
    biorbd_model = biorbd.Model(biorbd_model_path)
    tau_min, tau_max, tau_init = -100, 100, 0
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    n_tau = biorbd_model.nbGeneralizedTorque() - biorbd_model.nbRoot()

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, index=n_q + 5, weight=-1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=1e-6)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    # Initial guesses
    vz0 = 6.0
    x = np.vstack((np.zeros((n_q, n_shooting + 1)), np.ones((n_qdot, n_shooting + 1))))
    x[2, :] = (
        vz0 * np.linspace(0, final_time, n_shooting + 1) + -9.81 / 2 * np.linspace(0, final_time, n_shooting + 1) ** 2
    )
    x[3, :] = np.linspace(0, 2 * np.pi, n_shooting + 1)
    x[5, :] = np.linspace(0, 2 * np.pi, n_shooting + 1)
    x[6, :] = np.random.random((1, n_shooting + 1)) * np.pi - np.pi
    x[7, :] = np.random.random((1, n_shooting + 1)) * np.pi

    x[n_q + 2, :] = vz0 - 9.81 * np.linspace(0, final_time, n_shooting + 1)
    x[n_q + 3, :] = 2 * np.pi / final_time
    x[n_q + 5, :] = 2 * np.pi / final_time

    x_init = InitialGuessList()
    x_init.add(x, interpolation=InterpolationType.EACH_FRAME)

    # Path constraint
    x_bounds = BoundsList()
    x_min = np.zeros((n_q + n_qdot, 3))
    x_max = np.zeros((n_q + n_qdot, 3))
    x_min[:, 0] = [0, 0, 0, 0, 0, 0, -2.8, 2.8, -1, -1, 7, x[n_q + 3, 0], 0, x[n_q + 5, 0], 0, 0]
    x_max[:, 0] = [0, 0, 0, 0, 0, 0, -2.8, 2.8, 1, 1, 10, x[n_q + 3, 0], 0, x[n_q + 5, 0], 0, 0]
    x_min[:, 1] = [
        -1,
        -1,
        -0.001,
        -0.001,
        -np.pi / 4,
        -np.pi,
        -np.pi,
        0,
        -100,
        -100,
        -100,
        -100,
        -100,
        -100,
        -100,
        -100,
    ]
    x_max[:, 1] = [1, 1, 5, 2 * np.pi + 0.001, np.pi / 4, 50, 0, np.pi, 100, 100, 100, 100, 100, 100, 100, 100]
    x_min[:, 2] = [
        -0.1,
        -0.1,
        -0.1,
        2 * np.pi - 0.1,
        -15 * np.pi / 180,
        2 * np.pi,
        -np.pi,
        0,
        -100,
        -100,
        -100,
        -100,
        -100,
        -100,
        -100,
        -100,
    ]
    x_max[:, 2] = [
        0.1,
        0.1,
        0.1,
        2 * np.pi + 0.1,
        15 * np.pi / 180,
        20 * np.pi,
        0,
        np.pi,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
    ]
    x_bounds.add(bounds=Bounds(x_min, x_max, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * n_tau, [tau_max] * n_tau)

    u_mapping = BiMapping([None, None, None, None, None, None, 0, 1], [0, 1])

    u_init = InitialGuessList()
    u_init.add([tau_init] * n_tau)

    # Set time as a variable
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=0.5, max_bound=1.5)

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
        constraints,
        n_threads=8,
        tau_mapping=u_mapping,
        ode_solver=OdeSolver.RK8(),
    )


def states_to_euler_rate(states):
    # maximizing Lagrange twist velocity (indeterminate of quaternion to Euler of 2*pi*n)

    def body_vel_to_euler_rate(w, e):
        # xyz convention
        _ = e[0]
        th = e[1]
        ps = e[2]
        wx = w[0]
        wy = w[1]
        wz = w[2]
        dph = cas.cos(ps) / cas.cos(th) * wx - cas.sin(ps) / cas.cos(th) * wy
        dth = cas.sin(ps) * wx + cas.cos(ps) * wy
        dps = -cas.cos(ps) * cas.sin(th) / cas.cos(th) * wx + cas.sin(th) * cas.sin(ps) / cas.cos(th) * wy + wz
        return cas.vertcat(dph, dth, dps)

    quaternion_cas = cas.vertcat(states[8], states[3], states[4], states[5])
    quaternion_cas /= cas.norm_fro(quaternion_cas)

    quaternion = biorbd.Quaternion(quaternion_cas[0], quaternion_cas[1], quaternion_cas[2], quaternion_cas[3])

    omega = cas.vertcat(states[12:15])
    euler = biorbd.Rotation.toEulerAngles(biorbd.Quaternion.toMatrix(quaternion), "xyz").to_mx()
    eul_rate = body_vel_to_euler_rate(omega, euler)

    return cas.Function("max_twist", [states], [eul_rate])


def prepare_ocp_quaternion(biorbd_model_path, final_time, n_shooting):
    """
    Prepare the quaternion version of the ocp
    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod
    final_time: float
        The initial guess for the time at the final node
    n_shooting: int
        The number of shooting points
    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    # --- Options --- #
    biorbd_model = biorbd.Model(biorbd_model_path)
    tau_min, tau_max, tau_init = -100, 100, 0
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    n_tau = biorbd_model.nbGeneralizedTorque() - 6

    # Add objective functions
    objective_functions = ObjectiveList()
    states_mx = cas.MX.sym("states_mx", n_q + n_qdot)
    states_to_euler_rate_func = states_to_euler_rate(states_mx)
    states_to_euler_func = states_to_euler(states_mx)
    objective_functions.add(
        max_twist_quaternion,
        states_to_euler_rate_func=states_to_euler_rate_func,
        custom_type=ObjectiveFcn.Lagrange,
        weight=-1,
    )
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=1e-6)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    # Initial guesses
    vz0 = 6.0
    x = np.zeros((n_q + n_qdot, n_shooting + 1))

    x[2, :] = (
        vz0 * np.linspace(0, final_time, n_shooting + 1) + -9.81 / 2 * np.linspace(0, final_time, n_shooting + 1) ** 2
    )

    euler_mx = cas.MX.sym("euler_mx", 3)
    quaternion_mx = cas.MX.sym("quaternion_mx", 4)
    euler_rate_mx = cas.MX.sym("euler_rate_mx", 3)
    euler_to_quaternion_func = euler_to_quaternion(euler_mx)
    euler_rate_to_body_velocities_func = euler_rate_to_body_velocities(quaternion_mx, euler_rate_mx, euler_mx)
    root_euler = np.zeros((3, n_shooting + 1))
    root_euler_rate = np.zeros((3, n_shooting + 1))
    root_euler[0, :] = np.linspace(0.01, 2 * np.pi, n_shooting + 1)
    root_euler[2, :] = np.linspace(0.01, 2 * np.pi, n_shooting + 1)
    root_euler_rate[0, :] = 2 * np.pi / final_time
    root_euler_rate[2, :] = 2 * np.pi / final_time
    for i in range(n_shooting + 1):
        root_quaternion = euler_to_quaternion_func(root_euler[:, i])
        x[3:6, i] = np.reshape(root_quaternion[1:], 3)
        x[8, i] = np.reshape(root_quaternion[0], 1)
        root_omega = euler_rate_to_body_velocities_func(root_quaternion, root_euler_rate[:, i], root_euler[:, i])
        x[12:15, i] = np.reshape(root_omega, 3)

    x[6, :] = np.random.random((1, n_shooting + 1)) * np.pi - np.pi
    x[7, :] = np.random.random((1, n_shooting + 1)) * np.pi

    x[n_q + 2, :] = vz0 - 9.81 * np.linspace(0, final_time, n_shooting + 1)

    x_init = InitialGuessList()
    x_init.add(x, interpolation=InterpolationType.EACH_FRAME)

    # Path constraint
    x_bounds = BoundsList()
    x_min = np.zeros((n_q + n_qdot, 3))
    x_max = np.zeros((n_q + n_qdot, 3))
    x_min[:, 0] = [0, 0, 0, x[3, 0], x[4, 0], x[5, 0], -2.8, 2.8, -1.05, -1, -1, 4, x[12, 0], x[13, 0], x[14, 0], 0, 0]
    x_max[:, 0] = [0, 0, 0, x[3, 0], x[4, 0], x[5, 0], -2.8, 2.8, 1.05, 1, 1, 10, x[12, 0], x[13, 0], x[14, 0], 0, 0]
    x_min[:, 1] = [
        -1,
        -1,
        -0.001,
        -1.05,
        -1.05,
        -1.05,
        -np.pi,
        0,
        -1.05,
        -100,
        -100,
        -100,
        -100,
        -100,
        -100,
        -100,
        -100,
    ]
    x_max[:, 1] = [1, 1, 5, 1.05, 1.05, 1.05, 0, np.pi, 1.05, 100, 100, 100, 100, 100, 100, 100, 100]
    x_min[:, 2] = [
        -0.1,
        -0.1,
        -0.1,
        x[3, 0],
        -1.05,
        -1.05,
        -np.pi,
        0,
        -1.05,
        -100,
        -100,
        -100,
        -100,
        -100,
        -100,
        -100,
        -100,
    ]
    x_max[:, 2] = [0.1, 0.1, 0.1, x[3, 0], 1.05, 1.05, 0, np.pi, 1.05, 100, 100, 100, 100, 100, 100, 100, 100]
    x_bounds.add(bounds=Bounds(x_min, x_max, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * n_tau, [tau_max] * n_tau)

    u_mapping = BiMapping([None, None, None, None, None, None, 0, 1], [0, 1])

    u_init = InitialGuessList()
    u_init.add([tau_init] * n_tau)

    # Set time as a variable
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=0.5, max_bound=1.5)
    constraints.add(
        final_position_quaternion,
        states_to_euler_func=states_to_euler_func,
        node=Node.END,
        min_bound=-15 * np.pi / 180,
        max_bound=15 * np.pi / 180,
    )

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
        constraints,
        n_threads=8,
        tau_mapping=u_mapping,
        ode_solver=OdeSolver.RK8(),
    )


def max_twist_quaternion(pn: PenaltyNodes, states_to_euler_rate_func) -> cas.MX:
    val = []
    for i in range(pn.nlp.ns):
        val = cas.vertcat(val, states_to_euler_rate_func(pn.x[i])[-1])
    return val


def states_to_euler(states):
    quaternion_cas = cas.vertcat(states[8], states[3], states[4], states[5])
    quaternion_cas /= cas.norm_fro(quaternion_cas)
    quaternion = biorbd.Quaternion(quaternion_cas[0], quaternion_cas[1], quaternion_cas[2], quaternion_cas[3])
    euler = biorbd.Rotation.toEulerAngles(biorbd.Quaternion.toMatrix(quaternion), "xyz").to_mx()
    return cas.Function("states_to_euler", [states], [euler])


def final_position_quaternion(pn: PenaltyNodes, states_to_euler_func) -> cas.MX:
    val = states_to_euler_func(pn.x[0])[0]
    return val


def euler_to_quaternion(angle):
    quaternion = biorbd.Quaternion.fromMatrix(biorbd.Rotation.fromEulerAngles(angle, "xyz")).to_mx()
    quaternion /= cas.norm_fro(quaternion)
    return cas.Function("euler_to_quaternion", [angle], [quaternion])


def euler_rate_to_body_velocities(quaternion, euler_rate, euler_angle):
    quaternion_biorbd = biorbd.Quaternion(quaternion[0], quaternion[1], quaternion[2], quaternion[3])
    euler_rate_biorbd = biorbd.Vector3d(euler_rate[0], euler_rate[1], euler_rate[2])
    euler_biorbd = biorbd.Vector3d(euler_angle[0], euler_angle[1], euler_angle[2])
    omega = biorbd.Quaternion.eulerDotToOmega(quaternion_biorbd, euler_rate_biorbd, euler_biorbd, "xyz").to_mx()
    return cas.Function("euler_rate_to_body_velocities", [quaternion, euler_rate, euler_angle], [omega])
