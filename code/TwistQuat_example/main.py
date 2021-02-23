"""
This is an example on how to use quaternion to represent the orientation of the root of the model.
The avatar must complete one somersault rotation while maximizing the twist rotation.
"""

import biorbd
import casadi as cas
import numpy as np
from time import time
from .utils import *

from bioptim import (
    OptimalControlProgram,
    DynamicsFcn,
    Bounds,
    ConstraintFcn,
    ObjectiveFcn,
    Mapping,
    BiMapping,
    ConstraintList,
    InitialGuessList,
    InterpolationType,
    ObjectiveList,
    Node,
    DynamicsList,
    BoundsList,
    Shooting,
    OdeSolver,
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
    n_tau = biorbd_model.nbGeneralizedTorque()-biorbd_model.nbRoot()

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, index=n_q+5, weight=-1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=1e-6)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    # Path constraint
    X_bounds = BoundsList()
    x_min = np.zeros((n_q + n_qdot, 3))
    x_max = np.zeros((n_q + n_qdot, 3))
    x_min[:, 0] = [0, 0, 0, 0, 0, 0, -2.8, 2.8,
                     -1, -1, 7,  4,  0, 0, 0, 0]
    x_max[:, 0] = [0, 0, 0, 0, 0, 0, -2.8, 2.8,
                      1,  1, 10, 10, 0, 0, 0, 0]
    x_min[:, 1] = [-1, -1, -0.001, -0.001,        -np.pi/4, -np.pi, -np.pi, 0,
                   -100, -100, -100, -100, -100, -100, -100, -100]
    x_max[:, 1] = [ 1,  1,  5,      2*np.pi+0.001, np.pi/4,  50,     0,     np.pi,
                    100,  100,  100,  100,  100,  100,  100,  100]
    x_min[:, 2] = [-0.1, -0.1, -0.1, 2*np.pi-0.1, -15*np.pi/180, 2*np.pi, -np.pi, 0,
                   -100, -100, -100, -100, -100, -100, -100, -100]
    x_max[:, 2] = [ 0.1,  0.1,  0.1, 2*np.pi+0.1,  15*np.pi/180, 20*np.pi, 0,     np.pi,
                    100,  100,  100,  100,  100,  100,  100,  100]
    X_bounds.add(bounds=Bounds(x_min, x_max, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT))


    # Initial guesses
    vz0 = 6.0
    x = np.vstack((np.zeros((n_q, n_shooting + 1)), np.ones((n_qdot, n_shooting + 1))))
    x[2, :] = vz0 * np.linspace(0, final_time, n_shooting + 1) + -9.81/2 * np.linspace(0, final_time, n_shooting + 1)**2
    x[3, :] = np.linspace(0, 2 * np.pi, n_shooting + 1)
    x[5, :] = np.linspace(0, 2 * np.pi, n_shooting + 1)
    x[6, :] = np.random.random((1, n_shooting + 1)) * np.pi - np.pi
    x[7, :] = np.random.random((1, n_shooting + 1)) * np.pi

    x[n_q + 2, :] = vz0 -9.81 * np.linspace(0, final_time, n_shooting + 1)
    x[n_q + 3, :] = 2 * np.pi / final_time
    x[n_q + 5, :] = 2 * np.pi / final_time

    X_init = InitialGuessList()
    X_init.add(x, interpolation=InterpolationType.EACH_FRAME)


    # Define control path constraint
    U_bounds = BoundsList()
    U_bounds.add([tau_min] * n_tau, [tau_max] * n_tau)

    U_mapping = BiMapping([-1, -1, -1, -1, -1, -1, 0, 1], [0, 1])

    U_init = InitialGuessList()
    U_init.add([tau_init] * n_tau)

    # Set time as a variable
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=0.5, max_bound=1.5)

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        objective_functions,
        constraints,
        n_threads=4,
        tau_mapping=U_mapping,
        ode_solver=OdeSolver.RK8(),
    )


def prepare_ocp_Quat(biorbd_model_path, final_time, n_shooting):
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
    n_tau = biorbd_model.nbGeneralizedTorque()-6

    # Add objective functions
    objective_functions = ObjectiveList()
    states_MX = cas.MX.sym('states_MX', n_q + n_qdot)
    states2eulerRate_func = states2eulerRate(states_MX)
    states2euler_func = states2euler(states_MX)
    objective_functions.add(MaxTwistQuat, states2eulerRate_func=states2eulerRate_func,
                            custom_type=ObjectiveFcn.Lagrange, weight=-1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=1e-6)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    # Initial guesses
    vz0 = 6.0
    x = np.zeros((n_q + n_qdot, n_shooting + 1))

    x[2, :] = vz0 * np.linspace(0, final_time, n_shooting + 1) + -9.81 / 2 \
              * np.linspace(0, final_time, n_shooting + 1) ** 2

    Eul_MX = cas.MX.sym('Eul_MX', 3)
    Quat_MX = cas.MX.sym('Quat_MX', 4)
    EulRate_MX = cas.MX.sym('EulRate_MX', 3)
    Eul2Quat_func = Eul2Quat(Eul_MX)
    EulRate2BodyVel_func = EulRate2BodyVel(Quat_MX, EulRate_MX, Eul_MX)
    RootEuler = np.zeros((3, n_shooting + 1))
    RootEulerRate = np.zeros((3, n_shooting + 1))
    RootEuler[0, :] = np.linspace(0.01, 2 * np.pi, n_shooting + 1)
    RootEuler[2, :] = np.linspace(0.01, 2 * np.pi, n_shooting + 1)
    RootEulerRate[0, :] = 2 * np.pi / final_time
    RootEulerRate[2, :] = 2 * np.pi / final_time
    for i in range(n_shooting + 1):
        RootQuat = Eul2Quat_func(RootEuler[:, i])
        x[3:6, i] = np.reshape(RootQuat[1:], 3)
        x[8, i] = np.reshape(RootQuat[0], 1)
        RootOmega = EulRate2BodyVel_func(RootQuat, RootEulerRate[:,i], RootEuler[:,i])
        x[12:15, i] = np.reshape(RootOmega, 3)

    x[6, :] = np.random.random((1, n_shooting + 1)) * np.pi - np.pi
    x[7, :] = np.random.random((1, n_shooting + 1)) * np.pi

    x[n_q + 2, :] = vz0 - 9.81 * np.linspace(0, final_time, n_shooting + 1)

    X_init = InitialGuessList()
    X_init.add(x, interpolation=InterpolationType.EACH_FRAME)

    # Path constraint
    X_bounds = BoundsList()
    x_min = np.zeros((n_q + n_qdot, 3))
    x_max = np.zeros((n_q + n_qdot, 3))
    x_min[:, 0] = [0, 0, 0, x[3,0], x[4,0], x[5,0], -2.8, 2.8, -1.05,
                     -1, -1, 4,  x[12,0], x[13,0], x[14,0], 0, 0]
    x_max[:, 0] = [0, 0, 0, x[3,0], x[4,0], x[5,0], -2.8, 2.8,  1.05,
                      1,  1, 10, x[12,0], x[13,0], x[14,0], 0, 0]
    x_min[:, 1] = [-1, -1, -0.001, -1.05, -1.05, -1.05, -np.pi, 0,    -1.05,
                   -100, -100, -100, -100, -100, -100, -100, -100]
    x_max[:, 1] = [ 1,  1,  5,      1.05,  1.05,  1.05,  0,     np.pi, 1.05,
                    100,  100,  100,  100,  100,  100,  100,  100]
    x_min[:, 2] = [-0.1, -0.1, -0.1, x[3,0], -1.05, -1.05, -np.pi, 0,    -1.05,
                   -100, -100, -100, -100, -100, -100, -100, -100]
    x_max[:, 2] = [ 0.1,  0.1,  0.1, x[3,0],  1.05,  1.05,  0,     np.pi, 1.05,
                    100,  100,  100,  100,  100,  100,  100,  100]
    X_bounds.add(bounds=Bounds(x_min, x_max, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT))

    # Define control path constraint
    U_bounds = BoundsList()
    U_bounds.add([tau_min] * n_tau, [tau_max] * n_tau)

    U_mapping = BiMapping([-1, -1, -1, -1, -1, -1, 0, 1], [0, 1])

    U_init = InitialGuessList()
    U_init.add([tau_init] * n_tau)

    # Set time as a variable
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=0.5, max_bound=1.5)
    constraints.add(FinalPositionQuat, states2euler_func=states2euler_func, node=Node.END,
                    min_bound=-15*np.pi/180, max_bound=15*np.pi/180)


    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        objective_functions,
        constraints,
        n_threads=4,
        tau_mapping=U_mapping,
        ode_solver=OdeSolver.RK8(),
    )


def generate_table(out, Quaternion):

    if Quaternion:
        model_path = "/".join(__file__.split("/")[:-1]) + "/JeChMesh_RootQuat.bioMod"
        ocp = prepare_ocp_Quat(model_path,
                               final_time=1.5,
                               n_shooting=100)
    else:
        model_path = "/".join(__file__.split("/")[:-1]) + "/JeChMesh_8DoF.bioMod"
        ocp = prepare_ocp(model_path,
                          final_time=1.5,
                          n_shooting=100)

    # --- Solve the program --- #
    tic = time()
    sol = ocp.solve(solver_options={'tol': 1e-15, 'constr_viol_tol': 1e-15, 'max_iter': 1})
    toc = time() - tic

    out.nx = sol.states["all"].shape[0]
    out.nu = sol.controls["all"].shape[0]
    out.ns = sol.ns[0]
    out.solver.append(out.Solver("Ipopt"))
    out.solver[0].n_iteration = sol.iterations
    out.solver[0].cost = sol.cost
    out.solver[0].convergence_time = toc
    out.solver[0].compute_error_single_shooting(sol, 1)


if __name__ == "__main__":
    Quaternion = True
    np.random.seed(42)

    if Quaternion:
        model_path = "JeChMesh_RootQuat.bioMod"
        ocp = prepare_ocp_Quat(model_path,
                               final_time=1.5,
                               n_shooting=100)
    else:
        model_path = "JeChMesh_8DoF.bioMod"
        ocp = prepare_ocp(model_path,
                          final_time=1.5,
                          n_shooting=100)

    tic = time()
    sol = ocp.solve(solver_options={'ipopt.tol': 1e-15, 'ipopt.constr_viol_tol': 1e-15, 'ipopt.max_iter': 10000}) # solver_options={'ipopt.tol': 1e-15, 'ipopt.constr_viol_tol': 1e-15, 'ipopt.max_iter': 0}
    toc = time() - tic

    q_opt = sol.states['q']
    qdot_opt = sol.states['qdot']
    t_opt = sol.parameters['time'][0]

    if Quaternion:
        np.save('q_optim_quaternion_42', q_opt)
    else:
        np.save('q_optim_Euler_42', q_opt)