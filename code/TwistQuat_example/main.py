"""
This is an example on how to use quaternion to represent the orientation of the root of the model.
The avatar must complete one somersault rotation while maximizing the twist rotation.
"""

import biorbd
import casadi as cas
import numpy as np
from time import time
import matplotlib.pyplot as plt
import seaborn
import utils

from bioptim import (
    OptimalControlProgram,
    DynamicsFcn,
    Bounds,
    ConstraintFcn,
    ObjectiveFcn,
    Mapping,
    BidirectionalMapping,
    ConstraintList,
    InitialGuessList,
    InterpolationType,
    ObjectiveList,
    Node,
    Data,
    DynamicsList,
    BoundsList,
    ShowResult,
    Simulate,
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
    U_bounds.add(bounds=Bounds([tau_min] * n_tau, [tau_max] * n_tau))

    U_mapping = BidirectionalMapping(Mapping([-1, -1, -1, -1, -1, -1, 0, 1]), Mapping([0, 1]))

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
    states2eulerRate_func = utils.states2eulerRate(states_MX)
    states2euler_func = utils.states2euler(states_MX)
    objective_functions.add(utils.MaxTwistQuat, states2eulerRate_func=states2eulerRate_func,
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
    Eul2Quat_func = utils.Eul2Quat(Eul_MX)
    EulRate2BodyVel_func = utils.EulRate2BodyVel(Quat_MX, EulRate_MX, Eul_MX)
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
    U_bounds.add(bounds=Bounds([tau_min] * n_tau, [tau_max] * n_tau))

    U_mapping = BidirectionalMapping(Mapping([-1, -1, -1, -1, -1, -1, 0, 1]), Mapping([0, 1]))

    U_init = InitialGuessList()
    U_init.add([tau_init] * n_tau)

    # Set time as a variable
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=0.5, max_bound=1.5)
    constraints.add(utils.FinalPositionQuat, states2euler_func=states2euler_func, node=Node.END,
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
    )


if __name__ == "__main__":
    Quaternion = False

    if Quaternion:
        biorbd_model_path = "JeChMesh_RootQuat.bioMod"
        ocp = prepare_ocp_Quat(biorbd_model_path,
                               final_time=1.5,
                               n_shooting=100)
    else:
        biorbd_model_path = "JeChMesh_8DoF.bioMod"
        ocp = prepare_ocp(biorbd_model_path,
                          final_time=1.5,
                          n_shooting=100)


    # --- Solve the program --- #
    tic = time()
    sol = ocp.solve(solver_options={'ipopt.tol': 1e-15, 'ipopt.constr_viol_tol': 1e-15}) # solver_options={'ipopt.tol': 1e-5, 'ipopt.constr_viol_tol': 1e-5, 'ipopt.max_iter': 10000}
    toc = time() - tic

    result = ShowResult(ocp, sol)
    result.objective_functions()
    sol_opt = sol['x']
    sol_ss = Simulate.from_solve(ocp, sol, True)['x']
    ss_err = np.sqrt(np.mean((sol_ss - sol_opt) ** 2))
    print("*********************************************")
    print(f"Single shooting error : {ss_err}")
    print(f"Time to solve : {toc}sec")

    Solution_data = Data.get_data(ocp, sol, get_states=True)
    q = Solution_data[0]['q']

    if Quaternion:
        np.save('q_optim_quaternion', q)
        ligne_q, colone_q = 2, 4
    else:
        np.save('q_optim_Euler', q)
        ligne_q, colone_q = 3, 3


    fig = plt.figure(figsize=(20, 5))
    plt.gcf().subplots_adjust(left=0.1, right=0.9, wspace=0.35, hspace=0.1, bottom=0.15)

    seaborn.set_style("whitegrid")
    seaborn.color_palette()

    label_DoF = []
    for iplt in range(biorbd.Model(biorbd_model_path).nbQ()):
        label_DoF += biorbd.Model(biorbd_model_path).nameDof()[iplt].to_string()
        ax = plt.subplot(ligne_q, colone_q, iplt + 1)
        ax.plot(q[iplt, :], 'Optimal solution')
        ax.plot(sol_ss[iplt, :], label='Single shooting')
        ax.set_title(label_DoF[iplt])
        if iplt == 0:
            ax.legend(bbox_to_anchor=(1.1, 1.25), loc="upper center", borderaxespad=0.0, ncol=2, frameon=False)
    plt.show()










