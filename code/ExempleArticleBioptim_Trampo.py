import biorbd
import casadi as cas
import numpy as np
from time import time

from bioptim import (
    OptimalControlProgram,
    DynamicsFcn,
    Bounds,
    ConstraintFcn,
    ShowResult,
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
)


def states2eulerRate(states):
    # maximizing Lagrange twist velocity (indetermination of Quaternion to Euler of 2*pi*n)

    def bodyVel2eulerRate(w, e):
        # xyz convention
        ph = e[0]
        th = e[1]
        ps = e[2]
        wx = w[0]
        wy = w[1]
        wz = w[2]
        dph = cas.cos(ps) / cas.cos(th) * wx - cas.sin(ps) / cas.cos(th) * wy
        dth = cas.sin(ps) * wx + cas.cos(ps) * wy
        dps = -cas.cos(ps) * cas.sin(th) / cas.cos(th) * wx + cas.sin(th) * cas.sin(ps) / cas.cos(th) * wy + wz
        return cas.vertcat(dph, dth, dps)

    Quat_cas = cas.vertcat(states[8], states[3], states[4], states[5])
    Quat_cas /= cas.norm_fro(Quat_cas)

    Quaterion = biorbd.Quaternion(Quat_cas[0], Quat_cas[1], Quat_cas[2], Quat_cas[3])

    omega = cas.vertcat(states[12:15])
    euler = biorbd.Rotation.toEulerAngles(biorbd.Quaternion.toMatrix(Quaterion), 'xyz').to_mx()
    EulRate = bodyVel2eulerRate(omega, euler)

    Func = cas.Function('MaxTwist', [states], [EulRate])
    return Func


def MaxTwistQuat(ocp, nlp, t, x, u, p, states2eulerRate_func):
    val = []
    for i in range(nlp.ns):
        val = cas.vertcat(val, states2eulerRate_func(x[i])[-1])
    return val


def states2euler(states):
    Quat_cas = cas.vertcat(states[8], states[3], states[4], states[5])
    Quat_cas /= cas.norm_fro(Quat_cas)
    Quaterion = biorbd.Quaternion(Quat_cas[0], Quat_cas[1], Quat_cas[2], Quat_cas[3])
    euler = biorbd.Rotation.toEulerAngles(biorbd.Quaternion.toMatrix(Quaterion), 'xyz').to_mx()

    Func = cas.Function('states2euler', [states], [euler])
    return Func


def FinalPositionQuat(ocp, nlp, t, x, u, p, states2euler_func):
    val = states2euler_func(x[0])[0]
    return val


def Eul2Quat(Eul):
    Quat = biorbd.Quaternion.fromMatrix(biorbd.Rotation.fromEulerAngles(Eul, 'xyz')).to_mx()
    Quat /= cas.norm_fro(Quat)
    Func = cas.Function('Eul2Quat', [Eul], [Quat])
    return Func


def EulRate2BodyVel(Quat, EulRate, Eul):
    Quat_biorbd = biorbd.Quaternion(Quat[0], Quat[1], Quat[2], Quat[3])
    EulRate_biorbd = biorbd.Vector3d(EulRate[0], EulRate[1], EulRate[2])
    Eul_biorbd = biorbd.Vector3d(Eul[0], Eul[1], Eul[2])
    Omega = biorbd.Quaternion.eulerDotToOmega(Quat_biorbd, EulRate_biorbd, Eul_biorbd, 'xyz').to_mx()
    Func = cas.Function('Eul2Quat', [Quat, EulRate, Eul], [Omega])
    return Func


def prepare_ocp(biorbd_model_path, final_time, number_shooting_points, nb_threads):

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
    x = np.vstack((np.zeros((n_q, number_shooting_points + 1)), np.ones((n_qdot, number_shooting_points + 1))))
    x[2, :] = vz0 * np.linspace(0, final_time, number_shooting_points + 1) + -9.81/2 * np.linspace(0, final_time, number_shooting_points + 1)**2
    x[3, :] = np.linspace(0, 2 * np.pi, number_shooting_points + 1)
    x[5, :] = np.linspace(0, 2 * np.pi, number_shooting_points + 1)
    x[6, :] = np.random.random((1, number_shooting_points + 1)) * np.pi - np.pi
    x[7, :] = np.random.random((1, number_shooting_points + 1)) * np.pi

    x[n_q + 2, :] = vz0 -9.81 * np.linspace(0, final_time, number_shooting_points + 1)
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
        number_shooting_points,
        final_time,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        objective_functions,
        constraints,
        nb_threads=nb_threads,
        tau_mapping=U_mapping,
    )


def prepare_ocp_Quat(biorbd_model_path, final_time, number_shooting_points, nb_threads): # use_SX=False

    # --- Options --- #
    biorbd_model = biorbd.Model(biorbd_model_path)
    tau_min, tau_max, tau_init = -100, 100, 0
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    n_tau = biorbd_model.nbGeneralizedTorque()-6
    n_root = biorbd_model.nbRoot()

    # Add objective functions
    objective_functions = ObjectiveList()
    states_MX = cas.MX.sym('states_MX', n_q + n_qdot)
    states2eulerRate_func = states2eulerRate(states_MX)
    states2euler_func = states2euler(states_MX)
    objective_functions.add(MaxTwistQuat, states2eulerRate_func=states2eulerRate_func, custom_type=ObjectiveFcn.Lagrange, weight=-1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=1e-6)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    # Initial guesses
    vz0 = 6.0
    x = np.zeros((n_q + n_qdot, number_shooting_points + 1))

    x[2, :] = vz0 * np.linspace(0, final_time, number_shooting_points + 1) + -9.81 / 2 * np.linspace(0, final_time, number_shooting_points + 1) ** 2

    Eul_MX = cas.MX.sym('Eul_MX', 3)
    Quat_MX = cas.MX.sym('Quat_MX', 4)
    EulRate_MX = cas.MX.sym('EulRate_MX', 3)
    Eul2Quat_func = Eul2Quat(Eul_MX)
    EulRate2BodyVel_func = EulRate2BodyVel(Quat_MX, EulRate_MX, Eul_MX)
    RootEuler = np.zeros((3, number_shooting_points + 1))
    RootEulerRate = np.zeros((3, number_shooting_points + 1))
    RootEuler[0, :] = np.linspace(0.01, 2 * np.pi, number_shooting_points + 1)
    RootEuler[2, :] = np.linspace(0.01, 2 * np.pi, number_shooting_points + 1)
    RootEulerRate[0, :] = 2 * np.pi / final_time
    RootEulerRate[2, :] = 2 * np.pi / final_time
    for i in range(number_shooting_points + 1):
        RootQuat = Eul2Quat_func(RootEuler[:, i])
        x[3:6, i] = np.reshape(RootQuat[1:], 3)
        x[8, i] = np.reshape(RootQuat[0], 1)
        RootOmega = EulRate2BodyVel_func(RootQuat, RootEulerRate[:,i], RootEuler[:,i])
        x[12:15, i] = np.reshape(RootOmega, 3)

    x[6, :] = np.random.random((1, number_shooting_points + 1)) * np.pi - np.pi
    x[7, :] = np.random.random((1, number_shooting_points + 1)) * np.pi

    x[n_q + 2, :] = vz0 - 9.81 * np.linspace(0, final_time, number_shooting_points + 1)

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
    constraints.add(FinalPositionQuat, states2euler_func=states2euler_func, node=Node.END, min_bound=-15*np.pi/180, max_bound=15*np.pi/180)


    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        number_shooting_points,
        final_time,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        objective_functions,
        constraints,
        nb_threads=nb_threads,
        tau_mapping=U_mapping,
    )



if __name__ == "__main__":

    # ocp = prepare_ocp(biorbd_model_path="JeChMesh_8DoF.bioMod", final_time=1.5, number_shooting_points=100, nb_threads=4)
    ocp = prepare_ocp_Quat(biorbd_model_path="JeChMesh_RootQuat.bioMod",
                           final_time=1.5,
                           number_shooting_points=100,
                           nb_threads=4)

    # --- Solve the program --- #
    tic = time()
    sol = ocp.solve(solver_options={'ipopt.tol': 1e-5, 'ipopt.constr_viol_tol': 1e-5, 'ipopt.max_iter': 10000})

    toc = time() - tic
    print(f"Time to solve : {toc}sec")

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()

    Solution_data = Data.get_data(ocp, sol, get_states=True)
    q = Solution_data[0]['q']

    np.save('TrampoRootQuat_eul_2', q)














