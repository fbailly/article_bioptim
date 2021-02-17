from casadi import Function, MX, vertcat
import biorbd
import numpy as np


def forward_dynamics(states: MX.sym, controls: MX.sym, biorbd_model, use_activation=True) -> MX:
    nq = biorbd_model.nbQ()
    q = states[:nq]
    qdot = states[nq:nq*2]
    muscle_states = biorbd.VecBiorbdMuscleState(biorbd_model.nbMuscles())
    for k in range(biorbd_model.nbMuscles()):
        if use_activation:
            muscle_states[k].setActivation(controls[k])
        else:
            act = states[-biorbd_model.nbMuscles():]
            muscle_states[k].setExcitation(controls[k])
            muscle_states[k].setActivation(act[k])
            muscles_activations_dot = biorbd_model.activationDot(muscle_states).to_mx()
    muscles_tau = biorbd_model.muscularJointTorque(muscle_states, q, qdot).to_mx()
    qddot = biorbd.Model.ForwardDynamicsConstraintsDirect(biorbd_model, q, qdot, muscles_tau).to_mx()
    qdot = biorbd_model.computeQdot(q, qdot).to_mx()
    if use_activation:
        xdot = vertcat(qdot, qddot)
    else:
        xdot = vertcat(qdot, qddot, muscles_activations_dot)
    return xdot


def dxdt(x, u, get_xdot, biorbd_model, use_activation=True):
    if use_activation:
        xdot = np.ndarray((biorbd_model.nbQ() * 2, 1))
    else:
        xdot = np.ndarray((biorbd_model.nbQ() * 2 + biorbd_model.nbMuscles(), 1))
    for i in range(xdot.shape[0]):
        xdot[i] = get_xdot(x, u)[i]
    return xdot.squeeze()


def xdot_funct(biorbd_model, use_activation=True):
    uMX = MX.sym("uMX", biorbd_model.nbMuscles(), 1)
    xMX = MX.sym("xMX", biorbd_model.nbQ() * 2, 1) if use_activation else MX.sym(
        "xMX", biorbd_model.nbQ() * 2 + biorbd_model.nbMuscles(), 1
    )
    return  Function(
            "xdot", [xMX, uMX], [forward_dynamics(xMX, uMX, biorbd_model, use_activation=use_activation)], ["xMX", "uMX"], ["xdot"],
        ).expand()


def single_shooting(biorbd_model, x,  u, ss_start_del, step, ratio, T, use_activation=True):
    q = x[:biorbd_model.nbQ(), :]
    dq = x[biorbd_model.nbQ():biorbd_model.nbQ() * 2, :]
    if use_activation is not True:
        act = x[-biorbd_model.nbMuscles():, :]

    get_xdot = xdot_funct(biorbd_model, use_activation=use_activation)

    step_time = T / q.shape[1]
    h = step_time / step
    if use_activation:
        x = np.ndarray((biorbd_model.nbQ()*2, q.shape[1]))
        x[:, ss_start_del] = np.concatenate((q, dq))[:, ss_start_del]
        idx = range(ss_start_del + 1, q.shape[1])
    else:
        x = np.ndarray((biorbd_model.nbQ() * 2 + biorbd_model.nbMuscles(), q.shape[1]))
        x[:, ss_start_del * ratio] = np.concatenate((q, dq, act))[:, ss_start_del * ratio]
        idx = range(ss_start_del * ratio + 1, q.shape[1])

    for i in idx:
        x_prev = x[:, i - 1]
        k1 = dxdt(x_prev, u[:, i], get_xdot, biorbd_model, use_activation=use_activation)
        k2 = dxdt(x_prev + h / 2 * k1, u[:, i], get_xdot, biorbd_model, use_activation=use_activation)
        k3 = dxdt(x_prev + h / 2 * k2, u[:, i], get_xdot, biorbd_model, use_activation=use_activation)
        k4 = dxdt(x_prev + h * k3, u[:, i], get_xdot, biorbd_model, use_activation=use_activation)
        x[:, i] = x_prev + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    x = x[:, ::ratio]if use_activation is not True else x
    return x
