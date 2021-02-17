from casadi import Function, MX, vertcat
import biorbd
import numpy as np


def forward_dynamics(states: MX.sym, controls: MX.sym, biorbd_model, use_activation=True) -> MX:
    """
    Forward dynamics driven by muscle excitations (if use_activation = False) or activation.

    Parameters
    ----------
    states: MX.sym
       The state of the system
    controls: MX.sym
       The controls of the system

    Returns
    ----------
    MX.sym
       The derivative of the states
    """
    nq = biorbd_model.nbQ()
    q = states[:nq]
    qdot = states[nq : nq * 2]
    muscle_states = biorbd.VecBiorbdMuscleState(biorbd_model.nbMuscles())
    for k in range(biorbd_model.nbMuscles()):
        if use_activation:
            muscle_states[k].setActivation(controls[k])
        else:
            act = states[-biorbd_model.nbMuscles() :]
            muscle_states[k].setExcitation(controls[k])
            muscle_states[k].setActivation(act[k])
            muscles_activations_dot = biorbd_model.activationDot(muscle_states).to_mx()
    muscles_tau = biorbd_model.muscularJointTorque(muscle_states, q, qdot).to_mx()
    qddot = biorbd.Model.ForwardDynamics(biorbd_model, q, qdot, muscles_tau).to_mx()
    # qdot = biorbd_model.computeQdot(q, qdot).to_mx()
    if use_activation:
        xdot = vertcat(qdot, qddot)
    else:
        xdot = vertcat(qdot, qddot, muscles_activations_dot)

    return xdot


def dxdt(x: np.array, u: np.array, get_xdot: Function):
    """
    Fonction to derivate x

    Parameters
    ----------
    states: MX.sym
       The state of the system
    controls: MX.sym
       The controls of the system
    get_xdot: Function
        the casadi Function
    Returns
    ----------
    xdot
    """
    return np.array(get_xdot(x, u)).squeeze()


def xdot_funct(biorbd_model: biorbd.Model, use_activation=True):
    """
    Definition of the casadi function

    Parameters
    ----------
    biorbd_model: biorbd.Model
        biorbd model build with the bioMod
    use_activation: bool
        True if activation driven; False if excitation driven
    Returns
    ----------
    Casadi function
    """
    uMX = MX.sym("uMX", biorbd_model.nbMuscles(), 1)
    xMX = (
        MX.sym("xMX", biorbd_model.nbQ() * 2, 1)
        if use_activation
        else MX.sym("xMX", biorbd_model.nbQ() * 2 + biorbd_model.nbMuscles(), 1)
    )
    return Function(
        "xdot",
        [xMX, uMX],
        [forward_dynamics(xMX, uMX, biorbd_model, use_activation=use_activation)],
        ["xMX", "uMX"],
        ["xdot"],
    ).expand()


def compute_error_single_shooting(
    biorbd_model: biorbd.Model,
    x: np.array,
    u: np.array,
    step: int,
    ratio: int,
    Tf: int,
    duration: int,
    use_activation=True,
):
    """
    Compute error using single shooting method

    Parameters
    ----------
    biorbd_model: biorbd.Model
        biorbd model build with the bioMod
    states: MX.sym
       The state of the system
    controls: MX.sym
       The controls of the system
    step: int
        Step of runge kutta integrations
    Tf: int
        final time of optimisation
    duration: int
        Duration of the single shooting problem
    use_activation: bool
        True if activation driven; False if excitation driven
    Returns
    ----------
    Error at the last node of the duration for x integrated in single shooting.
    """
    if Tf < duration:
        raise ValueError(f"Single shooting integration duration must be smaller than ocp duration :{Tf} s")

    get_xdot = xdot_funct(biorbd_model, use_activation=use_activation)
    N = int(x.shape[1] / Tf) * duration
    step_time = duration / N
    x_ss = np.ndarray((x.shape[0], x.shape[1]))
    x_ss[:, 0] = x[:, 0]
    h = step_time / step
    x_tmp = []
    for i in range(1, x.shape[1]):
        for j in range(step):
            if i == 1 and j == 0:
                x_prev = x_ss[:, 0]
            else:
                x_prev = x_tmp
            k1 = dxdt(x_prev, u[:, i - 1], get_xdot)
            k2 = dxdt(x_prev + h / 2 * k1, u[:, i - 1], get_xdot)
            k3 = dxdt(x_prev + h / 2 * k2, u[:, i - 1], get_xdot)
            k4 = dxdt(x_prev + h * k3, u[:, i - 1], get_xdot)
            x_tmp = x_prev + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        x_ss[:, i] = x_tmp
    x_ss = x_ss[:, ::ratio] if use_activation is not True else x_ss
    err = [np.sqrt(np.mean((x[:, N] - x_ss[:, N]) ** 2))]
    return err
