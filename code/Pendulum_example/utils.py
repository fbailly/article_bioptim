import biorbd
import casadi as cas
from bioptim import (
    Problem,
    DynamicsFunctions,
)

def custom_dynamic(states, controls, parameters, nlp):
    q, qdot, tau = DynamicsFunctions.dispatch_q_qdot_tau_data(states, controls, nlp)

    force_vector = cas.MX.zeros(6)
    force_vector[5] = -200*q[0]

    f_ext = biorbd.VecBiorbdSpatialVector()
    f_ext.append(biorbd.SpatialVector(force_vector))
    qddot = nlp.model.ForwardDynamics(q, qdot, tau, f_ext).to_mx()

    dxdt = cas.vertcat(qdot, qddot)

    return dxdt


def custom_configure(ocp, nlp):
    Problem.configure_q_qdot(nlp, as_states=True, as_controls=False)
    Problem.configure_tau(nlp, as_states=False, as_controls=True)
    Problem.configure_dynamics_function(ocp, nlp, custom_dynamic)



