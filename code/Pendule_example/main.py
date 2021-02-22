"""
This is a basic example on how to use external forces to model a spring.
The mass attached to the spring must stabilize its position during the second phase of the movement while perturbed by
the oscillation of a pendulum.
"""

import biorbd
import numpy as np
from time import time
import utils

from bioptim import (
    OptimalControlProgram,
    DynamicsList,
    ObjectiveList,
    ObjectiveFcn,
    Bounds,
    BoundsList,
    InitialGuessList,
    InterpolationType,
    Shooting,
)

def prepare_ocp(biorbd_model_path: str) -> OptimalControlProgram:
    """
    Prepare the ocp
    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod
    Returns
    -------
    The OptimalControlProgram ready to be solved
    """
    # Model path
    biorbd_model = (
        biorbd.Model(biorbd_model_path),
        biorbd.Model(biorbd_model_path),
    )

    # Problem parameters
    number_shooting_points = (50, 50, )
    final_time = (5, 5, )
    tau_min, tau_max, tau_init = -500, 500, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, weight=1, index=0,
                            target=np.ones((1, number_shooting_points[0] + 1)) * -0.5, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=1e-6, phase=1)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(utils.custom_configure, dynamic_function=utils.custom_dynamic)
    dynamics.add(utils.custom_configure, dynamic_function=utils.custom_dynamic)

    # Path constraint
    X_bounds = BoundsList()
    X_bounds.add(bounds=Bounds(np.array([-10, -4 * np.pi, -1000, -1000]), np.array([10, 4 * np.pi, 1000, 1000]),
                               interpolation=InterpolationType.CONSTANT))
    X_bounds.add(bounds=Bounds(np.array([-10, -4*np.pi, -1000, -1000]), np.array([10, 4*np.pi, 1000, 1000]),
                               interpolation=InterpolationType.CONSTANT))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add(bounds=Bounds([0, 0], [0, 0]))
    u_bounds.add(bounds=Bounds([tau_min, 0], [tau_max, 0]))

    # Initial guess
    x_init = InitialGuessList()
    x_init.add(np.random.random(biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))
    x_init.add(np.random.random(biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))

    u_init = InitialGuessList()
    u_init.add([0, 0])
    u_init.add([0, 0])

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        number_shooting_points,
        final_time,
        x_init,
        u_init,
        X_bounds,
        u_bounds,
        objective_functions,
    )


if __name__ == "__main__":
    model_path = "MassPoint_pendulum.bioMod"
    np.random.seed(0)

    ocp = prepare_ocp(biorbd_model_path=model_path)

    # --- Solve the program --- #
    tic = time()
    sol = ocp.solve(show_online_optim=False)
    toc = time() - tic

    def compute_error_single_shooting(ocp, sol, duration):
        if ocp.nlp[0].tf < duration:
            raise ValueError(
                f'Single shooting integration duration must be smaller than ocp duration :{ocp.nlp[0].tf} s')
        sol_int = sol.integrate(shooting_type=Shooting.SINGLE, continuous=True)
        sn_1s = int(ocp.nlp[0].ns / ocp.nlp[0].tf * duration)  # shooting node at {duration} second
        err_0 = np.sqrt(np.mean((sol_int.states[0]['all'][:, 5 * sn_1s] - sol.states[0]['all'][:, sn_1s]) ** 2))
        err_1 = np.sqrt(np.mean((sol_int.states[0]['all'][:, 5 * sn_1s] - sol.states[0]['all'][:, sn_1s]) ** 2)) # J'imagine que tu vas tout uniformiser de toutes facons...
        return err_0

    # result = ShowResult(ocp, sol)
    # result.objective_functions()
    ss_err = compute_error_single_shooting(ocp, sol, 1)
    print("*********************************************")
    print(f"Single shooting error : {ss_err}")
    print(f"Time to solve : {toc}sec")

    q = np.hstack((sol.states[0]['q'], sol.states[1]['q']))
    qdot = np.hstack((sol.states[0]['qdot'], sol.states[1]['qdot']))
    u = np.hstack((sol.controls[0]['tau'], sol.controls[1]['tau']))

    np.save('q_optim', q)
    np.save('qdot_optim', qdot)
    np.save('u_optim', u)
