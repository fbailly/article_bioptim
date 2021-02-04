
import biorbd
import casadi as cas
import numpy as np
import matplotlib.pyplot as plt
import bioviz


from bioptim import (
    OptimalControlProgram,
    DynamicsList,
    Problem,
    ObjectiveList,
    DynamicsFunctions,
    ObjectiveFcn,
    Bounds,
    OdeSolver,
    BoundsList,
    InitialGuessList,
    Data,
    InterpolationType,
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
    Problem.configure_forward_dyn_func(ocp, nlp, custom_dynamic)


def prepare_ocp_toile_2phase_pendule(biorbd_model_path, ode_solver=OdeSolver.RK):
    # --- Options --- #
    # Model path
    biorbd_model = (
        biorbd.Model(biorbd_model_path),
        biorbd.Model(biorbd_model_path),
    )

    # Problem parameters
    number_shooting_points = (50,
                              50,
                              )
    final_time = (5,
                  5,
                  )
    tau_min, tau_max, tau_init = -500, 500, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, weight=1, index=0,
                            target=np.ones((1, number_shooting_points[0] + 1)) * -0.5, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=1e-6, phase=1)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(custom_configure, dynamic_function=custom_dynamic)
    dynamics.add(custom_configure, dynamic_function=custom_dynamic)

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


    # ------------- #

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
        ode_solver=ode_solver,
    )



if __name__ == "__main__":
    model_path_toile = "/home/user/Documents/Programmation/Eve/Modeles/MassPoint_pendule.bioMod"

    ocp = prepare_ocp_toile_2phase_pendule(biorbd_model_path=model_path_toile)

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=False)

    Solution_data = Data.get_data(ocp, sol, get_states=True, get_controls=True, get_parameters=True)
    q = Solution_data[0]['q']# [0]
    qdot = Solution_data[0]['q_dot']# [0]
    u = Solution_data[1]['tau']# [0]

    b = bioviz.Viz(model_path_toile)
    b.load_movement(q)
    b.exec()

    time_vector = np.linspace(0, 10, 101)
    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    axs = axs.ravel()

    axs[0].set_title('Position')
    axs[0].plot(time_vector[:-1], q[0, :-1], '-', color='tab:blue', label='Mass')
    axs[0].plot(np.array([5, 5]), np.array([min(q[0, :-1]), max(q[0, :-1])]), '--', color='tab:gray')
    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel('Mass position [m]', color='tab:blue')
    axs[0].tick_params(axis='y', labelcolor='tab:blue')

    ax0 = axs[0].twinx()
    ax0.plot(time_vector[:-1], q[1, :-1], '-', color='tab:red', label='Pendulum')
    ax0.set_ylabel('Pendulum position [rad]', color='tab:red')
    ax0.tick_params(axis='y', labelcolor='tab:red')

    axs[1].set_title('Velocity')
    axs[1].plot(time_vector[:-1], qdot[0, :-1], '-', color='tab:blue', label='Mass')
    axs[1].plot(np.array([5, 5]), np.array([min(qdot[0, :-1]), max(qdot[0, :-1])]), '--', color='tab:gray')
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('Mass velocity [m/s]', color='tab:blue')
    axs[1].tick_params(axis='y', labelcolor='tab:blue')

    ax1 = axs[1].twinx()
    ax1.plot(time_vector[:-1], qdot[1, :-1], '-', color='tab:red', label='Pendulum')
    ax1.set_ylabel('Pendulum velocity [rad/s]', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    axs[2].set_title('Forces')
    axs[2].plot(time_vector[:-1], u[0, :-1], '-', color='tab:blue', label='Mass')
    axs[2].plot(np.array([5, 5]), np.array([min(u[0, :-1]), max(u[0, :-1])]), '--', color='tab:gray')
    axs[2].set_xlabel('Time [s]')
    axs[2].set_ylabel('Mass force actuation [N]', color='tab:blue')
    axs[2].tick_params(axis='y', labelcolor='tab:blue')

    ax2 = axs[2].twinx()
    ax2.plot(time_vector[:-1], -10*q[0, :-1], '-', color='tab:green', label='Spring')
    ax2.set_ylabel('Spring external force [N]', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    fig.tight_layout()
    plt.show()
    plt.savefig('Mass_Pendulum_Fext.png', dpi=900)






