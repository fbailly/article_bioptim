import numpy as np
import matplotlib.pyplot as plt
import bioviz

model_path = "MassPoint_pendulum.bioMod"

q = np.load('q_optim.npy')
qdot = np.load('qdot_optim.npy')
u = np.load('u_optim.npy')

time_vector = np.linspace(0, 10, 101)

fig = plt.figure(figsize=(20, 5))
plt.gcf().subplots_adjust(left=0.1, right=0.9, wspace=0.35, hspace=0.1, bottom=0.15)

axs_0 = plt.subplot(1, 3, 1)
axs_0.set_title('Position')
axs_0.plot(time_vector[:-1], q[0, :-1], '-', color='#1f77b4', label='Mass Position')
axs_0.axvline(x=5, color="k", linewidth=0.7)
axs_0.set_xlabel('Time [s]')
axs_0.set_ylabel('Mass position [m]', color='#1f77b4')
axs_0.tick_params(axis='y', labelcolor='#1f77b4')

ax0 = axs_0.twinx()
ax0.plot(0, 0, '-', color='#1f77b4', label='Mass Position')
ax0.plot(time_vector[:-1], q[1, :-1], '-', color='#ff7f0e', label='Pendulum Position')
ax0.set_ylabel('Pendulum position [rad]', color='#ff7f0e')
ax0.set_ylim(top=15)
ax0.tick_params(axis='y', labelcolor='#ff7f0e')
ax0.legend(loc="upper right", borderpad=0.5)

axs_1 = plt.subplot(1, 3, 2)
axs_1.set_title('Velocity')
axs_1.plot(time_vector[:-1], qdot[0, :-1], '-', color='#1f77b4', label='Mass Velocity')
l1, = axs_1.plot(np.array([5, 5]), np.array([-0.55, 0.5]), '-', linewidth=0.7, color='k')
axs_1.set_xlabel('Time [s]')
axs_1.set_ylabel('Mass velocity [m/s]', color='#1f77b4')
axs_1.set_ylim(-0.55, 0.5)
axs_1.tick_params(axis='y', labelcolor='#1f77b4')

ax1 = axs_1.twinx()
lines1, = ax1.plot(0, 0, '-', color='#1f77b4', label='Mass Velocity')
lines2, = ax1.plot(time_vector[:-1], qdot[1, :-1], '-', color='#ff7f0e', label='Pendulum Velocity')
ax1.set_ylabel('Pendulum velocity [rad/s]', color='#ff7f0e')
ax1.set_ylim(top=9)
ax1.tick_params(axis='y', labelcolor='#ff7f0e')

ax1.legend(
    handles=[lines1, lines2],
    labels=['Mass Velocity', 'Pendulum Velocity'],
    loc="upper right",
)
axs_1.legend(
    handles=[l1],
    labels=["Phase Transition"],
    bbox_to_anchor=(0.5, 1.115),
    loc="upper center", borderaxespad=0.0,
    frameon=False
)

axs_2 = plt.subplot(1, 3, 3)
axs_2.set_title('Forces')
axs_2.step(time_vector[:-1], u[0, :-1], '-', color='#1f77b4', label='Mass Force')
axs_2.axvline(x=5, color="k", linewidth=0.7)
axs_2.set_xlabel('Time [s]')
axs_2.set_ylabel('Mass force actuation [N]', color='#1f77b4')
axs_2.set_ylim(top=47)
axs_2.tick_params(axis='y', labelcolor='#1f77b4')

ax2 = axs_2.twinx()
ax2.plot(0, 0, '-', color='#1f77b4', label='Mass Force')
ax2.step(time_vector[:-1], -10*q[0, :-1], '-', color='#2ca02c', label='Spring Force')
ax2.set_ylabel('Spring external force [N]', color='#2ca02c')
ax2.tick_params(axis='y', labelcolor='#2ca02c')
ax2.legend(loc="upper right", borderpad=0.5)

# plt.show()
plt.savefig('Mass_Pendulum_Fext.eps', format='eps')
plt.show()

print('RMS q_m / q*_m : ', np.std(q[0, 51:]-0.5))

b = bioviz.Viz(model_path)
b.load_movement(q)
b.exec()


