import numpy as np
import matplotlib.pyplot as plt
import seaborn

q_euler = np.load('q_optim_Euler_42.npy')
q_quat = np.load('q_optim_quaternion_42.npy')

time_vector = np.linspace(0, 100, 101)

seaborn.set_style("whitegrid")
seaborn.color_palette()

fig = plt.figure("Arm strategies")
plt.gcf().subplots_adjust(bottom=0.15, top=0.9, left=0.2, right=0.95, hspace=0.1)
# grid = plt.GridSpec(2, 1, wspace=0.15, hspace=0.4, left=0.3, right=0.99)

ax0 = plt.subplot(2, 1, 1)
# ax0.xlabel("Time [s]", fontsize=12)
ax0.tick_params(axis='x', labelcolor='w')
ax0.set_ylabel("Right arm\nposition [°]", fontsize=15)
ax0.set_xlim(0, 100)
ax0.plot(time_vector, -q_euler[6, :] * 180/np.pi, label='Euler angles')
ax0.plot(time_vector, -q_quat[6, :] * 180/np.pi, label='Quaternion')

ax1 = plt.subplot(2, 1, 2)
ax1.set_xlabel("Time [%]", fontsize=15)
ax1.set_ylabel("Left arm\nposition [°]", fontsize=15)
ax1.set_xlim(0, 100)
l1 = ax1.plot(time_vector, q_euler[7, :] * 180/np.pi, label='Euler angles')
l2 = ax1.plot(time_vector, q_quat[7, :] * 180/np.pi, label='Quaternion')

ax1.legend(bbox_to_anchor=(0.5, 2.35),
            loc="upper center",
            borderaxespad=0.0,
            frameon=False,
            ncol=2,
            fontsize=15,
        )

plt.savefig('Twisting_armTech.eps', format='eps')
plt.show()
