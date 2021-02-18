"""
This is a basic code to plot and animate the results of MHE example.

Please note that before to use this code you have to run main.py to generate results.
"""

import biorbd
import numpy as np
import seaborn
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import scipy.io as sio

model = "arm_wt_rot_scap.bioMod"
biorbd_model = biorbd.Model(model)

# Same offset used to compute RMSE
T = 8
Ns = 800
final_offset = 1
init_offset = 1

# Get data from MHE problem
mat_content = sio.loadmat(f"Data/MHE_results.mat")
Ns_mhe = int(mat_content["N_mhe"])
ratio = int(mat_content["rt_ratio"])
x_est = mat_content["X_est"]
q_est = mat_content["X_est"][: biorbd_model.nbQ(), :]
dq_est = mat_content["X_est"][biorbd_model.nbQ() : biorbd_model.nbQ() * 2, :]
u_est = mat_content["U_est"]
f_est = mat_content["f_est"]
x_ref = mat_content["x_ref"]
q_ref = mat_content["x_ref"]
u_ref = mat_content["u_ref"]
f_ref = mat_content["f_ref"]

# PLOT
seaborn.set_style("whitegrid")
seaborn.color_palette()
q_ref = q_ref[:, ::ratio]
t_x = np.linspace(0, T, q_est.shape[1] - init_offset - final_offset)
t_u = np.linspace(0, T, u_est.shape[1] - init_offset - final_offset)

# ----- Plot Q -----#
size_police = 12
Q_name = ["Glenohumeral plane of elevation", "Glenohumeral elevation", "Glenohumeral axial rotation", "Elbow flexion"]
fig = plt.figure("MHE_Results")
grid = plt.GridSpec(2, 4, wspace=0.15, hspace=0.4, left=0.06, right=0.99)
for i in [1, 3]:
    fig = plt.subplot(grid[0, :2]) if i == 1 else plt.subplot(grid[0, 2:])
    plt.xlabel("Time (s)", fontsize=size_police)
    if i == 1:
        plt.ylabel("Joint angle (°)", fontsize=size_police)
    plt.plot(t_x, x_est[i, init_offset:-final_offset] * 180 / np.pi)
    plt.plot(t_x, q_ref[i, init_offset : -Ns_mhe - final_offset] * 180 / np.pi, alpha=0.8)
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    plt.title(Q_name[i], fontsize=size_police)
    if i == 1:
        plt.legend(
            labels=["Estimation", "Reference"],
            bbox_to_anchor=(1.05, 1.2),
            loc="upper center",
            borderaxespad=0.0,
            ncol=2,
            frameon=False,
            fontsize=size_police,
        )

# ----- Plot muscle force -----#
muscles_names = ["Tri Long", "Delt Middle", "Infraspin", "Bic Short"]
fest_to_plot = f_est[[6, 13, 15, 18], :]
fref_to_plot = f_ref[[6, 13, 15, 18], :]
for i in range(len(muscles_names)):
    fig = plt.subplot(grid[1, i])
    plt.xlabel("Time (s)", fontsize=size_police)
    if i == 0:
        plt.ylabel("Muscle Force(N)", fontsize=size_police)
    plt.plot(t_u, fest_to_plot[i, init_offset:-final_offset])
    plt.plot(t_u, fref_to_plot[i, init_offset : -Ns_mhe - final_offset], alpha=0.8)
    plt.title(muscles_names[i], fontsize=size_police)
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
plt.show()
