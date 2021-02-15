"""
This is a basic code to plot and animate the results of MHE example.

Please note that before to use this code you have to run main.py to generate results.
"""

import biorbd
import numpy as np
from math import ceil
from scipy.integrate import solve_ivp
import bioviz
import seaborn
import pickle
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import scipy.io as sio
from single_shooting_funct import *

model = 'arm_wt_rot_scap.bioMod'
biorbd_model = biorbd.Model(model)
# Same offset used to compute RMSE

start_delay = 25
T = 8
Ns = 800
Ns = Ns - start_delay
T = T * 800 / Ns
final_offset = 5
init_offset = 0

# Get data from MHE problem
nb_step = "_step5_ERK"  # "", "_step1_stat10", "_step3", "_step5", "_step5_ERK"
mat_content = sio.loadmat(f"Data/MHE_results{nb_step}.mat")
with open(f"Data/sim_ac_8000ms_800sn_REACH2_co_level_0_step5_ERK.bob", "rb") as file:
    data = pickle.load(file)
states = data["data"][0]
controls = data["data"][1]
q_ref = states["q"][:, start_delay:]
dq_ref = states["qdot"][:, start_delay:]
a_ref = states["muscles"][:, start_delay:]
u_ref = controls["muscles"][:, start_delay:]
Ns_mhe = int(mat_content["N_mhe"])
N = mat_content["N_tot"]
ratio = int(mat_content["rt_ratio"])
x_est = mat_content["X_est"]
q_est = mat_content["X_est"][: biorbd_model.nbQ(), :]
dq_est = mat_content["X_est"][biorbd_model.nbQ() : biorbd_model.nbQ() * 2, :]
u_est = mat_content["U_est"]
f_est = mat_content["f_est"]
q_init = mat_content["x_init"][: biorbd_model.nbQ(), ::ratio]
# x_ref = mat_content["x_ref"]
# q_ref = mat_content["x_ref"][: biorbd_model.nbQ(), ::ratio]
# u_ref = mat_content["u_ref"]
f_ref = mat_content["f_ref"]
x_ref = np.concatenate((q_ref, dq_ref, a_ref))
# Single shooting
ss_start_del = 0
x_est = single_shooting(biorbd_model, x_est,  u_est, ss_start_del, step=5, ratio=ratio, T=T, use_activation=True)
x_ref = single_shooting(biorbd_model, x_ref,  u_ref, ss_start_del, step=5, ratio=ratio, T=T, use_activation=False)

# PLOT
seaborn.set_style("whitegrid")
seaborn.color_palette()
t_x = np.linspace(0, T, q_est.shape[1] - init_offset - final_offset)
t_u = np.linspace(0, T, u_est.shape[1] - init_offset - final_offset)
# ----- Plot Q -----#
Q_name = ["Glenohumeral plane of elevation", "Glenohumeral elevation", "Glenohumeral axial rotation", "Elbow flexion"]
fig = plt.figure("Q")
colone_q = 2
ligne_q = ceil(biorbd_model.nbQ() / colone_q)
plt.gcf().subplots_adjust(left=0.06, right=0.99, wspace=0.25, hspace=0.2)
c = 0
q_ref = q_ref[:, ::ratio]
# q_ref = mat_content["x_ref"][: biorbd_model.nbQ(), ::ratio]
# for i in [1, 3]:
for i in [1, 3]:
    fig = plt.subplot(2, colone_q, c + 1)
    plt.xlabel("Time (s)")
    if i == 1:
        plt.ylabel("Joint angle (°)")
    # plt.plot(t_x[ss_start_del:], q_est[i, init_offset+ss_start_del:-final_offset] * 180 / np.pi)
    plt.plot(t_x[ss_start_del:], x_ref[i, init_offset + ss_start_del:-Ns_mhe-final_offset] * 180 / np.pi, alpha=0.8)
    plt.plot(t_x[ss_start_del:], q_ref[i, init_offset+ss_start_del:-Ns_mhe - final_offset] * 180 / np.pi, alpha=0.8)
    # plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    # plt.plot(t_x[ss_start_del:], x[i, ss_start_del+ init_offset:-final_offset] *180/np.pi)
    plt.title(Q_name[i])
    c += 1
    if i == 1:
        plt.legend(
            labels=["Estimation", "single shooting Reference", "Reference", "single shooting Estimation"],
            bbox_to_anchor=(1.11, 1.1),
            loc="upper center", borderaxespad=0.0,
            ncol=2,
            frameon=False
        )
muscles_names = [
    #"Pec Sternal",
    # "Pec Rib",
    # "Lat Thoracic",
    # "Lat Lumbar",
    # "Lat Iliac",
    # "Delt Posterior",
    "Tri Long",
    # "Tri Lat",
    # "Tri Med",
    # "Brachial",
    # "Brachioradial",
    # "Pec Clavicular",
    # "Delt Anterior",
    "Delt Middle",
    # "Supraspin",
    "Infraspin",
    # "Subscap",
    # "Bic Long",
    "Bic Short",
]
fest_to_plot = f_est[[6, 13, 15, 18], :]
fref_to_plot = f_ref[[6, 13, 15, 18], :]

# ----- Plot muscle force -----#
fig = plt.figure("Muscles Forces")
plt.gcf().subplots_adjust(left=0.05, right=0.99, wspace=0.15, hspace=0.15, top=0.75, bottom=0.25,)
for i in range(len(muscles_names)):
# for i in range(19):
    fig = plt.subplot(1, 4, i + 1)
    plt.xlabel("Time (s)")
    if i == 0:
        plt.ylabel("Muscle Force(N)")
    plt.plot(t_u, fest_to_plot[i, init_offset:-final_offset])
    plt.plot(t_u, fref_to_plot[i, init_offset:-Ns_mhe-final_offset], alpha=0.8)

    # plt.plot(t_u, f_est[i, init_offset:-final_offset])
    # plt.plot(t_u, f_ref[i, init_offset:-Ns_mhe-final_offset], alpha=0.8)
    plt.title(muscles_names[i])
    # plt.title(biorbd_model.muscleNames()[i].to_string())

    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    if i == 1:
        plt.legend(labels=["Estimation", "Reference"],
                   bbox_to_anchor=(1.15, 1.15),
                   loc="upper center", borderaxespad=0.0,
                   ncol=2,
                   frameon=False)
plt.show()


