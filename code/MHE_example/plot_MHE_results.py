import biorbd
import numpy as np
from math import ceil
import bioviz
import seaborn
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import scipy.io as sio

model = 'arm_wt_rot_scap.bioMod'
biorbd_model = biorbd.Model(model)
start_delay = 25
T = 8
Ns = 800
T = T * 800 / Ns

# Get data from MHE problem
mat_content = sio.loadmat(f"MHE_results.mat")
Ns_mhe = int(mat_content["N_mhe"])
N = mat_content["N_tot"]
Ns = int(N - Ns_mhe)
ratio = int(mat_content["rt_ratio"])
q_est = mat_content["X_est"][: biorbd_model.nbQ(), :]
u_est = mat_content["U_est"]
f_est = mat_content["f_est"]
q_init = mat_content["x_init"][: biorbd_model.nbQ(), ::ratio]
q_ref = mat_content["x_ref"][: biorbd_model.nbQ(), ::ratio]
u_ref = mat_content["u_sol"]
f_ref = mat_content["f_ref"]

Q_name = ["Glenohumeral plane of elevation", "Glenohumeral elevation", "Glenohumeral axial rotation", "Elbow flexion"]

t_x = np.linspace(0, T, q_est.shape[1])
t_u = np.linspace(0, T, u_est.shape[1])

seaborn.set_style("whitegrid")
seaborn.color_palette()

# ----- Plot Q -----#
fig = plt.figure("Q")
colone_q = 2
ligne_q = ceil(biorbd_model.nbQ() / colone_q)
plt.gcf().subplots_adjust(left=0.06, right=0.99, wspace=0.25, hspace=0.2)
for i in range(biorbd_model.nbQ()):
    fig = plt.subplot(ligne_q, colone_q, i + 1)
    if i in [2, 3]:
        plt.xlabel("Time (s)")
    else:
        fig.set_xticklabels([])
    if i in [0, 2]:
        plt.ylabel("Joint angle (°)")
    plt.plot(t_x, q_est[i, :] * 180 / np.pi)
    plt.plot(t_x, q_ref[i, :-Ns_mhe] * 180 / np.pi, alpha=0.8)
    plt.title(Q_name[i])
    if i == 0:
        plt.legend(
            labels=["Estimation", "Reference"],
            bbox_to_anchor=(1.1, 1.25),
            loc="upper center", borderaxespad=0.0,
            ncol=2,
            frameon=False
        )

muscles_names = [
    "Tri Long",
    "Tri Lat",
    "Tri Med",
    "Brachial",
    "Brachioradial",
    "Pec Clavicular",
    "Delt Middle",
    "Infraspin",
    "Subscap",
    "Bic Long",
    "Bic Short",
]

significant_fest = f_est[[6, 7, 8, 9, 10, 12, 14, 15, 16, 17, 18], :]
significant_fref = f_ref[[6, 7, 8, 9, 10, 12, 14, 15, 16, 17, 18], :]

# ----- Plot muscle force -----#
fig = plt.figure("Muscles Forces")
plt.gcf().subplots_adjust(left=0.06, right=0.99, wspace=0.2, hspace=0.2)
for i in range(len(muscles_names)):
    fig = plt.subplot(3, 4, i + 1)
    if i in [7, 8, 9, 10]:
        plt.xlabel("Time (s)")
    else:
        fig.set_xticklabels([])
    if i in [0, 4, 8]:
        plt.ylabel("Muscle Force(N)")
    plt.plot(t_u, significant_fest[i, :])
    plt.plot(t_u, significant_fref[i, :-Ns_mhe], alpha=0.8)
    plt.title(muscles_names[i])
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
plt.legend(labels=["Estimation", "Reference"],
           bbox_to_anchor=(1.05, 0.80),
           loc="upper left",
           frameon=False)
plt.show()

# ------ Animate ------ #
b = bioviz.Viz(model)
b.load_movement(q_est)
b.exec()
