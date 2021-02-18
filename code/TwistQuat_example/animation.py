import numpy as np
import bioviz

Quaternion = True

if Quaternion:
    model_path = "JeChMesh_RootQuat.bioMod"
    q = np.load('q_optim_quaternion')
else:
    model_path = "JeChMesh_8DoF.bioMod"
    q = np.load('q_optim_Euler')

b = bioviz.Viz(model_path)
b.load_movement(q)
b.exec()












