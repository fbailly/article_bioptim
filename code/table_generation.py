from Pendulum_example.main import generate_table as pendulum_table
from muscle_exc_pointing.main import generate_table as pointing_table
from TwistQuat_example.main import generate_table as TwistQuat_table
from Example_multiphase_walking.main import generate_table as gait_table
from jumper.main import generate_table as jumper_table



import numpy as np
from bioptim import Shooting
import casadi as cas

class TableOCP:
    def __init__(self):
        self.cols = None

    def add(self, name):
        if not self.cols:
            self.cols = [TableOCP.OCP(name)]
        else:
            self.cols.append(TableOCP.OCP(name))

    def __getitem__(self, item_name):
        return self.cols[[col.name for col in self.cols].index(item_name)]

    def print(self):
        for col in self.cols:
            col.print()

    class OCP:
        def __init__(self, name):
            self.name = name
            self.nx = -1
            self.nu = -1
            self.ns = -1
            self.solver = []

        def print(self):
            print(f"task = {self.name}")
            print(f"\tns = {self.ns}")
            print(f"\tnx = {self.nx}")
            print(f"\tnu = {self.nu}")
            for solver in self.solver:
                solver.print()

        class Solver:
            def __init__(self, name):
                self.name = name
                self.n_iteration = -1
                self.cost = 0
                self.convergence_time = -1
                self.single_shoot_error_t = -1
                self.single_shoot_error_r = -1

            def print(self):
                print(f"\t\tsolver = {self.name}")
                print(f"\t\t\titerations = {self.n_iteration}")
                print(f"\t\t\tcost = {self.cost}")
                print(f"\t\t\tconvergence_time (s) = {self.convergence_time}")
                print(f"\t\t\tsingle_shoot_error translation (mm) = {self.single_shoot_error_t}")
                print(f"\t\t\tsingle_shoot_error rotation (°) = {self.single_shoot_error_r}")

            def compute_error_single_shooting(self, sol, duration, use_final_time=False):
                sol_merged = sol.merge_phases()

                if sol_merged.phase_time[-1] < duration and not use_final_time:
                    raise ValueError(
                        f'Single shooting integration duration must be smaller than ocp duration :{sol_merged.phase_time[-1]} s. '
                        f'You can set use_final_time=True if you want to use the final time for the Single shooting integration duration')

                trans_idx = []
                rot_idx = []
                for i in sol.ocp.nlp[0].mapping['q'].to_second.map_idx:
                    if i is not None:
                        if sol.ocp.nlp[0].model.nameDof()[i].to_string()[-4:-1] == 'Rot':
                            rot_idx += [i]
                        else:
                            trans_idx += [i]
                rot_idx = np.array(list(set(rot_idx)))
                trans_idx = np.array(list(set(trans_idx)))

                sol_int = sol.integrate(shooting_type=Shooting.SINGLE_CONTINUOUS, merge_phases=True, keepdims=True)
                if use_final_time:
                    sn_1s = -1
                else:
                    sn_1s = int(sol_int.ns[0] / sol_int.phase_time[-1] * duration)  # shooting node at {duration} second
                if len(rot_idx) > 0:
                    self.single_shoot_error_r = np.sqrt(
                            np.mean((sol_int.states['q'][rot_idx, sn_1s] - sol_merged.states['q'][rot_idx, sn_1s]) ** 2))\
                                                * 180 / np.pi
                else:
                    self.single_shoot_error_r = 'N.A.'
                if len(trans_idx) > 0:
                    self.single_shoot_error_t = np.sqrt(np.mean(
                        (sol_int.states['q'][trans_idx, sn_1s] - sol_merged.states['q'][trans_idx, sn_1s]) ** 2)) / 1000
                else:
                    self.single_shoot_error_t = 'N.A.'



table = TableOCP()

# table.add("pointing")
# table.add("pendulum")
# table.add("jumper")
table.add("TwistQuat_quaternion")
table.add("TwistQuat_euler")
# table.add("gait")

# pointing_table(table["pointing"])
# pendulum_table(table["pendulum"])
# jumper_table(table["jumper"])
TwistQuat_table(table["TwistQuat_quaternion"], True)
TwistQuat_table(table["TwistQuat_euler"], False)
# gait_table(table["gait"])

table.print()