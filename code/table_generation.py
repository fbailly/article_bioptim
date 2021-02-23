from Pendule_example.main import generate_table as pendulum_table
from jumper.main import generate_table as jumper_table


import numpy as np
from bioptim import Shooting


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
                self.single_shoot_error = -1

            def print(self):
                print(f"\t\tsolver = {self.name}")
                print(f"\t\t\titerations = {self.n_iteration}")
                print(f"\t\t\tcost = {self.cost}")
                print(f"\t\t\tconvergence_time = {self.convergence_time}")
                print(f"\t\t\tsingle_shoot_error = {self.single_shoot_error}")

            def compute_error_single_shooting(self, sol, duration):
                sol_int = sol.integrate(shooting_type=Shooting.SINGLE_CONTINUOUS, merge_phases=True)
                sol_merged = sol.merge_phases()
                if sol_merged.phase_time[-1] < duration:
                    raise ValueError(
                        f'Single shooting integration duration must be smaller than ocp duration :{sol_merged.phase_time[-1]} s')

                sn_1s = int(sol_int.ns / sol_int.phase_time[-1] * duration)  # shooting node at {duration} second
                self.single_shoot_error = np.sqrt(np.mean((sol_int.states['all'][:, sn_1s] - sol_merged.states['all'][:, sn_1s]) ** 2))


table = TableOCP()

table.add("pendulum")
table.add("jumper")

pendulum_table(table["pendulum"])
jumper_table(table["jumper"])

table.print()
