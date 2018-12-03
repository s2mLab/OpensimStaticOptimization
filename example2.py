import ipopt
from pyomeca import Analogs3d
from matplotlib import pyplot as plt

from OpensimStaticOptimization import *

# ---- SETUP ---- #
# Paths
model_path = f"arm26.osim"
mot_path = f"arm26_InverseKinematics.mot"
low_pass_filter_param = None  # (4, 6) is also usually performed
use_muscle_physiology = True

# Optimization type
# so_model = ClassicalStaticOptimization(model_path, mot_path, low_pass_filter_param=low_pass_filter_param)
so_model0 = ClassicalStaticOptimization(model_path, mot_path,
                                       low_pass_filter_param=low_pass_filter_param,
                                       use_muscle_physiology=use_muscle_physiology)

so_model = LocalOptimizationLinearConstraints(model_path, mot_path,
                                               low_pass_filter_param=low_pass_filter_param,
                                               use_muscle_physiology=use_muscle_physiology)


# Optim options
activation_initial_guess = np.zeros([so_model.n_muscles])
lb0, ub0 = so_model0.get_bounds()
# --------------- #

prob0 = ipopt.problem(
        n=so_model.n_muscles,  # Nb of variables
        lb=lb0,  # Variables lower bounds
        ub=ub0,  # Variables upper bounds
        m=so_model.n_dof,  # Nb of constraints
        cl=np.zeros(so_model.n_dof),  # Lower bound constraints
        cu=np.zeros(so_model.n_dof),  # Upper bound constraints
        problem_obj=so_model0  # Class that defines the problem
    )

prob0.addOption('tol', 1e-7)
prob0.addOption('print_level', 0)

activations = list()

frame0=1
x=np.nan

while np.isnan(x):
    so_model0.upd_model_kinematics(frame0)

    condition = False
    # Optimize
    try:
        x, info = prob0.solve(activation_initial_guess)

    except RuntimeError:
        print(f"Error while computing the frame {frame}.")
        x = np.ndarray(activation_initial_guess.shape) * np.nan


    # The answer is the initial guess for next frame
    activation_initial_guess = x
    activations.append(x)
    print(f"time = {so_model.get_time(frame)}, Performance = {info.get('obj_val')}, "
          f"Constraint violation = {np.linalg.norm(info.get('g'))}")
    frame0 = frame0+1



activation_frame = x
activation_initial_guess = np.zeros([so_model.n_muscles])
prob = ipopt.problem(
        n=so_model.n_muscles,  # Nb of variables
        m=so_model.n_dof,  # Nb of constraints
        cl=np.zeros(so_model.n_dof),  # Lower bound constraints
        cu=np.zeros(so_model.n_dof),  # Upper bound constraints
        problem_obj=so_model  # Class that defines the problem
    )

prob.addOption('tol', 1e-7)
prob.addOption('print_level', 0)



for frame in range(frame0,so_model.nFrame):
    so_model.set_previous_activations(activation_frame)
    so_model.upd_model_kinematics(frame)
    prob.lb = lb0 / activation_frame # check for div_by_0 ... np.divide
    prob.ub = ub0 / activation_frame # check for div_by_0

    # Optimize
    try:
        x, info = prob.solve(activation_initial_guess)

    except RuntimeError:
        print(f"Error while computing the frame {frame}.")
        x = np.ndarray(activation_initial_guess.shape) * np.nan

    activation_frame = x * activation_frame
    activations.append(x)
    print(f"time = {so_model.get_time(frame)}, Performance = {info.get('obj_val')}, "
          f"Constraint violation = {np.linalg.norm(info.get('g'))}")






data_from_python = Analogs3d(np.array(activations))
data_from_GUI = Analogs3d.from_csv("arm26_StaticOptimization_activation.sto",
                                   delimiter='\t', time_column=0, header=7, first_column=1, first_row=8)

data_from_python.plot()
data_from_GUI.plot()
(data_from_python - data_from_GUI).plot()


plt.show()
