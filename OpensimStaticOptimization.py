import numpy as np

from scipy.optimize import approx_fprime

import opensim as osim

# ---- GUIDE ---- #
# You will find in this file anything you need to call IPOPT in order to perform Static Optimization using Opensim as
# basis for the dynamics.
# DYNAMIC MODELS at the bottom of this file are actually the model to send to IPOPT. So far ClassicalStaticOptimization
# and ClassicalOptimizationLinearConstraints are defined.
#
# If you want to create your own optimization procedure, you are very welcome. Your class could derive from the three
# mandatory section (or you can reimplement them) a other optional classes:
# KinematicModel => Prepare the state variable
# Objective => Provide Objective and Gradient methods
# Constraints => Provide Constraints and Jacobian methods
# ResidualForces => Provides method to add residual forces
# ExternalForces => Provides method to add external forces


# ---- KINEMATICS METHOD CLASS ---- #
# -- The dynamic model should inherit from this class in order to get the proper kinematics -- #
class KinematicModel:
    def __init__(self, model_path, mot_path, low_pass_filter_param=None):
        # Load the model
        self.model = osim.Model(model_path)
        self.state = self.model.initSystem()

        # Prepare data
        self.actual_q = []
        self.actual_qdot = []
        self.actual_qddot = []

        # Get some reference that will be modified during the optimization (for speed sake)
        self.n_dof = self.state.getQ().size()
        self.muscle_actuators = self.model.getMuscles()
        self.n_muscles = self.muscle_actuators.getSize()
        self.actuators = self.model.getForceSet()
        self.n_actuators = self.actuators.getSize()

        # Get the data
        self.__data_storage = osim.Storage(mot_path)
        if low_pass_filter_param is None:
            self.should_low_pass_filter = False
        else:
            self.should_low_pass_filter = True
            self.low_pass_order = low_pass_filter_param[0]
            self.low_pass_frequency = low_pass_filter_param[1]
        self.__dispatch_kinematics()

    def get_time(self, frame):
        return self.__data_storage.getStateVector(frame).getTime()

    def __dispatch_kinematics(self):
        # Load the kinematics from inverse dynamics
        self.model.getSimbodyEngine().convertDegreesToRadians(self.__data_storage)
        if self.should_low_pass_filter:
            self.__data_storage.lowpassFIR(self.low_pass_order, self.low_pass_frequency)  # TODO read value in xml file or class parameters  MICKAEL removed filter
        self.gen_coord_function = \
            osim.GCVSplineSet(5, self.__data_storage)  # TODO read value in xml file or class parameters
        self.nFrame = self.__data_storage.getSize()

        # Dispatch the kinematics
        self.all_q = []
        self.all_qdot = []
        self.all_qddot = []
        for frame in range(self.nFrame):
            q = list()
            qdot = list()
            qddot = list()
            # TODO Make sure Q and storageQ are in the same order
            for idx_q in range(self.__data_storage.getStateVector(frame).getSize()):
                q.append(self.gen_coord_function.evaluate(idx_q, 0,
                                                          self.__data_storage.getStateVector(frame).getTime()))
                qdot.append(self.gen_coord_function.evaluate(idx_q, 1,
                                                             self.__data_storage.getStateVector(frame).getTime()))
                qddot.append(self.gen_coord_function.evaluate(idx_q, 2,
                                                              self.__data_storage.getStateVector(frame).getTime()))
            self.all_q.append(q)
            self.all_qdot.append(qdot)
            self.all_qddot.append(qddot)

        self.upd_model_kinematics(0)

    def upd_model_kinematics(self, frame):
        # Get a fresh state
        self.model.initStateWithoutRecreatingSystem(self.state)
        self.state.setTime(self.get_time(frame))

        # Update kinematic states
        self.actual_q = self.all_q[frame]
        self.actual_qdot = self.all_qdot[frame]
        self.actual_qddot = self.all_qddot[frame]
        self.state.setQ(osim.Vector(self.actual_q))
        self.state.setU(osim.Vector(self.actual_qdot))
        self.model.realizeVelocity(self.state)


# ---- Forces ---- #
class ResidualForces:
    def __init__(self,  residual_actuator_xml_path=None):
        # Add residual forces from an XML file is asked
        if residual_actuator_xml_path is not None:
            force_set = osim.ArrayStr()
            force_set.append(residual_actuator_xml_path)

            analyze_tool = osim.AnalyzeTool(self.model)
            analyze_tool.setModel(self.model)
            analyze_tool.setForceSetFiles(force_set)
            analyze_tool.updateModelForces(self.model, residual_actuator_xml_path)
            self.state = self.model.initSystem()

            self.n_actuators = self.actuators.getSize()

            fs = self.model.getForceSet()
            for i in range(fs.getSize()):
                act = osim.CoordinateActuator.safeDownCast(fs.get(i))
                if act:
                    act.overrideActuation(self.state, True)


class ExternalForces:
    def __init__(self, external_loads_xml_path=None):
        # Add residual forces from an XML file is asked
        if external_loads_xml_path is not None:
            # TODO To be tested
            analyze_tool = osim.AnalyzeTool(self.model)
            analyze_tool.setModel(self.model)
            analyze_tool.setExternalLoadsFileName(external_loads_xml_path)


# ---- OBJECTIVES CLASS ---- #
# ObjMinimizeActivation minimizes the absolute activation raised at the activation_exponent (with 2 as default)
class ObjMinimizeActivation:
    def __init__(self, activation_exponent=2):
        self.activation_exponent = activation_exponent

        # Aliases
        self.muscle_actuators = self.model.getMuscles()
        self.n_muscles = self.muscle_actuators.getSize()

    def objective(self, x):
        #
        # The callback for calculating the objective
        #
        return np.power(np.abs(x), self.activation_exponent).sum()

    def gradient(self, x):
        #
        # The callback for calculating the gradient
        #
        return self.activation_exponent * np.power(np.abs(x), self.activation_exponent - 1)

    def get_bounds(self):
        forces = self.model.getForceSet()
        activation_min = []
        activation_max = []
        for i in range(forces.getSize()):
            f = osim.CoordinateActuator.safeDownCast(forces.get(i))
            if f:
                activation_min.append(f.get_min_control())
                activation_max.append(f.get_max_control())
            m = osim.Muscle.safeDownCast(forces.get(i))
            if m:
                activation_min.append(m.get_min_control())
                activation_max.append(m.get_max_control())
        return activation_min, activation_max


# ---- CONSTRAINTS CLASS ---- #
# ConstraintAccelerationTarget create a strict constraint for the optimization such as the generalized accelerations
# computed from muscle activation equal the generalized accelerations from inverse kinematics
class ConstraintAccelerationTarget:
    def __init__(self):
        pass

    def constraints(self, x, idx=None):
        #
        # The callback for calculating the constraints
        #
        qddot_from_muscles = self.forward_dynamics(x)
        c = [self.actual_qddot[idx_q] - qddot_from_muscles.get(idx_q) for idx_q in range(qddot_from_muscles.size())]
        if idx is not None:
            return c[idx]
        else:
            return c

    def jacobian(self, x):
        jac = np.ndarray([self.n_dof, self.n_actuators])
        for i in range(self.n_dof):
            jac[i, :] = approx_fprime(x, self.constraints, 1e-10, i)
        return jac


# ---- DYNAMIC MODELS ---- #
# These are the actual classes to send to IPOPT

# ClassicalStaticOptimization computes the muscle activations in order to minimize them
# while targeting the acceleration from inverse kinematics.
# This is the most classical approach to Static Opimization
class ClassicalStaticOptimization(KinematicModel,
                                  ObjMinimizeActivation,
                                  ConstraintAccelerationTarget,
                                  ResidualForces,
                                  ExternalForces):
    # TODO WRITE TESTS
    def __init__(self,
                 model_path, mot_path, low_pass_filter_param=None,
                 activation_exponent=2,
                 residual_actuator_xml_path=None,
                 external_loads_xml_path=None):
        KinematicModel.__init__(self, model_path=model_path, mot_path=mot_path, low_pass_filter_param=low_pass_filter_param)
        ObjMinimizeActivation.__init__(self, activation_exponent=activation_exponent)
        ResidualForces.__init__(self, residual_actuator_xml_path=residual_actuator_xml_path)
        ExternalForces.__init__(self, external_loads_xml_path=external_loads_xml_path)

    def forward_dynamics(self, x):
        # Set all residual forces
        fs = self.model.getForceSet()
        for i in range(self.n_muscles, fs.getSize()):
            act = osim.ScalarActuator.safeDownCast(fs.get(i))
            if act:
                act.setOverrideActuation(self.state, x[i])

        # Then update muscles
        muscle_activation = x[:self.n_muscles]
        [self.muscle_actuators.get(m).setActivation(self.state, muscle_activation[m]) for m in range(self.n_muscles)]
        self.model.equilibrateMuscles(self.state)
        self.model.realizeAcceleration(self.state)
        return self.state.getUDot()


# ClassicalOptimizationLinearConstraints intends to mimic the classical approach but with the constraints linearized.
# It makes the assumption that muscle length is constant at a particular position and velocity, whatever the muscle
# activation.
class ClassicalOptimizationLinearConstraints(KinematicModel,
                                             ObjMinimizeActivation,
                                             ConstraintAccelerationTarget,
                                             ResidualForces,
                                             ExternalForces):
    # TODO WRITE TESTS
    def __init__(self,
                 model_path, mot_path, low_pass_filter_param=None,
                 activation_exponent=2,
                 use_muscle_physiology=True,
                 residual_actuator_xml_path=None,
                 external_loads_xml_path=None):
        # Options
        self.use_muscle_physiology = use_muscle_physiology

        # Constructor
        KinematicModel.__init__(self, model_path=model_path, mot_path=mot_path, low_pass_filter_param=low_pass_filter_param)
        ObjMinimizeActivation.__init__(self, activation_exponent=activation_exponent)
        ResidualForces.__init__(self, residual_actuator_xml_path=residual_actuator_xml_path)
        ExternalForces.__init__(self, external_loads_xml_path=external_loads_xml_path)

        # Prepare linear constraint variables
        self.optimal_forces = []
        self.constraint_vector = []
        self.constraint_matrix = []
        self.jacobian_matrix = []  # Precomputed jacobian
        self.__prepare_constraints()
        self.previous_state = []

    def forward_dynamics(self, x):
        fs = self.model.getForceSet()
        for i in range(fs.getSize()):
            act = osim.ScalarActuator.safeDownCast(fs.get(i))
            if act:
                act.setOverrideActuation(self.state, x[i] * self.optimal_forces[i])

        self.model.realizeAcceleration(self.state)
        return self.state.getUDot()

    def upd_model_kinematics(self, frame):
        super().upd_model_kinematics(frame)
        self.__prepare_constraints(np.ones([self.n_muscles]))

    def __prepare_constraints(self, activation):
        self.model.realizeVelocity(self.state)

        # Get optimal param
        forces = self.model.getForceSet()

        self.optimal_forces = []
        for i in range(forces.getSize()):
            muscle = osim.Muscle.safeDownCast(forces.get(i))
            if muscle:
                if self.use_muscle_physiology:
                    self.model.setAllControllersEnabled(True)
                    self.optimal_forces.append(muscle.calcInextensibleTendonActiveFiberForce(self.state, activation[i]))
                    self.model.setAllControllersEnabled(False)
                else:
                    self.optimal_forces.append(muscle.getMaxIsometricForce())
            coordinate = osim.CoordinateActuator.safeDownCast(forces.get(i))
            if coordinate:
                self.optimal_forces.append(coordinate.getOptimalForce())

        # Construct linear constraints
        self.linear_constraints(activation)

    def constraints(self, x, idx=None):
        x_tp = x.reshape((x.shape[0], 1))
        x_mul = np.ndarray((self.constraint_matrix.shape[0], x_tp.shape[1]))
        np.matmul(self.constraint_matrix, x_tp, x_mul)
        x_constraint = x_mul.reshape(x_mul.shape[0])
        c = x_constraint + self.constraint_vector
        if idx is not None:
            return c[idx]
        else:
            return c

    def linear_constraints(self, activation):
        fs = self.model.getForceSet()
        for i in range(fs.getSize()):
            act = osim.ScalarActuator.safeDownCast(fs.get(i))
            if act:
                act.overrideActuation(self.state, True)

        p_vector = np.zeros(self.n_actuators)
        self.constraint_vector = np.array(super().constraints(p_vector))
        self.constraint_vector = np.array(super().constraints(p_vector))

        self.constraint_matrix = np.zeros((self.n_dof, self.n_actuators))
        for p in range(0, self.n_actuators):
            p_vector[p] = activation[p]
            self.constraint_matrix[:, p] = np.array(super().constraints(p_vector)) - self.constraint_vector
            p_vector[p] = 0

    def jacobian(self, x):
        return self.constraint_matrix



class LocalOptimizationLinearConstraints(KinematicModel, ClassicalOptimizationLinearConstraints):
    def set_previous_activation(self, x):
        self.previous_activation = x

    def get_previous_activation(self):
        return self.previous_activation

    def __xfrompreviousx(self, x):
        return x * self.previous_activation # !!! addition will not work !!!

    def forward_dynamics(self, x):
        super().forward_dynamics(x) # x[i] * self.optimal_forces[i]
        #return self.state.getUDot()

    def upd_model_kinematics(self, frame):
        super(KinematicModel, self).upd_model_kinematics(frame)
        self.__prepare_constraints(self.previous_activations)

    def objective(self, x):
        super().objective(self.__xfrompreviousx(x))

    def gradient(self, x):
        super().gradient(self.__xfrompreviousx(x))





