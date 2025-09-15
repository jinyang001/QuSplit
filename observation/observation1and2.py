import pickle
import time
from qiskit_aer.noise import depolarizing_error, thermal_relaxation_error
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit.circuit.library import EfficientSU2
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_algorithms import VQE
from matplotlib import pyplot as plt
from qiskit_aer.noise import NoiseModel

# setup
np.random.seed(42)
algorithm_globals.random_seed = 42
qubit_op = None

H2_op = SparsePauliOp.from_list(
    [
        ("II", -1.052373245772859),
        ("IZ", 0.39793742484318045),
        ("ZI", -0.39793742484318045),
        ("ZZ", -0.01128010425623538),
        ("XX", 0.18093119978423156),
    ]
)

qubit_op = H2_op
# # Load the `qubit_op` from the file
# with open('qubit_op_co2_4qubit.pkl', 'rb') as file:
#     qubit_op = pickle.load(file)


print(f"Loaded qubit_op: {qubit_op}")
# sss
print(f"Number of qubits: {qubit_op.num_qubits}")
numpy_solver = NumPyMinimumEigensolver()
result = numpy_solver.compute_minimum_eigenvalue(operator=qubit_op)
ref_value = result.eigenvalue.real
print(f"Reference value: {ref_value:.5f}")


def create_noise_model(q1_error, q2_error, spam_error, q1_length, q2_length, t1, t2):
    noise_model = NoiseModel()

    # Gate error rates
    one_qubit_gate_error = q1_error
    two_qubit_gate_error = q2_error
    spam_error = spam_error

    # Gate times (in nanoseconds, converted from microseconds)
    one_qubit_gate_time = q1_length
    two_qubit_gate_time = q2_length

    # Relaxation times (in microseconds, converted from seconds)
    T1 = t1
    T2 = t2

    # 1-qubit gate depolarizing error
    one_qubit_depolarizing_error = depolarizing_error(one_qubit_gate_error, 1)

    # 2-qubit gate depolarizing error
    two_qubit_depolarizing_error = depolarizing_error(two_qubit_gate_error, 2)

    # Thermal relaxation errors
    one_qubit_thermal_error = thermal_relaxation_error(T1, T2, one_qubit_gate_time)
    two_qubit_thermal_error = thermal_relaxation_error(T1, T2, two_qubit_gate_time)

    # Combine depolarizing and thermal errors for more realism
    combined_one_qubit_error = one_qubit_depolarizing_error.compose(one_qubit_thermal_error)
    combined_two_qubit_error = two_qubit_depolarizing_error.compose(two_qubit_thermal_error)

    # Add errors to the noise model
    noise_model.add_all_qubit_quantum_error(combined_one_qubit_error,
                                            ['x', 'y', 'z', 'h', 'rx', 'ry', 'rz', 'sx', 'id'])
    noise_model.add_all_qubit_quantum_error(combined_two_qubit_error, ['cx', 'ecr', 'cz'])

    # SPAM error (for readout errors)
    spam_probabilities = [[1 - spam_error, spam_error], [spam_error, 1 - spam_error]]

    noise_model.add_all_qubit_readout_error(spam_probabilities)
    return noise_model


noise_model_B1 = create_noise_model(3.382e-4, 3.12e-2, 2.350e-2, 45.00, 660, 114.97, 94.38)  # ibm_nazca
noise_model_B2 = create_noise_model(3.264e-4, 1.28e-2, 2.110e-2, 18, 68, 171.1, 139.86)  # ibm_torino
noise_model_B3 = create_noise_model(0.0003, 0.00212, 0.0051, 135e3, 600e3, 100.0 * 1e6, 1.0 * 1e6)  # ionq

# Instantiate samplers for each noise model
estimator_1 = AerEstimator(
    backend_options={"noise_model": noise_model_B1},
    run_options={"seed": algorithm_globals.random_seed, "shots": 1000},
    transpile_options={"seed_transpiler": algorithm_globals.random_seed}
)

estimator_2 = AerEstimator(
    backend_options={"noise_model": noise_model_B2},
    run_options={"seed": algorithm_globals.random_seed, "shots": 1000},
    transpile_options={"seed_transpiler": algorithm_globals.random_seed}
)
estimator_3 = AerEstimator(
    backend_options={"noise_model": noise_model_B3},
    run_options={"seed": algorithm_globals.random_seed, "shots": 1000},
    transpile_options={"seed_transpiler": algorithm_globals.random_seed}
)
# Define the feature map, ansatz, and optimizer
ansatz = EfficientSU2(num_qubits=qubit_op.num_qubits, reps=3)
ansatz.decompose().draw(output="mpl", style="clifford", fold=20)
plt.show()

# Initialize objective function values for both VQCs
objective_func_vals_1 = []
objective_func_vals_2 = []
objective_func_vals_3 = []
saved_params_1 = []
saved_params_2 = []
saved_params_3 = []

step_size_1 = []
step_size_2 = []
step_size_3 = []

optimizer1 = SPSA(maxiter=20)
optimizer2 = SPSA(maxiter=20)
optimizer3 = SPSA(maxiter=20)

counts_1 = []
counts_2 = []
counts_3 = []


def callback_graph1(eval_count, parameters, mean, std):
    counts_1.append(eval_count)
    objective_func_vals_1.append(mean)
    saved_params_1.append(parameters)


def callback_graph2(eval_count, parameters, mean, std):
    counts_2.append(eval_count)
    objective_func_vals_2.append(mean)
    saved_params_2.append(parameters)


def callback_graph3(eval_count, parameters, mean, std):
    counts_3.append(eval_count)
    objective_func_vals_3.append(mean)
    saved_params_3.append(parameters)


counts_optimizer_1 = []
counts_optimizer_2 = []
counts_optimizer_3 = []

accepted_function_values_1 = []
accepted_function_values_2 = []
accepted_function_values_3 = []


# Define the callback function
def spsa_callback_1(num_evaluations: int, params: np.ndarray, function_value: float, stepsize: float, accepted: bool):
    # Store the function value only if the step was accepted
    if accepted:
        accepted_function_values_1.append(function_value)
        counts_optimizer_1.append(num_evaluations)


# Define the callback function
def spsa_callback_2(num_evaluations: int, params: np.ndarray, function_value: float, stepsize: float, accepted: bool):
    # Store the function value only if the step was accepted
    if accepted:
        accepted_function_values_2.append(function_value)
        counts_optimizer_2.append(num_evaluations)


def spsa_callback_3(num_evaluations: int, params: np.ndarray, function_value: float, stepsize: float, accepted: bool):
    # Store the function value only if the step was accepted
    if accepted:
        accepted_function_values_3.append(function_value)
        counts_optimizer_3.append(num_evaluations)


# Define two VQC models with the same feature map, ansatz, and optimizer setup
vqe_1 = VQE(
    estimator=estimator_1,
    ansatz=ansatz,
    optimizer=optimizer1,
    callback=callback_graph1,

)

vqe_2 = VQE(
    estimator=estimator_2,
    ansatz=ansatz,
    optimizer=optimizer2,
    callback=callback_graph2,

)

vqe_3 = VQE(
    estimator=estimator_3,
    ansatz=ansatz,
    optimizer=optimizer3,
    callback=callback_graph3,

)

learning_rate_array = np.linspace(0.5, 0.01, 150)
perturbation_array = np.linspace(0.1, 0.05, 150)


# Custom function to handle staged training for both VQCs
def custom_training_vqes(schedules):
    start_time = time.time()
    # Track the current iteration index separately for each VQE
    vqe_1_current_iteration = 0
    vqe_2_current_iteration = 0
    vqe_3_current_iteration = 0
    for i, (vqe, estimator, iterations, stage) in enumerate(schedules):
        print(f"Stage {stage}: Training VQE with {iterations} iterations using {estimator}")

        if vqe == vqe_1:
            learning_rate_subarray = learning_rate_array[vqe_1_current_iteration:vqe_1_current_iteration + iterations]
            perturbation_subarray = perturbation_array[vqe_1_current_iteration:vqe_1_current_iteration + iterations]
            vqe_1_current_iteration += iterations  # Update current iteration for VQE1
            callbackfunction = spsa_callback_1
        elif vqe == vqe_2:
            learning_rate_subarray = learning_rate_array[vqe_2_current_iteration:vqe_2_current_iteration + iterations]
            perturbation_subarray = perturbation_array[vqe_2_current_iteration:vqe_2_current_iteration + iterations]
            vqe_2_current_iteration += iterations  # Update current iteration for VQE2
            callbackfunction = spsa_callback_2
        elif vqe == vqe_3:
            learning_rate_subarray = learning_rate_array[vqe_3_current_iteration:vqe_3_current_iteration + iterations]
            perturbation_subarray = perturbation_array[vqe_3_current_iteration:vqe_3_current_iteration + iterations]
            vqe_3_current_iteration += iterations  # Update current iteration for VQE3
            callbackfunction = spsa_callback_3

            # Create the SPSA optimizer with the selected subarrays
        new_optimizer = SPSA(
            maxiter=iterations,
            learning_rate=learning_rate_subarray,
            perturbation=perturbation_subarray,
            callback=callbackfunction
        )

        vqe.estimator = estimator
        vqe.optimizer = new_optimizer

        if stage == 'stage1':
            print('first stages')
            np.random.seed(42)
            algorithm_globals.random_seed = 42
            vqe.initial_point = np.asarray([0.5] * ansatz.num_parameters)
        if stage != 'stage1':
            print('not first stages')
            if vqe == vqe_1:
                last_saved_params = saved_params_1[-1]  # Get the last saved parameters
                vqe.initial_point = last_saved_params
            if vqe == vqe_2:
                last_saved_params = saved_params_2[-1]  # Get the last saved parameters
                vqe.initial_point = last_saved_params
            if vqe == vqe_3:
                last_saved_params = saved_params_3[-1]  # Get the last saved parameters
                vqe.initial_point = last_saved_params

        try:
            print("before")
        except:
            print('no weight yet')
        if vqe == vqe_1:
            result_1 = vqe.compute_minimum_eigenvalue(operator=qubit_op)
        if vqe == vqe_2:
            result_2 = vqe.compute_minimum_eigenvalue(operator=qubit_op)
        if vqe == vqe_3:
            result_3 = vqe.compute_minimum_eigenvalue(operator=qubit_op)

        print("after")

    elapsed = time.time() - start_time
    print(f"Total training time: {round(elapsed)} seconds")

    print(f"Backend 1 Energy: {result_1.eigenvalue.real:.5f}")
    print(f"Deviation from reference energy value: {(result_1.eigenvalue.real - ref_value):.5f}")
    print(
        f"Normalized Score (between 0 and 1): {1 / (1 + abs(result_1.eigenvalue.real - ref_value) / abs(ref_value)):.6f}")

    print(f"Backend 2 Energy: {result_2.eigenvalue.real:.5f}")
    print(f"Deviation from reference energy value: {(result_2.eigenvalue.real - ref_value):.5f}")
    print(
        f"Normalized Score (between 0 and 1): {1 / (1 + abs(result_2.eigenvalue.real - ref_value) / abs(ref_value)):.6f}")

    print(f"Backend 3 Energy: {result_3.eigenvalue.real:.5f}")
    print(f"Deviation from reference energy value: {(result_3.eigenvalue.real - ref_value):.5f}")

    print(
        f"Normalized Score (between 0 and 1): {1 / (1 + abs(result_3.eigenvalue.real - ref_value) / abs(ref_value)):.6f}")


# Example schedule with easy control

numberofstage2 = int(0.2 * 150)
numberofstage1 = 150 - numberofstage2
schedule_example = [
    (vqe_1, estimator_1, 150, 'stage1'),  # run all iterations on B1

    (vqe_2, estimator_1, numberofstage1, 'stage1'),  # Stage 1 on B1
    (vqe_2, estimator_3, numberofstage2, 'stage2'),  # Stage 2 on B2

    (vqe_3, estimator_3, 150, 'stage1'),  # run all iterations on B3

]


def plot_loss_curve_stages(objective_func_vals_1, objective_func_vals_2, objective_func_vals_3, stages):
    plt.figure(figsize=(10, 8))

    start_1 = 0
    start_2 = 0
    start_3 = 0

    for i, (vqe, sampler, iterations, _) in enumerate(stages):
        # Determine the color based on the VQE
        if vqe == vqe_1:
            color = 'r'  # Red for VQE1
        elif vqe == vqe_2:
            color = 'b'  # Blue for VQE2
        else:
            color = 'black'

        # Determine the line style
        if sampler == estimator_1:
            linestyle = '-'  # Solid line for Backend 1
        elif sampler == estimator_2:
            linestyle = '-'  # Dashed line for Backend 2
        else:
            linestyle = '-'
        if vqe == vqe_1:
            end = start_1 + iterations
            if start_1 == 0:
                plt.plot(range(start_1, end), objective_func_vals_1[start_1:end], color=color, linestyle=linestyle)
            else:
                plt.plot(range(start_1 - 1, end),
                         [objective_func_vals_1[start_1 - 1]] + objective_func_vals_1[start_1:end],
                         color=color, linestyle=linestyle)
            start_1 = end

        elif vqe == vqe_2:
            end = start_2 + iterations
            if start_2 == 0:
                plt.plot(range(start_2, end), objective_func_vals_2[start_2:end], color=color, linestyle=linestyle)
            else:
                plt.plot(range(start_2 - 1, end),
                         [objective_func_vals_2[start_2 - 1]] + objective_func_vals_2[start_2:end],
                         color=color, linestyle=linestyle)
            start_2 = end
        else:
            end = start_3 + iterations
            if start_3 == 0:
                plt.plot(range(start_3, end), objective_func_vals_3[start_3:end], color=color, linestyle=linestyle)
            else:
                plt.plot(range(start_3 - 1, end),
                         [objective_func_vals_3[start_3 - 1]] + objective_func_vals_3[start_3:end],
                         color=color, linestyle=linestyle)
            start_3 = end

        plt.axvline(x=end - 1, color='k', linestyle=':')
        plt.axhline(y=ref_value, color='orange', linestyle='-')
    # Simplify the legend
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='r', lw=2, linestyle='-'),
        Line2D([0], [0], color='b', lw=2, linestyle='-'),
        Line2D([0], [0], color='black', lw=2, linestyle='-'),
        Line2D([0], [0], color='orange', lw=2, linestyle='-')
    ]
    plt.legend(custom_lines, ['Backend 1', 'Backend Split 1-3', 'Backend 3', 'Ref_Val - Classical'])

    plt.xlabel("Iterations")
    plt.ylabel("VQE Energy")
    plt.title("Training Process for VQE on different backends")
    plt.show()


# Example usage
# Perform the custom training and save model
custom_training_vqes(schedule_example)
# Plot
plot_loss_curve_stages(accepted_function_values_1, accepted_function_values_2, accepted_function_values_3,
                       schedule_example)
