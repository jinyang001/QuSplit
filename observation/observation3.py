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
import pickle
import csv

np.random.seed(42)
algorithm_globals.random_seed = 42


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

# Initialize global parameters
learning_rate_array = np.linspace(0.5, 0.01, 150)  # Learning rate decay
perturbation_array = np.linspace(0.1, 0.05, 150)  # Perturbation for SPSA

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

# with open('qubit_op_co2_4qubit.pkl', 'rb') as file:
#     qubit_op = pickle.load(file)

print(f"Loaded qubit_op: {qubit_op}")
# sss
print(f"Number of qubits: {qubit_op.num_qubits}")
numpy_solver = NumPyMinimumEigensolver()
result = numpy_solver.compute_minimum_eigenvalue(operator=qubit_op)
ref_value = result.eigenvalue.real
print(f"Reference value: {ref_value:.5f}")


# Define callback functions to capture optimizer behavior
def spsa_callback_1(num_evaluations: int, params: np.ndarray, function_value: float, stepsize: float, accepted: bool):
    if accepted:
        print(f"Stage 1 Callback: Eval {num_evaluations}, function value: {function_value}")


def spsa_callback_2(num_evaluations: int, params: np.ndarray, function_value: float, stepsize: float, accepted: bool):
    if accepted:
        print(f"Stage 2 Callback: Eval {num_evaluations}, function value: {function_value}")


# Custom training function for VQE
def custom_training_vqe_dynamic(splits):
    results = []

    for split in splits:
        # Set a fixed random seed before each split
        np.random.seed(42)
        algorithm_globals.random_seed = 42

        start_time = time.time()

        # Initialize number of iterations for Stage 1 and Stage 2
        iterations_stage1 = split
        iterations_stage2 = 150 - split

        # Initialize the estimator and VQE for each split
        estimator_1 = AerEstimator(
            backend_options={"noise_model": noise_model_B3},
            run_options={"seed": algorithm_globals.random_seed, "shots": 1000},
            transpile_options={"seed_transpiler": algorithm_globals.random_seed}
        )

        estimator_2 = AerEstimator(
            backend_options={"noise_model": noise_model_B2},
            run_options={"seed": algorithm_globals.random_seed, "shots": 1000},
            transpile_options={"seed_transpiler": algorithm_globals.random_seed}
        )
        estimator_3 = AerEstimator(
            backend_options={"noise_model": noise_model_B1},
            run_options={"seed": algorithm_globals.random_seed, "shots": 1000},
            transpile_options={"seed_transpiler": algorithm_globals.random_seed}
        )
        # Ansatz (circuit template)
        ansatz = EfficientSU2(num_qubits=qubit_op.num_qubits, reps=3)

        # Current index to track iterations across stages
        current_iteration = 0

        # Split logic (stage 1 and stage 2)
        schedule_example = []
        if iterations_stage1 > 0:
            schedule_example.append((estimator_3, iterations_stage1, 'stage1'))
        if iterations_stage2 > 0:
            schedule_example.append((estimator_1, iterations_stage2, 'stage2'))

        print(f"Split: Stage 1 = {iterations_stage1} iterations, Stage 2 = {iterations_stage2} iterations")

        result = None  # To hold the final result of the split
        saved_params_2 = []

        def callback_graph(eval_count, parameters, mean, std):
            saved_params_2.append(parameters)

        # Train on each stage
        for i, (estimator, iterations, stage) in enumerate(schedule_example):
            print(f"Stage {i + 1}: Training VQE with {iterations} iterations using {estimator}")

            # Initialize the learning rate and perturbation arrays with current index offset
            learning_rate_subarray = learning_rate_array[current_iteration:current_iteration + iterations]
            perturbation_subarray = perturbation_array[current_iteration:current_iteration + iterations]

            # Update the current iteration index for the next stage
            current_iteration += iterations

            # Set the callback and initial points based on the stage
            if stage == 'stage1':
                print('First stage')

                initial_point = np.asarray([0.5] * ansatz.num_parameters)
            if stage != 'stage1':
                print('not first stages')
                last_saved_params = saved_params_2[-1]  # Get the last saved parameters
                initial_point = last_saved_params

            # Create new SPSA optimizer for each stage
            optimizer = SPSA(
                maxiter=iterations,
                learning_rate=learning_rate_subarray,
                perturbation=perturbation_subarray,
            )
            vqe = VQE(
                estimator=estimator,
                ansatz=ansatz,
                callback=callback_graph,
                optimizer=optimizer,
                initial_point=initial_point
            )

            # Compute the minimum eigenvalue and store the result
            result = vqe.compute_minimum_eigenvalue(operator=qubit_op)

        # After both stages are done, log the final result for this split
        results.append((iterations_stage1, iterations_stage2, result.eigenvalue.real))

        elapsed = time.time() - start_time
        print(f"Total training time for split {split}: {round(elapsed)} seconds")
        print(f"VQE Energy: {result.eigenvalue.real:.5f}")

    return results


# Generate splits
splits = list(range(50, 151, 5))

# Reading the CSV file and extracting data
csv_filename = "vqe_performance_splits_h2_1_3.csv"
stage2_iterations = []
vqe2_energies = []

with open(csv_filename, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        stage2_iterations.append(int(row["Iterations Stage 2"]))
        vqe2_energies.append(float(row["VQE Energy"]))

# Plotting the results

plt.plot(stage2_iterations, vqe2_energies, marker='o', markersize=4, linestyle='--', color='b', label="Backend 2")

# Adding the two horizontal reference lines
plt.axhline(y=ref_value, color='orange', linestyle='-', label='Ref_Val-Classical')
plt.axhline(y=-1.80385, color='red', linestyle='-', label='Backend 1')

# Customizing the plot
plt.xlabel("Iterations # on Backend 1")
plt.ylabel("VQE Energy")
plt.title("VQE Energy vs Iterations for Stage 2(trained on less noise backend)")
plt.grid(True)
plt.ylim(-1.9, -1.3)
plt.yticks(np.arange(-1.9, -1.2, 0.1))
plt.xlim(0, 100)
plt.xticks(np.arange(0, 100, 20))
plt.yticks(visible=False)
plt.xticks(visible=False)
plt.ylim(-1.2, -0.7)
plt.yticks(np.arange(-1.2, -0.65, 0.05))
# Displaying the legend
plt.legend()
# plt.savefig('./plots/observation3_h2_1_3.eps', format='eps', dpi=1000)
# Show the plot
plt.show()
