from  load_data import EEGDataLoader
from connections import GMatrixCalculator
from visualization import Visualization

if __name__ == '__main__':
    eeg_loader = EEGDataLoader('Data')
    all_trials = eeg_loader.get_all_trials('SREP')
    listening_data = all_trials['listening']
    # print(listening_data.shape)

    # Compute G matrices
    g_calculator = GMatrixCalculator()
    listening_G_matrices = g_calculator.compute_G_matrices(listening_data)
    print(listening_G_matrices[0])  # (samples, 12, 12)

    # # Visualize trial and G matrix
    # visualization = Visualization()
    # visualization.plot_trial_and_G_matrix(listening_data[0], listening_G_matrices[0], 1)

    