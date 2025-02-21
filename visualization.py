import matplotlib.pyplot as plt

class Visualization:
    def __init__(self):
        """Initializes the Visualization class."""
        pass
    
    def plot_trial_and_G_matrix(self, trial_data, G_matrix, trial_index):
        """Plots an EEG trial alongside its corresponding G matrix with enhanced clarity.
        :param trial_data: Numpy array of shape (12, time_points) representing EEG signals.
        :param G_matrix: Numpy array of shape (12, 12) representing the correlation matrix.
        :param trial_index: Index of the trial for labeling.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot EEG signals with better visualization
        for i in range(trial_data.shape[0]):
            axes[0].plot(trial_data[i, :], label=f'Channel {i+1}', linewidth=1.2, alpha=0.8)
        axes[0].set_title(f'EEG Trial {trial_index}', fontsize=14)
        axes[0].set_xlabel('Time Points', fontsize=12)
        axes[0].set_ylabel('Amplitude', fontsize=12)
        axes[0].legend(fontsize=10, loc='upper right')
        axes[0].grid(True, linestyle='--', alpha=0.5)
        
        # Plot G matrix with enhanced clarity
        im = axes[1].imshow(G_matrix, cmap='hot', interpolation='nearest')
        axes[1].set_title(f'G Matrix for Trial {trial_index}', fontsize=14)
        axes[1].set_xlabel('Channels', fontsize=12)
        axes[1].set_ylabel('Channels', fontsize=12)
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.show()