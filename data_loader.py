import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt
from tqdm import tqdm
import torch
from connections import GMatrixCalculator
from torch.utils.data import Dataset

class EEGDataLoader(Dataset):
    def __init__(self, root_path: str, class_type, trial_type):
        """
        Initializes the EEGDataLoader class.
        :param root_path: Path to the root directory containing all session folders.
        """
        self.root_path = Path(root_path)
        self.sessions = [d for d in sorted(self.root_path.iterdir()) if d.is_dir()]

        # Electrode selection for different conditions
        self.selected_electrodes = {
            'AK-SREP': ["F8", "FCz", "Fp1", "AF7", "AF3", "C1", "FC4", "F1", "Pz", "F2", "P5", "P6"],
            'CTK-SREP': ["F1", "P5", "F4", "AF7", "PO5", "FC1", "FCz", "Fp2", "Fz", "PO3", "TP8", "F4"],

            'AK-SRES': ["AF8", "FC6", "F8", "T8", "C6", "AF7", "Fp2", "CP6", "Fpz", "F4", "TP8", "P6"],
            'CTK-SRES': ["F1", "P4", "F4", "AF3", "CP6", "P5", "Fz", "FC2", "FC1", "P6", "PC6", "F2"]
        }
        self.data, self.labels = self.get_trials(class_type = class_type, trial_type = trial_type)


    def load_eeg_data(self, csv_file: str):
        """
        Loads EEG data from a CSV file.
        :param csv_file: Path to the CSV file.
        :return: Pandas DataFrame of EEG data.
        """
        df = pd.read_csv(csv_file)
        return df

    def _downsample(self, data, factor=2):
        """Downsamples the EEG signal along the last axis."""
        return data[:, :, ::factor]

    def _butterworth_lowpass_filter(self, data, fs=1000, cutoff=50, order=7):
        """
        Applies a Butterworth low-pass filter to EEG data.

        :param data: np.ndarray (trials, channels, time_samples)
        :param fs: Sampling frequency of EEG data (Hz)
        :param cutoff: Cutoff frequency (default = 50 Hz)
        :param order: Order of the filter (default = 7)
        :return: Filtered EEG data
        """
        nyquist = fs / 2
        normalized_cutoff = cutoff / nyquist
        b, a = butter(order, normalized_cutoff, btype='low', analog=False)

        # Apply zero-phase filtering
        filtered_data = filtfilt(b, a, data, axis=-1)

        return filtered_data

    def _min_max_normalization_per_signal(self, data, feature_range=(0, 1)):
        """
        Normalizes EEG data using Min-Max scaling per trial.

        :param data: np.ndarray of shape (trials, channels, time_samples)
        :param feature_range: Tuple defining the min and max range (default (0,1))
        :return: Normalized EEG data
        """
        min_val, max_val = feature_range
        data_min = np.min(data, axis=-1, keepdims=True)
        data_max = np.max(data, axis=-1, keepdims=True)
        data_range = data_max - data_min

        # Avoid division by zero
        data_range[data_range == 0] = 1

        # Normalize each signal independently
        normalized_data = (data - data_min) / data_range
        normalized_data = normalized_data * (max_val - min_val) + min_val

        return normalized_data

    def extract_trials(self, df, class_type="AK-SREP", trial_type="reading", target_samples=4000):
        """
        Extracts EEG trials and labels from a DataFrame.

        :param df: Pandas DataFrame with EEG data
        :param class_type: 'AK-SREP' or 'CTK-SREP' for electrode selection
        :param trial_type: 'reading' (21) or 'speaking' (30) to define trial start events
        :param target_samples: Number of samples per trial
        :return: (np.ndarray of trials, np.ndarray of labels)
        """
        event_column = df.iloc[:, -1]  # Assuming event markers are in the last column
        selected_electrodes = self.selected_electrodes[class_type]

        trials = []
        labels = []

        # Choose the correct event code based on trial type
        start_event = 21 if trial_type == "reading" else 30

        # Get event indices and values
        event_indices = event_column[event_column.notna()].index.tolist()
        event_values = event_column.dropna().values

        for i in range(len(event_values)):
            if event_values[i] == start_event:  # Start of a trial (21 for reading, 30 for speaking)
                if i == 0:
                    continue  # No previous event to use as label
                if trial_type == "reading":
                    label = event_values[i - 1]  # The event before start_event is the label
                else:
                    label = event_values[i - 3]
                    
                start_idx = event_indices[i]  # Trial start index

                # Ensure there is a next event to define the end
                if i + 1 < len(event_indices):
                    end_idx = event_indices[i + 1]
                else:
                    continue

                # Extract EEG data for the selected electrodes
                trial_data = df.loc[start_idx:end_idx, selected_electrodes].values.T  # Transpose to (channels, time_samples)

                # Ensure consistent shape
                current_samples = trial_data.shape[1]

                if current_samples < target_samples:
                    # Pad with zeros
                    padding = np.zeros((trial_data.shape[0], target_samples - current_samples))
                    trial_data = np.hstack((trial_data, padding))
                elif current_samples > target_samples:
                    # Trim excess samples
                    trial_data = trial_data[:, :target_samples]


                trials.append(trial_data)
                labels.append(label)
        
        return np.array(trials), np.array(labels)

    def get_trials(self, class_type="AK-SREP", trial_type="reading"):
        """
        Extracts and processes all trials from all EEG files.

        :param class_type: 'AK-SREP' or 'CTK-SREP'
        :param trial_type: 'reading' (21) or 'speaking' (30)
        :return: (np.ndarray EEG data, np.ndarray labels)
        """
        all_trials = []
        all_labels = []

        for session in tqdm(self.sessions, desc="Processing Sessions"):
            eeg_files = list(session.glob("*.csv"))
            for eeg_file in eeg_files:
                # print(eeg_file)
                df = self.load_eeg_data(str(eeg_file))
                trials, labels = self.extract_trials(df, class_type=class_type, trial_type=trial_type)
                all_trials.extend(trials)
                all_labels.extend(labels)

        # Convert lists to NumPy arrays
        all_trials = np.array(all_trials)
        all_labels = np.array(all_labels)

        # Apply filtering, downsampling, and normalization
        filtered_data = self._butterworth_lowpass_filter(all_trials, cutoff=50)
        downsampled_data = self._downsample(filtered_data)
        normalized_data = self._min_max_normalization_per_signal(downsampled_data)

        return normalized_data, all_labels

    def __getitem__(self, idx):

        adjacency_matrix =  torch.tensor(GMatrixCalculator._compute_G_matrix(self.data[idx]), dtype=torch.float32)  
        feature_matrix = torch.tensor(self.data[idx], dtype=torch.float32)  
        label = torch.tensor(self.labels[idx], dtype=torch.long) 

        return feature_matrix, adjacency_matrix, label - 1
    
    def __len__(self):
        reading_data, reading_labels = self.get_trials(class_type="AK-SREP", trial_type="reading")
        return reading_data.shape[0]
    


if __name__ == '__main__':
    dataset = EEGDataLoader("Data")

    # # Extract reading trials
    # reading_data, reading_labels = dataset.get_trials(class_type="AK-SREP", trial_type="reading")
    # print(f"Reading Trials - EEG Data Shape: {reading_data.shape}, Labels Shape: {reading_labels.shape}")

    # Extract speaking trials
    # speaking_data, speaking_labels = loader.get_trials(class_type="AK-SREP", trial_type="speaking")
    # print(f"Speaking Trials - EEG Data Shape: {speaking_data.shape}, Labels Shape: {speaking_labels.shape}")