import mne
import json
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt

class EEGDataLoader:
    def __init__(self, root_path: str):
        """
        Initializes the EEGDataLoader class.
        :param root_path: Path to the root directory containing all session folders.
        """
        self.root_path = Path(root_path)
        self.sessions = [d for d in sorted(self.root_path.iterdir()) if d.is_dir()]
        # Electrode selection for SREP and SRES (Table 3 & Table 5)
        self.selected_electrodes = {
            'SREP': ["F3", "F5", "FC3", "FC5", "C3", "C5", "T7", "CP3", "CP5", "P3", "P5", "PO3"],
            'SRES': ["F4", "F6", "FC4", "FC6", "C4", "C6", "T8", "CP4", "CP6", "P4", "P6", "PO4"]
        }
        # self.selected_electrodes = {
        #     'SREP': ["F3", "F5", "FC3", "FC5", "C3", "C5", "T7", "CP3", "CP5", "P3", "P5", "PO3"],
        #     'SRES': ["F4", "F6", "FC4", "FC6", "C4", "C6", "T8", "CP4", "CP6", "P4", "P6", "PO4"]
        # }
    
    def _load_metadata(self, session_path):
        """Load metadata from the JSON file."""
        info_path = session_path / "recordInformation.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                return json.load(f)
        return {}

    def _load_eeg_data(self, file_path):
        """Load EEG data from the .bdf file using MNE-Python."""
        return mne.io.read_raw_bdf(file_path, preload=True)
    
    def _load_event_markers(self, evt_path):
        """Extract event markers from the evt.bdf file."""
        raw_evt = mne.io.read_raw_bdf(evt_path, preload=True)
        events, _ = mne.events_from_annotations(raw_evt)
        return events

    def _downsample(self, data, factor=2):
        """Applies downsampling to the EEG signal using a sliding window approach."""
        return np.mean(data[:, :, :data.shape[2] // factor * factor].reshape(data.shape[0], data.shape[1], -1, factor), axis=-1)
    def _butterworth_lowpass_filter(self, data, fs=1000, cutoff=50, order=7):
        """
        Applies a 7th-order Butterworth low-pass filter with a 50Hz cutoff to EEG data.

        Parameters:
        - data: np.ndarray -> EEG signal of shape (samples, channels, time_samples)
        - fs: int -> Sampling frequency of the EEG data (Hz)
        - cutoff: int -> Cutoff frequency in Hz (default = 50 Hz, per paper)
        - order: int -> Filter order (default = 7, per paper)

        Returns:
        - np.ndarray -> Filtered EEG data with the same shape
        """
        nyquist = fs / 2  # Nyquist frequency
        normalized_cutoff = cutoff / nyquist  # Normalize the cutoff frequency
        
        # Design the Butterworth low-pass filter
        b, a = butter(order, normalized_cutoff, btype='low', analog=False)

        # Apply the filter using zero-phase filtering to prevent phase distortion
        filtered_data = filtfilt(b, a, data, axis=-1)

        return filtered_data
    def _min_max_normalization_per_signal(self,data, feature_range=(0, 1)):
        """
        Applies Min-Max Normalization to each EEG signal independently.

        Parameters:
        - data: np.ndarray -> EEG signal (shape: trials × channels × time_samples)
        - feature_range: tuple -> Desired range (default: (0,1))

        Returns:
        - np.ndarray -> Normalized EEG data with the same shape
        """
        min_val, max_val = feature_range  # Desired range (e.g., (0,1) or (-1,1))
        
        # Compute min and max for each signal separately (independent per signal)
        data_min = np.min(data, axis=-1, keepdims=True)  # Shape: (trials, 1, 1)
        data_max = np.max(data, axis=-1, keepdims=True)  # Shape: (trials, 1, 1)

        # Prevent division by zero
        data_range = data_max - data_min
        data_range[data_range == 0] = 1  # Avoids NaN if min == max

        # Apply Min-Max Normalization for each signal separately
        normalized_data = (data - data_min) / data_range  # Scale to (0,1)
        normalized_data = normalized_data * (max_val - min_val) + min_val  # Adjust range

        return normalized_data

    def get_trials(self, trial_type: str, dataset: str):
        """
        Extracts all trials of a given type from all sessions, selecting only the relevant electrodes.
        Applies downsampling after extracting trials.
        :param trial_type: 'listening', 'silent reading', or 'speaking'
        :param dataset: 'SREP' or 'SRES'
        :return: Numpy array of all trials concatenated together with selected electrodes.
        """
        trial_markers = {
            'listening': [1, 9],  # Adjust based on event marker definitions
            'silent reading': [21, 22],
            'speaking': [30, 31]
        }
        
        if trial_type not in trial_markers:
            raise ValueError("Invalid trial type. Choose from 'listening', 'silent reading', or 'speaking'")
        
        if dataset not in self.selected_electrodes:
            raise ValueError("Invalid dataset. Choose from 'SREP' or 'SRES'")
        
        all_trials = []
        
        for session in self.sessions:
            data_path = session / "data.bdf"
            evt_path = session / "evt.bdf"
            
            if not data_path.exists() or not evt_path.exists():
                continue
            
            raw_data = self._load_eeg_data(data_path)
            events = self._load_event_markers(evt_path)
            selected_events = [event for event in events if event[2] in trial_markers[trial_type]]
            
            if selected_events:
                epochs = mne.Epochs(
                    raw_data, selected_events, event_id=None, tmin=0, tmax=4, baseline=None, preload=True
                )
                epochs.pick_channels(self.selected_electrodes[dataset])
                filtered_data = self._butterworth_lowpass_filter(epochs.get_data(), cutoff=50)
                downsampled_data = self._downsample(filtered_data)
                normalized_data = self._min_max_normalization_per_signal(downsampled_data)
                all_trials.append(normalized_data)
        
        return np.concatenate(all_trials, axis=0) if all_trials else np.array([])
    
    def get_all_trials(self, dataset: str):
        """Extracts all trials for 'listening', 'silent reading', and 'speaking' from all sessions with selected electrodes."""
        return {
            'listening': self.get_trials('listening', dataset),
            'silent reading': self.get_trials('silent reading', dataset),
            'speaking': self.get_trials('speaking', dataset)
        }

# Example Usage:
# eeg_loader = EEGDataLoader('path/to/Data')
# all_trials = eeg_loader.get_all_trials('SREP')
# listening_data = all_trials['listening']