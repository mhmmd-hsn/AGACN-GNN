import mne
import json
import numpy as np
from pathlib import Path

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
        return data[:, :, ::factor]

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
                downsampled_data = self._downsample(epochs.get_data())
                all_trials.append(downsampled_data)
        
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
