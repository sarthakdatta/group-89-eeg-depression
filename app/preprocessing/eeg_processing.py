"""
EEG Data Preprocessing Module
Extracted from the model development notebook for use in FastAPI backend.
"""

import numpy as np
from scipy import signal
from scipy.io import loadmat
from typing import Tuple

# Constants
SAMPLING_RATE = 250
LOW_CUTOFF = 1.0
HIGH_CUTOFF = 45.0
NOTCH = 50.0
TRIM_SEC = 30

WIN_SEC = 2.0
STEP_SEC = WIN_SEC / 2

FREQ_BANDS = [
    (1, 4),    # Delta (δ)
    (4, 8),    # Theta (θ)
    (8, 13),   # Alpha (α)
    (13, 30),  # Beta (β)
    (30, 45)   # Gamma (γ)
]


def butter_bandpass_filter(data: np.ndarray, sampling_rate: float, 
                          low_cutoff: float, high_cutoff: float, 
                          filter_order: int = 4) -> np.ndarray:
    """Apply Butterworth bandpass filter to EEG data."""
    nyquist_freq = 0.5 * sampling_rate
    low_norm = low_cutoff / nyquist_freq
    high_norm = high_cutoff / nyquist_freq
    b_coeffs, a_coeffs = signal.butter(filter_order, [low_norm, high_norm], btype='band')
    filtered_data = signal.filtfilt(b_coeffs, a_coeffs, data, axis=-1)
    return filtered_data


def notch_filter(data: np.ndarray, sampling_rate: float, 
                 notch: float, quality_factor: float = 30.0) -> np.ndarray:
    """Apply notch filter to remove line noise (e.g., 50Hz)."""
    nyquist_freq = 0.5 * sampling_rate
    notch_norm = notch / nyquist_freq
    b_coeffs, a_coeffs = signal.iirnotch(w0=notch_norm, Q=quality_factor)
    filtered_data = signal.filtfilt(b_coeffs, a_coeffs, data, axis=-1)
    return filtered_data


def preprocess_data(data: np.ndarray, sampling_rate: float = SAMPLING_RATE,
                   low_cutoff: float = LOW_CUTOFF, high_cutoff: float = HIGH_CUTOFF,
                   notch: float = NOTCH, trim_sec: float = TRIM_SEC) -> np.ndarray:
    """
    Preprocess EEG data: detrend, bandpass filter, notch filter, and trim edges.
    
    Args:
        data: Raw EEG data (channels x time_samples)
        sampling_rate: Sampling rate in Hz
        low_cutoff: Low frequency cutoff for bandpass filter
        high_cutoff: High frequency cutoff for bandpass filter
        notch: Notch frequency (typically 50Hz or 60Hz)
        trim_sec: Seconds to trim from beginning and end
        
    Returns:
        Preprocessed EEG data
    """
    # Detrend (remove DC offset)
    processed_data = data - data.mean(axis=0, keepdims=True)
    
    # Bandpass filter
    processed_data = butter_bandpass_filter(processed_data, sampling_rate, low_cutoff, high_cutoff)
    
    # Notch filter
    processed_data = notch_filter(processed_data, sampling_rate, notch)
    
    # Trim edges
    start_sample = int(trim_sec * sampling_rate)
    end_sample = processed_data.shape[1] - int(trim_sec * sampling_rate)
    end_sample = max(end_sample, start_sample + 1)
    
    return processed_data[:, start_sample:end_sample]


def slide_wins(preprocessed_data: np.ndarray, sampling_rate: float = SAMPLING_RATE,
               win_sec: float = WIN_SEC, step_sec: float = STEP_SEC) -> Tuple[np.ndarray, list]:
    """
    Extract sliding windows from preprocessed data.
    
    Args:
        preprocessed_data: Preprocessed EEG data (channels x time_samples)
        sampling_rate: Sampling rate in Hz
        win_sec: Window size in seconds
        step_sec: Step size in seconds
        
    Returns:
        Tuple of (windows_array, window_indices)
    """
    win_cnts = int(win_sec * sampling_rate)
    step_cnts = int(step_sec * sampling_rate)

    wins = []
    win_idxs = []
    for start_idx in range(0, preprocessed_data.shape[1] - win_cnts + 1, step_cnts):
        end_idx = start_idx + win_cnts
        wins.append(preprocessed_data[:, start_idx:end_idx])
        win_idxs.append((start_idx, end_idx))

    if wins:
        wins_array = np.stack(wins, axis=0)
    else:
        wins_array = np.empty((0, preprocessed_data.shape[0], win_cnts))

    return wins_array, win_idxs


def discard_noisy_wins(wins_array: np.ndarray, z_score_threshold: float = 7.0) -> np.ndarray:
    """
    Remove noisy windows based on z-score threshold.
    
    Args:
        wins_array: Array of windows (n_windows x channels x samples)
        z_score_threshold: Maximum z-score threshold for keeping windows
        
    Returns:
        Filtered windows array with noisy windows removed
    """
    if len(wins_array) == 0:
        return wins_array
    
    mean = wins_array.mean(axis=(1, 2), keepdims=True)
    std = wins_array.std(axis=(1, 2), keepdims=True) + 1e-6
    z_scores = (wins_array - mean) / std
    is_not_noisy_win_tf = np.max(np.abs(z_scores), axis=(1, 2)) < z_score_threshold
    
    return wins_array[is_not_noisy_win_tf]


def welch(wins_array: np.ndarray, sampling_rate: float = SAMPLING_RATE,
          freq_bands: list = FREQ_BANDS) -> np.ndarray:
    """
    Extract features using Welch's method for power spectral density.
    
    Args:
        wins_array: Array of windows (n_windows x channels x samples)
        sampling_rate: Sampling rate in Hz
        freq_bands: List of frequency band tuples (low, high)
        
    Returns:
        Feature array (n_windows x n_features) where n_features = channels * n_bands
    """
    if len(wins_array) == 0:
        return np.empty((0, 0), dtype=np.float32)

    win_cnts, ch_cnts, sample_cnts = wins_array.shape
    samples_per_seg = 256
    features = []

    for idx in range(win_cnts):
        curr_win = wins_array[idx]
        freqs, psds = signal.welch(curr_win, fs=sampling_rate, nperseg=samples_per_seg, axis=-1)
        integrated_psd_in_all_freqs = np.trapz(psds, freqs, axis=-1) + 1e-12

        curr_win_bands = []
        for (low_freq, high_freq) in freq_bands:
            is_in_this_freq_band_tf = (freqs >= low_freq) & (freqs < high_freq)
            integrated_psd_in_this_freq_band = np.np.trapz(
                psds[:, is_in_this_freq_band_tf], 
                freqs[is_in_this_freq_band_tf], 
                axis=-1
            )
            relative_psd_in_this_freq_band_in_perc = (
                integrated_psd_in_this_freq_band / integrated_psd_in_all_freqs
            )
            curr_win_bands.append(relative_psd_in_this_freq_band_in_perc)

        win_features = np.stack(curr_win_bands, axis=-1)
        features.append(win_features)

    features_array = np.stack(features, axis=0)
    features_array = features_array.reshape(win_cnts, -1).astype(np.float32)

    return features_array


def load_mat(path: str, verbose: bool = True) -> np.ndarray:
    """
    Load EEG data from .mat file.
    
    Args:
        path: Path to .mat file
        verbose: Whether to print loading information
        
    Returns:
        EEG data array (channels x time_samples)
    """
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    keys_wo__ = []
    for keyname in mat.keys():
        if not keyname.startswith("__"):
            keys_wo__.append(keyname)
    
    target_key = keys_wo__[0]
    target_data = np.asarray(mat[target_key], dtype=np.float32)

    # Handle 129-channel data (remove reference channel)
    if target_data.shape[0] == 129:
        target_data = target_data[:128, :]

    return target_data


def process_eeg_file_to_features(mat_path: str) -> np.ndarray:
    """
    Complete pipeline: Load .mat file and extract features.
    
    Args:
        mat_path: Path to .mat file
        
    Returns:
        Mean feature vector (1D array of size n_features)
    """
    raw_data = load_mat(mat_path, verbose=False)
    preprocessed_data = preprocess_data(raw_data, SAMPLING_RATE)
    wins_array, _ = slide_wins(preprocessed_data, SAMPLING_RATE)
    clean_wins = discard_noisy_wins(wins_array)
    features_array = welch(wins_array, SAMPLING_RATE)
    
    # Return mean features across all windows (same as training)
    return features_array.mean(axis=0)


def process_eeg_array_to_features(eeg_data: np.ndarray, sampling_rate: float = SAMPLING_RATE) -> np.ndarray:
    """
    Complete pipeline: Process raw EEG array and extract features.
    
    Args:
        eeg_data: Raw EEG data array (channels x time_samples)
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Mean feature vector (1D array of size n_features)
    """
    preprocessed_data = preprocess_data(eeg_data, sampling_rate)
    wins_array, _ = slide_wins(preprocessed_data, sampling_rate)
    clean_wins = discard_noisy_wins(wins_array)
    features_array = welch(wins_array, sampling_rate)
    
    # Return mean features across all windows (same as training)
    return features_array.mean(axis=0)
