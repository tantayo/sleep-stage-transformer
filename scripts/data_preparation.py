import os
import glob
import numpy as np
import mne

import warnings
from tqdm import tqdm

# Stages 3 and 4 are considered a single class "Deep sleep"
LABEL_MAPPING = {
    'Sleep stage W': 0, 'Sleep stage 1': 1, 'Sleep stage 2': 2,
    'Sleep stage 3': 3, 'Sleep stage 4': 3, 'Sleep stage R': 4
}

def process_subject_files(psg_path, hyp_path):
    """Loads, filters, and extracts 30s epochs for one recording night."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Channels contain.*")
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Highpass cutoff.*")
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Limited.*annotation.*")

            raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
            annotations = mne.read_annotations(hyp_path)
            raw.set_annotations(annotations)

        raw.pick(picks=["EEG Fpz-Cz", "EEG Pz-Oz"])
        raw.filter(l_freq=0.5, h_freq=30.0, fir_design="firwin", verbose=False)

        events, event_id = mne.events_from_annotations(raw, chunk_duration=30.0, verbose=False)
        valid_event_ids = {k: v for k, v in event_id.items() if k in LABEL_MAPPING}

        if not valid_event_ids:
            return None, None

        epochs = mne.Epochs(
            raw, events, event_id=valid_event_ids,
            tmin=0, tmax=30.0 - (1 / raw.info['sfreq']),
            baseline=None, preload=True, verbose=False
        )

        x_data = epochs.get_data(copy=True)  # Shape: (N_epochs, 2, 3000)
        reversed_clean_id = {v: k for k, v in epochs.event_id.items()}

        y_labels = np.array([
            LABEL_MAPPING[reversed_clean_id[code]] for code in epochs.events[:, -1] if code in reversed_clean_id
        ])

        # Trim excessive wake periods
        sleep_indices = np.where(y_labels != 0)[0]
        first_sleep_idx = sleep_indices[0]
        last_sleep_idx = sleep_indices[-1]

        start_idx = max(0, first_sleep_idx - 60)            # 60 epochs of wake before ...
        end_idx = min(len(y_labels), last_sleep_idx + 61)   # ... and after

        x_data = x_data[start_idx:end_idx]
        y_labels = y_labels[start_idx:end_idx]

        # Per-channel normalization
        mean = np.mean(x_data, axis=(0, 2), keepdims=True)
        std = np.std(x_data, axis=(0, 2), keepdims=True)
        x_data = (x_data - mean) / (std + 1e-7)

        return x_data, y_labels
    except Exception as e:
        print(f"Error processing {os.path.basename(psg_path)}: {e}")
        return None, None


def save_split_data(output_dir, split_name, subject_list, subject_groups):
    """Processes subjects and saves individual files to prevent RAM crashes."""
    print(f"Starting {split_name.upper()} Split Extraction")

    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    pbar = tqdm(subject_list, desc=f"Processing {split_name} subjects")

    for sub_id in pbar:
        pbar.set_postfix_str(f"Sub: {sub_id}")

        sub_x = []
        sub_y = []

        for psg_file, hyp_file in subject_groups[sub_id]:
            x_data, y_labels = process_subject_files(psg_file, hyp_file)
            if x_data is not None:
                sub_x.append(x_data)
                sub_y.append(y_labels)

        if sub_x:
            final_x = np.concatenate(sub_x, axis=0).astype(np.float32)
            final_y = np.concatenate(sub_y, axis=0).astype(np.int64)

            # Save to disk
            out_x_path = os.path.join(split_dir, f"{sub_id}_x.npy")
            out_y_path = os.path.join(split_dir, f"{sub_id}_y.npy")

            np.save(out_x_path, final_x)
            np.save(out_y_path, final_y)

            del final_x, final_y

# --- Execution Directory Setup ---
DATA_DIR = "/media/santiago/T9/data/physionet/sleep-cassette"
OUTPUT_DIR = "/media/santiago/T9/data/physionet/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Find and group files by subject
psg_files = sorted(glob.glob(os.path.join(DATA_DIR, "*PSG.edf")))
hyp_files = sorted(glob.glob(os.path.join(DATA_DIR, "*Hypnogram.edf")))

subject_groups = {}
for p_file, h_file in zip(psg_files, hyp_files):
    sub_id = os.path.basename(p_file)[:6]
    if sub_id not in subject_groups:
        subject_groups[sub_id] = []
    subject_groups[sub_id].append((p_file, h_file))

unique_subjects = list(subject_groups.keys())
np.random.seed(42)
np.random.shuffle(unique_subjects)

# Split Subjects
n_sub = len(unique_subjects)
train_subs = unique_subjects[:int(n_sub * 0.7)]
val_subs = unique_subjects[int(n_sub * 0.7):int(n_sub * 0.85)]
test_subs = unique_subjects[int(n_sub * 0.85):]

# Run the saving pipeline
save_split_data(OUTPUT_DIR, "train", train_subs, subject_groups)
save_split_data(OUTPUT_DIR, "val", val_subs, subject_groups)
save_split_data(OUTPUT_DIR, "test", test_subs, subject_groups)
