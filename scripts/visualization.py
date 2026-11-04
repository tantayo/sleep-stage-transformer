import os
import glob
import mne

DATA_DIR = "/media/santiago/T9/data/physionet/sleep-cassette"

psg_files = sorted(glob.glob(os.path.join(DATA_DIR, "*PSG.edf")))
hyp_files = sorted(glob.glob(os.path.join(DATA_DIR, "*Hypnogram.edf")))

if not psg_files or not hyp_files:
    raise FileNotFoundError(
        f"Could not find .edf files in '{DATA_DIR}'. "
        "Make sure the path is correct and points to the folder containing the raw files."
    )

# Pick the first available subject
raw_path = psg_files[0]
annotation_path = hyp_files[0]

print(f"Loading local data:\n -> EEG: {raw_path}\n -> Labels: {annotation_path}")

raw = mne.io.read_raw_edf(raw_path, preload=True)

annotations = mne.read_annotations(annotation_path)
raw.set_annotations(annotations)

target_channels = ["EEG Fpz-Cz", "EEG Pz-Oz"]

available_channels = raw.ch_names
picked_channels = [ch for ch in target_channels if ch in available_channels]

if len(picked_channels) < 2:
    print(f"Warning: Couldn't find exact matches. Available names: {available_channels}")
    picked_channels = [ch for ch in available_channels if "Fpz" in ch or "Pz" in ch][:2]

raw.pick(picks=picked_channels)

# Standard bandpass filter (0.5 to 30 Hz)
raw.filter(l_freq=0.5, h_freq=30.0, fir_design="firwin")

# Visualize a 30-second window
start_time = 3600  # in seconds
duration = 30      # in seconds

raw.plot(
    start=start_time,
    duration=duration,
    n_channels=len(picked_channels),
    scalings={"eeg": 100e-6},  # Set scale to 100 microvolts
    title=f"Local Sleep Data: {os.path.basename(raw_path)}",
    show=True,
)