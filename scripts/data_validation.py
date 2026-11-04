import os
import glob
import numpy as np

PROCESSED_DIR = "/media/santiago/T9/data/physionet/data"

def run_sanity_check(split_name):
    split_path = os.path.join(PROCESSED_DIR, split_name)
    x_files = sorted(glob.glob(os.path.join(split_path, "*_x.npy")))
    y_files = sorted(glob.glob(os.path.join(split_path, "*_y.npy")))

    if not x_files:
        print(f"Error: No data files found in the '{split_name}' directory!")
        return

    if len(x_files) != len(y_files):
        print(
            f"Mismatch in file counts! Found {len(x_files)} signal files and {len(y_files)} label files.")
        return

    print(f"\nLooking into {split_name.upper()} split ({len(x_files)} subjects)...")

    total_epochs = 0
    nan_found = False

    for x_f, y_f in zip(x_files, y_files):
        # Verify that the subject codes match across pairs
        sub_id_x = os.path.basename(x_f).replace("_x.npy", "")
        sub_id_y = os.path.basename(y_f).replace("_y.npy", "")
        if sub_id_x != sub_id_y:
            print(f"Mismatched subjects! {sub_id_x} does not line up with {sub_id_y}.")
            return

        x = np.load(x_f)
        y = np.load(y_f)

        # Shape should be [Num_Epochs, 2, 3000]
        if len(x.shape) != 3 or x.shape[1] != 2 or x.shape[2] != 3000:
            print(f"Dimension Error in {os.path.basename(x_f)}: Got shape {x.shape}")
            return

        # Signals & labels
        if x.shape[0] != y.shape[0]:
            print(
                f"X has {x.shape[0]} epochs but y has {y.shape[0]} labels.")
            return

        # Labels must be in [0, 4] (W, 1, 2, 3, R)
        unique_labels = np.unique(y)
        if not set(unique_labels).issubset({0, 1, 2, 3, 4}):
            print(f"Label Error in {os.path.basename(y_f)}: Found invalid sleep stage codes: {unique_labels}")
            return

        # No NaNs nor Inf
        if np.isnan(x).any() or np.isinf(x).any():
            nan_found = True
            print(f"Warning: NaNs or Infs detected in subject file: {os.path.basename(x_f)}")

        total_epochs += x.shape[0]

    print(f"{split_name.upper()} Split Passed All Tests!")
    print(f" -> Combined Epochs: {total_epochs}")
    print(f" -> Total 30s Sleep Time: {total_epochs * 30 / 3600:.2f} hours")
    if nan_found:
        print("   -> Check noted warning files above for invalid values before training.")


if __name__ == "__main__":
    run_sanity_check("train")
    run_sanity_check("val")
    run_sanity_check("test")
