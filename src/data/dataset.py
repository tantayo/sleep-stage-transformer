import os
import glob
import bisect
import numpy as np
import torch
from torch.utils.data import Dataset

class SleepDataset(Dataset):
    def __init__(self, split_dir):
        self.split_dir = split_dir

        self.x_files = sorted(glob.glob(os.path.join(split_dir, "*_x.npy")))
        self.y_files = sorted(glob.glob(os.path.join(split_dir, "*_y.npy")))

        if not self.x_files:
            raise FileNotFoundError(f"No files found inside folder: '{split_dir}'")

        assert len(self.x_files) == len(self.y_files), "Mismatch between X and Y file counts!"

        self.epoch_offsets = []
        self.file_shapes = []
        total_epochs = 0

        print(f"Indexing subject metadata from {split_dir}...")

        for x_f, y_f in zip(self.x_files, self.y_files):
            y_shape = np.load(y_f, mmap_mode='r').shape
            num_epochs = y_shape[0]

            self.epoch_offsets.append(total_epochs)
            self.file_shapes.append(num_epochs)
            total_epochs += num_epochs

        self.total_epochs = total_epochs
        print(f"Indexed {len(self.x_files)} subjects. Total global epochs: {self.total_epochs}")

    def __len__(self):
        return self.total_epochs

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_epochs:
            raise IndexError(f"Index {idx} out of range for dataset size {self.total_epochs}")

        # Fast binary search using standard library bisect
        file_idx = bisect.bisect_right(self.epoch_offsets, idx) - 1
        local_idx = idx - self.epoch_offsets[file_idx]

        # Open mmap on-demand to avoid descriptor limits across parallel workers
        x_path = self.x_files[file_idx]
        y_path = self.y_files[file_idx]

        mmap_x = np.load(x_path, mmap_mode='r')
        mmap_y = np.load(y_path, mmap_mode='r')

        raw_x = mmap_x[local_idx]
        raw_y = mmap_y[local_idx]

        # Use copy() to prevent negative stride / layout warnings when converting to PyTorch Tensor
        x_tensor = torch.from_numpy(np.array(raw_x, copy=True)).float()
        y_tensor = torch.tensor(raw_y, dtype=torch.long)

        return x_tensor, y_tensor


class SleepSequenceDataset(Dataset):
    def __init__(self, data_dir, seq_len=21):
        self.seq_len = seq_len
        self.samples = []  # Stores metadata for valid sliding windows

        # Scan data directory for your exact naming convention: {sub_id}_x.npy
        x_files = sorted([f for f in os.listdir(data_dir) if f.endswith("_x.npy")])

        for x_file in x_files:
            sub_id = x_file.replace("_x.npy", "")
            x_path = os.path.join(data_dir, f"{sub_id}_x.npy")
            y_path = os.path.join(data_dir, f"{sub_id}_y.npy")

            # Peek at the shape using memmap header to get total epochs without loading RAM
            signals_memmap = np.load(x_path, mmap_mode='r')
            num_epochs = signals_memmap.shape[0]

            # Ensure the subject recording is long enough to form at least one full window
            if num_epochs >= self.seq_len:
                max_start = num_epochs - self.seq_len
                for start_idx in range(max_start + 1):
                    self.samples.append({
                        'x_path': x_path,
                        'y_path': y_path,
                        'start_idx': start_idx
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        meta = self.samples[idx]
        x_path = meta['x_path']
        y_path = meta['y_path']
        start = meta['start_idx']
        end = start + self.seq_len

        # Stream the 21-epoch window directly from disk via memmap
        signals_mmap = np.load(x_path, mmap_mode='r')
        labels_mmap = np.load(y_path, mmap_mode='r')

        window_signals = signals_mmap[start:end]  # Shape: [21, 2, 3000]
        window_labels = labels_mmap[start:end]  # Shape: [21]

        return (
            torch.tensor(window_signals, dtype=torch.float32),
            torch.tensor(window_labels, dtype=torch.long)
        )