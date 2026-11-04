import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.dataset import SleepSequenceDataset
from src.models.temporal_wrapper import SleepTemporalWrapper


def evaluate_phase2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating Phase 2 on device: {device}")

    PROCESSED_DATA_DIR = "/media/santiago/T9/data/physionet/data"

    # Load test sequence dataset
    test_dataset = SleepSequenceDataset(os.path.join(PROCESSED_DATA_DIR, "test"), seq_len=21)
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # Initialize Phase 2 model and load best checkpoint
    checkpoint_path = "checkpoints/best_phase2_sleep_transformer.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'. Ensure Phase 2 training has finished.")

    # Backbone path is required for architecture initialization before loading wrapper weights
    backbone_ckpt = "checkpoints/best_phase1_sleep_vit.pt"
    model = SleepTemporalWrapper(
        backbone_checkpoint_path=backbone_ckpt,
        d_model=128, nhead=4, num_layers=2, num_classes=5
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    print("Running Phase 2 inference on test dataset...")
    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            outputs = model(signals)  # Shape: [Batch, 21, 5]

            # Evaluate the center epoch (index 10) of each sliding window
            # to maintain clean 1-to-1 epoch evaluation without overlap duplication.
            center_idx = outputs.shape[1] // 2
            preds_center = outputs[:, center_idx, :]
            labels_center = labels[:, center_idx]

            _, predicted = torch.max(preds_center, dim=1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels_center.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate overall metrics
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average='macro')
    class_names = ["Wake (W)", "N1", "N2", "N3", "REM"]

    print("\n==============================================")
    print(f"PHASE 2 TEST RESULTS SUMMARY")
    print(f"Overall Accuracy: {acc * 100:.2f}%")
    print(f"Macro F1-Score:   {macro_f1:.4f}")
    print("==============================================\n")

    print("Detailed Classification Report:")
    print(classification_report(all_targets, all_preds, target_names=class_names, digits=4))

    # Compute Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    print("Raw Confusion Matrix (Rows: True, Columns: Predicted):")
    print(cm)

    # Plot and save Confusion Matrix heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names
    )
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.title('Phase 2 Temporal Wrapper - Test Confusion Matrix', fontsize=12, pad=12)
    plt.tight_layout()

    os.makedirs("outputs/phase2", exist_ok=True)
    report_str = classification_report(all_targets, all_preds, target_names=class_names, digits=4)
    with open("outputs/phase2/metrics_summary.txt", "w") as f:
        f.write(f"Phase 2 Temporal Wrapper Evaluation\n")
        f.write(f"Overall Accuracy: {acc * 100:.2f}%\n")
        f.write(f"Macro F1-Score:   {macro_f1:.4f}\n\n")
        f.write(report_str)

    plot_path = os.path.join("outputs", "phase2", "confusion_matrix.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\n💾 Phase 2 confusion matrix plot successfully saved to '{plot_path}'")


if __name__ == "__main__":
    evaluate_phase2()