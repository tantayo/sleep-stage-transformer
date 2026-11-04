import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.dataset import SleepDataset
from src.models.sleep_vit import FullyConvSleepViT


def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    PROCESSED_DATA_DIR = "/media/santiago/T9/data/physionet/data"

    # Evaluate strictly on the test set for true generalization metrics
    test_dataset = SleepDataset(os.path.join(PROCESSED_DATA_DIR, "test"))
    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # Initialize model and load best checkpoint
    model = FullyConvSleepViT(num_channels=2, d_model=128, nhead=4, num_layers=4, num_classes=5)
    checkpoint_path = "checkpoints/best_phase1_sleep_vit.pt"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'. Ensure pretraining has finished.")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    print("Running inference on test dataset...")
    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            outputs = model(signals)

            _, predicted = torch.max(outputs, dim=1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate overall metrics
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average='macro')
    class_names = ["Wake (W)", "N1", "N2", "N3", "REM"]

    print("\n==============================================")
    print(f"TEST RESULTS SUMMARY")
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
    plt.title('Sleep Stage Classification - Test Confusion Matrix', fontsize=12, pad=12)
    plt.tight_layout()

    report_str = classification_report(all_targets, all_preds, target_names=class_names, digits=4)
    os.makedirs("outputs/phase1", exist_ok=True)
    with open("outputs/phase1/metrics_summary.txt", "w") as f:
        f.write(f"Phase 1 Backbone Evaluation\n")
        f.write(f"Overall Accuracy: {acc * 100:.2f}%\n")
        f.write(f"Macro F1-Score:   {macro_f1:.4f}\n\n")
        f.write(report_str)

    plot_path = os.path.join("outputs", "phase1", "confusion_matrix.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\n💾 Confusion matrix plot successfully saved to '{plot_path}'")


if __name__ == "__main__":
    evaluate_model()