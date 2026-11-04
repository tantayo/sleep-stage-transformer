import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter

from src.data.dataset import SleepSequenceDataset
from src.models.temporal_wrapper import SleepTemporalWrapper


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    """Executes a single training pass over sequence windows."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for signals, labels in dataloader:
        # signals shape: [Batch, 21, 2, 3000]
        # labels shape: [Batch, 21]
        signals, labels = signals.to(device), labels.to(device)

        outputs = model(signals)  # Shape: [Batch, 21, 5]

        # Flatten sequence and batch dimensions to compute cross-entropy loss
        batch_size, seq_len, num_classes = outputs.shape
        outputs_flat = outputs.view(-1, num_classes)
        labels_flat = labels.view(-1)

        loss = criterion(outputs_flat, labels_flat)

        optimizer.zero_grad()
        loss.backward()

        # Clip gradients to ensure temporal transformer stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * labels_flat.size(0)

        _, predicted = torch.max(outputs_flat, dim=1)
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(labels_flat.cpu().numpy())

    epoch_loss = running_loss / len(all_targets)
    epoch_f1 = f1_score(all_targets, all_preds, average='macro')
    return epoch_loss, epoch_f1


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Evaluates the temporal wrapper on unseen subjects."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for signals, labels in dataloader:
        signals, labels = signals.to(device), labels.to(device)

        outputs = model(signals)

        batch_size, seq_len, num_classes = outputs.shape
        outputs_flat = outputs.view(-1, num_classes)
        labels_flat = labels.view(-1)

        loss = criterion(outputs_flat, labels_flat)

        running_loss += loss.item() * labels_flat.size(0)

        _, predicted = torch.max(outputs_flat, dim=1)
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(labels_flat.cpu().numpy())

    epoch_loss = running_loss / len(all_targets)
    epoch_f1 = f1_score(all_targets, all_preds, average='macro')
    return epoch_loss, epoch_f1


if __name__ == "__main__":
    device = torch.device("cuda" if str(torch.device("cuda") if torch.cuda.is_available() else "cpu") else "cpu")
    print(f"Target Computing Device: {device}")

    PROCESSED_DATA_DIR = "/media/santiago/T9/data/physionet/data"

    # Load sequence datasets (sliding windows of 21 epochs)
    print("Initializing sequence datasets...")
    train_dataset = SleepSequenceDataset(os.path.join(PROCESSED_DATA_DIR, "train"), seq_len=21)
    val_dataset = SleepSequenceDataset(os.path.join(PROCESSED_DATA_DIR, "val"), seq_len=21)

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True,
        drop_last=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False,
        num_workers=4, pin_memory=True
    )

    print(f"Train windows: {len(train_dataset):,} | Val windows: {len(val_dataset):,}")

    # Initialize Phase 2 Temporal Wrapper (loads frozen Phase 1 backbone automatically)
    checkpoint_path = "checkpoints/best_phase1_sleep_vit.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Phase 1 checkpoint not found at '{checkpoint_path}'.")

    model = SleepTemporalWrapper(
        backbone_checkpoint_path=checkpoint_path,
        d_model=128, nhead=4, num_layers=2, num_classes=5
    )
    model = model.to(device)

    print("\n==============================================")
    print("PHASE 2 MODEL SUMMARY (Temporal Wrapper)")
    print("==============================================")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable Parameters (Temporal Transformer + Head): {trainable_params:,}")
    print(f"Frozen Parameters (Phase 1 Backbone):                 {frozen_params:,}")
    print("==============================================\n")

    # Weighted loss for PhysioNet distribution
    class_weights = torch.tensor([1.0, 3.5, 0.7, 1.2, 1.2], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer and Scheduler (smaller learning rate since backbone is frozen)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=1e-4
    )

    NUM_EPOCHS = 15
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-4,
        epochs=NUM_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )

    print(f"Running Phase 2 Sequence Training for {NUM_EPOCHS} epochs...\n")

    os.makedirs("runs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    writer = SummaryWriter(log_dir="runs/sleep_vit_phase2")

    best_val_f1 = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        val_loss, val_f1 = validate(model, val_loader, criterion, device)

        # Log metrics to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Macro_F1/Train', train_f1, epoch)
        writer.add_scalar('Macro_F1/Validation', val_f1, epoch)

        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_Rate', current_lr, epoch)

        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS}] "
              f"| Train Loss: {train_loss:.4f} - Macro F1: {train_f1:.4f} "
              f"| Val Loss: {val_loss:.4f} - Macro F1: {val_f1:.4f}")

        # Save checkpoint strictly based on validation Macro F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            checkpoint_path = "checkpoints/best_phase2_sleep_transformer.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"   💾 Saved new best Phase 2 checkpoint: {val_f1:.4f}")

    print(f"\nPhase 2 Training finalized! Best Val Macro F1: {best_val_f1:.4f}")
    writer.close()