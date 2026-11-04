import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter

from src.data.dataset import SleepDataset
from src.models.sleep_vit import FullyConvSleepViT


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    """Executes a single training pass over the entire split dataset."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for signals, labels in dataloader:
        signals, labels = signals.to(device), labels.to(device)

        outputs = model(signals)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()

        # Clip gradients to prevent early Transformer instability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Step the scheduler PER BATCH, not per epoch
        scheduler.step()

        running_loss += loss.item() * signals.size(0)

        _, predicted = torch.max(outputs, dim=1)
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(all_targets)
    epoch_f1 = f1_score(all_targets, all_preds, average='macro')
    return epoch_loss, epoch_f1


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Evaluates the model on unseen subjects to check true generalization."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for signals, labels in dataloader:
        signals, labels = signals.to(device), labels.to(device)

        outputs = model(signals)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * signals.size(0)

        _, predicted = torch.max(outputs, dim=1)
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(all_targets)
    epoch_f1 = f1_score(all_targets, all_preds, average='macro')
    return epoch_loss, epoch_f1


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Computing Device: {device}")

    PROCESSED_DATA_DIR = "/media/santiago/T9/data/physionet/data"

    # Optimized DataLoaders for Local Hardware
    train_dataset = SleepDataset(os.path.join(PROCESSED_DATA_DIR, "train"))
    val_dataset = SleepDataset(os.path.join(PROCESSED_DATA_DIR, "val"))

    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True,
        drop_last=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # Initialize the patched model (ensure your FullyConvSleepViT includes the CLS token update)
    model = FullyConvSleepViT(num_channels=2, d_model=128, nhead=4, num_layers=4, num_classes=5)
    model = model.to(device)

    print("\n==============================================")
    print("MODEL SUMMARY")
    print("==============================================")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")
    print("==============================================\n")

    # Weighted loss for PhysioNet distribution
    class_weights = torch.tensor([1.0, 3.5, 0.7, 1.2, 1.2], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    NUM_EPOCHS = 20
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # OneCycleLR natively handles the warmup slope + cosine decay
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-4,  # Peak learning rate after warmup
        epochs=NUM_EPOCHS,  # Total epochs
        steps_per_epoch=len(train_loader),  # Needs to know total batches per epoch
        pct_start=0.1  # Spends the first 10% of training warming up
    )

    print(f"Running Phase 1 Pretraining for {NUM_EPOCHS} epochs...\n")

    writer = SummaryWriter(log_dir="runs/sleep_vit_phase1")

    best_val_f1 = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        val_loss, val_f1 = validate(model, val_loader, criterion, device)

        # Log metrics to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Macro_F1/Train', train_f1, epoch)
        writer.add_scalar('Macro_F1/Validation', val_f1, epoch)

        # Log the current learning rate to verify OneCycleLR is curving correctly
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_Rate', current_lr, epoch)

        # Step the scheduler at the end of the epoch
        # scheduler.step()

        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS}] "
              f"| Train Loss: {train_loss:.4f} - Macro F1: {train_f1:.4f} "
              f"| Val Loss: {val_loss:.4f} - Macro F1: {val_f1:.4f}")

        # Save checkpoints strictly based on Macro F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            checkpoint_path = "checkpoints/best_phase1_sleep_vit.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"   💾 Saved new best validation model checkpoint: {val_f1:.4f}")

    print(f"\nPhase 1 Training finalized! Best Val Macro F1: {best_val_f1:.4f}")
    writer.close()