import torch
import torch.nn as nn
import pandas as pd
import os
import gc
import winsound
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# Local imports
from dataset import MOSDataset, collate_fn
from model import WavLMTransformerMOS

# --- Stability Config ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# --- Configuration ---
BASE_DIR = "D:/BUU_PC_DATA/audio_mos_prediction_projectt/"
WAV_DIR = os.path.join(BASE_DIR, "data/wav_files/")

# Define the two experiment root folders
EXPERIMENT_ROOTS = [
    os.path.join(BASE_DIR, "experiments"),
    os.path.join(BASE_DIR, "experiments_only_english")
]

BATCH_SIZE = 2
ACCUMULATION_STEPS = 2  # Effective batch size = 4
LR = 1e-4
PATIENCE = 16
NUM_EPOCHS = 500


def play_notification_sound(sound_type="epoch"):
    try:
        if sound_type == "epoch":
            winsound.Beep(1000, 200);
            winsound.Beep(1200, 200)
        elif sound_type == "success":
            [winsound.Beep(f, 300) for f in [440, 554, 659, 880]]
    except:
        pass


def get_latest_checkpoint(ckpt_path, emergency_path):
    ckpt = torch.load(ckpt_path, map_location='cuda') if os.path.exists(ckpt_path) else None
    emergency = torch.load(emergency_path, map_location='cuda') if os.path.exists(emergency_path) else None
    if ckpt and emergency:
        if emergency.get('epoch', 0) > ckpt.get('epoch', 0): return emergency
        if emergency.get('epoch', 0) == ckpt.get('epoch', 0):
            return emergency if emergency.get('batch_idx', 0) > ckpt.get('batch_idx', 0) else ckpt
        return ckpt
    return ckpt if ckpt else emergency


def train_single_experiment(root_path, exp_name):
    CASE_DIR = os.path.join(root_path, exp_name)
    SAVE_DIR = os.path.join(CASE_DIR, "checkpoints")
    LOG_PATH = os.path.join(CASE_DIR, "training_log.csv")

    # --- AUTO-DETECT CSV NAMING ---
    # Case A: train_split.csv (Standard experiments)
    # Case B: train.csv (Only English experiments)
    if os.path.exists(os.path.join(CASE_DIR, "train_split.csv")):
        TRAIN_CSV = os.path.join(CASE_DIR, "train_split.csv")
        VAL_CSV = os.path.join(CASE_DIR, "val_split.csv")
    else:
        TRAIN_CSV = os.path.join(CASE_DIR, "train.csv")
        VAL_CSV = os.path.join(CASE_DIR, "val.csv")

    if not os.path.exists(TRAIN_CSV):
        print(f"⏩ Skipping {exp_name}: No CSV files found.")
        return

    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Model Setup
    model = WavLMTransformerMOS(freeze_wavlm=True).to(device)
    model.wavlm.feature_extractor._freeze_parameters()
    model.wavlm.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scaler = GradScaler('cuda')

    ckpt_path = os.path.join(SAVE_DIR, "resume_checkpoint.pth")
    emergency_path = os.path.join(SAVE_DIR, "emergency_backup.pth")
    latest_ckpt = get_latest_checkpoint(ckpt_path, emergency_path)

    start_epoch, start_batch, best_val_loss, history, wait = 0, 0, float('inf'), [], 0

    if latest_ckpt:
        model.load_state_dict(latest_ckpt['model_state_dict'])
        optimizer.load_state_dict(latest_ckpt['optimizer_state_dict'])
        start_epoch = latest_ckpt['epoch']
        start_batch = latest_ckpt.get('batch_idx', 0) + 1
        best_val_loss = latest_ckpt.get('best_val_loss', float('inf'))
        history = latest_ckpt.get('history', [])

        if history:
            val_losses = [h['val_loss'] for h in history]
            actual_best = min(val_losses)
            wait = 0
            for v in reversed(val_losses):
                if v > actual_best:
                    wait += 1
                else:
                    break
        print(f"🔄 Resuming {exp_name} | E{start_epoch} | Wait: {wait}/{PATIENCE}")

    if wait >= PATIENCE:
        print(f"🛑 {exp_name} already reached Early Stopping.")
        return

    train_df, val_df = pd.read_csv(TRAIN_CSV), pd.read_csv(VAL_CSV)
    train_loader = DataLoader(MOSDataset(train_df, WAV_DIR), batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(MOSDataset(val_df, WAV_DIR), batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=4, pin_memory=True)

    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            gc.collect();
            torch.cuda.empty_cache();
            torch.cuda.synchronize()

            model.train()
            running_loss = 0.0
            optimizer.zero_grad(set_to_none=True)
            loop = tqdm(train_loader, desc=f"{exp_name} | E{epoch}")

            for i, (inputs, labels) in enumerate(loop):
                if epoch == start_epoch and i < start_batch: continue
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                with autocast(device_type='cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels) / ACCUMULATION_STEPS

                scaler.scale(loss).backward()

                if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if (i + 1) % 100 == 0:
                        torch.cuda.empty_cache();
                        gc.collect()

                running_loss += (loss.detach().item() * ACCUMULATION_STEPS)
                if i % 50 == 0: loop.set_postfix(loss=(loss.item() * ACCUMULATION_STEPS))

                if i > 0 and i % 5000 == 0:
                    torch.save({'epoch': epoch, 'batch_idx': i, 'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(), 'wait': wait,
                                'best_val_loss': best_val_loss, 'history': history}, emergency_path)
                del inputs, labels, outputs, loss

            avg_train = running_loss / len(train_loader)
            start_batch = 0

            # Validation Cleanup
            gc.collect();
            torch.cuda.empty_cache();
            torch.cuda.synchronize()
            model.eval()
            total_val_loss = 0.0
            with torch.inference_mode():
                for v_idx, (inputs, labels) in enumerate(tqdm(val_loader, desc="Validating")):
                    inputs, labels = inputs.to(device), labels.to(device)
                    with autocast(device_type='cuda'):
                        preds = model(inputs)
                        v_loss = criterion(preds, labels)
                        total_val_loss += v_loss.item()
                    del preds, v_loss, inputs, labels
                    if (v_idx + 1) % 50 == 0: torch.cuda.empty_cache()

            avg_val = total_val_loss / len(val_loader)
            print(f"✅ {exp_name} E{epoch} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | Best: {best_val_loss:.4f}")
            play_notification_sound("epoch")

            if avg_val < best_val_loss:
                best_val_loss, wait = avg_val, 0
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
                print("🌟 New Best Model Saved!")
            else:
                wait += 1
                print(f"⚠️ Patience: {wait}/{PATIENCE}")

            history.append({'epoch': epoch, 'train_loss': avg_train, 'val_loss': avg_val, 'wait': wait})
            pd.DataFrame(history).to_csv(LOG_PATH, index=False)

            torch.save({'epoch': epoch + 1, 'batch_idx': 0, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'best_val_loss': best_val_loss,
                        'wait': wait, 'history': history}, ckpt_path)

            if wait >= PATIENCE:
                print(f"🛑 Early Stopping triggered for {exp_name}.")
                break

    except Exception as e:
        print(f"❌ Error in {exp_name}: {e}")
    finally:
        # Clear VRAM before moving to the next experiment case
        del model, optimizer, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    for root in EXPERIMENT_ROOTS:
        if not os.path.exists(root):
            print(f"⚠️ Root folder not found: {root}")
            continue

        print(f"\n--- Processing Root: {os.path.basename(root)} ---")
        # Get all subdirectories (experiment cases)
        cases = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

        for case in cases:
            train_single_experiment(root, case)

    print("\n🎉 ALL EXPERIMENTS FROM ALL DIRECTORIES FINISHED.")
    play_notification_sound("success")